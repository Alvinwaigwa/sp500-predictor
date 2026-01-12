import pandas as pd
import numpy as np
from typing import Dict, Iterator, Tuple


def run_backtest(data: pd.DataFrame, commission: float = 0.0005) -> Dict:
    """
    Run vectorized backtest with transaction costs.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain `close` and `position` columns. `position` should be 1 for long, 0 for flat, -1 for short.
    commission : float
        Per-trade commission rate.

    Returns
    -------
    Dict
        Performance metrics, equity curve, and enriched backtest data.
    """
    returns = data['close'].pct_change()
    strategy_returns = data['position'] * returns

    # Account for transaction costs (apply when position changes)
    trades = data['position'].diff().abs().fillna(0)
    strategy_returns = strategy_returns - (trades * commission)

    equity_curve = (1 + strategy_returns.fillna(0)).cumprod()
    daily_returns = strategy_returns.dropna()

    sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-12)
    max_dd = (equity_curve / equity_curve.cummax() - 1).min()
    sortino = np.sqrt(252) * daily_returns.mean() / (daily_returns[daily_returns < 0].std() + 1e-12)

    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'total_return': float(equity_curve.iloc[-1] - 1),
        'win_rate': float((strategy_returns[trades > 0] > 0).mean()) if trades.sum() > 0 else 0.0,
        'num_trades': int(trades.sum()),
        'equity_curve': equity_curve,
        'data': data.assign(strategy_returns=strategy_returns, trades=trades)
    }


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> Dict:
    """Return regression metrics commonly used for forecasting tasks."""
    y_true = y_true.dropna()
    y_pred = y_pred.reindex_like(y_true)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100
    return {'mae': float(mae), 'rmse': float(rmse), 'mape_pct': float(mape)}


def backtest_from_predictions(prices: pd.Series, predictions: pd.Series, commission: float = 0.0005) -> Dict:
    """
    Convert point forecasts into simple long/flat signals and run a backtest.

    Signal rule (default): go long if predicted next-step price > current price.
    Predictions should be aligned to the same index as `prices` and represent forecasted prices for that same index (e.g., t+1 forecasts placed at t).
    """
    prices = prices.sort_index()
    preds = predictions.reindex(prices.index)

    # Simple signal: predict price will rise -> go long
    signal = (preds > prices).astype(int)
    position = signal.shift(1).fillna(0)  # avoid lookahead

    df = pd.DataFrame({'close': prices, 'prediction': preds, 'signal': signal, 'position': position})
    return run_backtest(df, commission=commission)


def walk_forward_splits(n_samples: int, initial_train_size: int, test_size: int, step: int) -> Iterator[Tuple[slice, slice]]:
    """
    Generate rolling walk-forward train/test slices.

    Parameters
    ----------
    n_samples: int
        Total number of samples in the time-series.
    initial_train_size: int
        Number of samples used for the first training window.
    test_size: int
        Number of samples to evaluate on each step.
    step: int
        How many samples to move the window forward each iteration.
    """
    start = initial_train_size
    while start + test_size <= n_samples:
        train_slice = slice(0, start)
        test_slice = slice(start, start + test_size)
        yield train_slice, test_slice
        start += step


def walk_forward_backtest(df: pd.DataFrame, price_col: str = 'close', predictions_col: str = 'prediction', initial_train_size: int = 500, test_size: int = 50, step: int = 50, commission: float = 0.0005) -> Dict:
    """
    Run a walk-forward backtest over a dataframe that contains price and model predictions.

    Returns aggregated metrics and per-fold results.
    """
    n = len(df)
    results = []
    for train_slice, test_slice in walk_forward_splits(n, initial_train_size, test_size, step):
        prices_test = df[price_col].iloc[test_slice]
        preds_test = df[predictions_col].iloc[test_slice]
        res = backtest_from_predictions(prices_test, preds_test, commission=commission)
        reg_metrics = evaluate_regression(prices_test, preds_test)
        results.append({'backtest': res, 'regression': reg_metrics})

    # Aggregate metrics
    sharpe_vals = [r['backtest']['sharpe'] for r in results]
    avg_sharpe = float(np.nanmean(sharpe_vals)) if sharpe_vals else 0.0

    return {'folds': results, 'avg_sharpe': avg_sharpe, 'n_folds': len(results)}