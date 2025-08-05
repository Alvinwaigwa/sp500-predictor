import pandas as pd
import numpy as np

def backtest(close_prices, signals):
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.squeeze()
    returns = close_prices.pct_change()
    strategy_returns = signals.shift(1) * returns  # avoid lookahead bias
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    equity_curve = (1 + strategy_returns).cumprod()
    max_drawdown = (equity_curve.cummax() - equity_curve).max()
    return {
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'equity_curve': equity_curve
    }