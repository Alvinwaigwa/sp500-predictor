import pandas as pd
import numpy as np
from typing import Dict

def run_backtest(data: pd.DataFrame,
                commission: float = 0.0005) -> Dict:
    """
    Run vectorized backtest with transaction costs
    
    Parameters:
        data (pd.DataFrame): Must contain 'close' and 'position'
        commission (float): Per-trade commission rate
    
    Returns:
        Dict: Performance metrics and equity curve
    """
    returns = data['close'].pct_change()
    strategy_returns = data['position'] * returns
    
    # Account for transaction costs
    trades = data['position'].diff().abs()
    strategy_returns = strategy_returns - (trades * commission)
    
    # Calculate metrics
    equity_curve = (1 + strategy_returns).cumprod()
    daily_returns = strategy_returns.dropna()
    
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    max_dd = (equity_curve / equity_curve.cummax() - 1).min()
    sortino = np.sqrt(252) * daily_returns.mean() / daily_returns[daily_returns < 0].std()
    
    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'total_return': equity_curve.iloc[-1] - 1,
        'win_rate': (strategy_returns[trades > 0] > 0).mean(),
        'num_trades': trades.sum(),
        'equity_curve': equity_curve,
        'data': data
    }