import pandas as pd
import numpy as np

def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) using Wilder's EMA method
    
    Parameters:
        close_prices (pd.Series): Daily closing prices
        window (int): Lookback period
    
    Returns:
        pd.Series: RSI values
    """
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))