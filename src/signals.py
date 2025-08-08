import numpy as np
import pandas as pd

def generate_signals(data: pd.DataFrame, 
                   rsi_col: str = 'rsi',
                   lower: float = 30, 
                   upper: float = 70) -> pd.DataFrame:
    """
    Generate trading signals based on RSI thresholds
    
    Parameters:
        data (pd.DataFrame): Must contain RSI column
        rsi_col (str): Column name with RSI values
        lower (float): Buy threshold (30 = oversold)
        upper (float): Sell threshold (70 = overbought)
    
    Returns:
        pd.DataFrame: Data with signals and positions
    """
    df = data.copy()
    df['signal'] = 0
    df.loc[df[rsi_col] < lower, 'signal'] = 1  # Buy signal
    df.loc[df[rsi_col] > upper, 'signal'] = -1  # Sell signal
    
    # Forward fill positions (hold until next signal)
    df['position'] = df['signal'].replace(0, method='ffill').shift(1)
    return df.dropna()