import numpy as np
import pandas as pd

def rsi_strategy(df, rsi_col="RSI", lower=30, upper=70):
    df = df.copy()
    df["Signals"] = 0
    df.loc[df[rsi_col] < lower, "Signals"] = 1   # Buy signal
    df.loc[df[rsi_col] > upper, "Signals"] = -1  # Sell signal
    return df

