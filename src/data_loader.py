import yfinance as yf
import pandas as pd

def load_sp500_data():
    df = yf.download("^GSPC", start="2020-01-01", end="2023-01-01", progress=False)
    return df[["Close"]]