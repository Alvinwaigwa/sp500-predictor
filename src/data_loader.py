import yfinance as yf
import pandas as pd
import warnings
import os
import pickle
from typing import Tuple


def load_sp500_data(start_date: str = "2020-01-01", end_date: str = "2023-01-01") -> pd.DataFrame:
    """
    Download S&P500 daily OHLC data and return a single-column DataFrame with `close`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError("No data returned - check date range")

    # Flatten multiindex columns to single-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip().lower() for col in df.columns.values]
    else:
        df.columns = df.columns.str.lower()

    # Find the close column with the ticker suffix, e.g., close_^gspc
    close_cols = [col for col in df.columns if col.startswith('close')]
    if not close_cols:
        raise ValueError(f"No 'close' column found in columns: {df.columns}")

    # Select the first matching close column
    close_col = close_cols[0]

    # Return a DataFrame with only that close column, renamed to 'close' for simplicity
    return df[[close_col]].rename(columns={close_col: 'close'})


def load_cached_data(cache_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load cached dataframe if exists and matches date range.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(cache_path)

    with open(cache_path, 'rb') as f:
        obj = pickle.load(f)

    if not isinstance(obj, pd.DataFrame):
        raise ValueError("Cache does not contain a DataFrame")

    df = obj
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df


def train_val_test_split(df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15, shuffle: bool = False, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic time-series aware train/val/test split.

    For time series we typically do not shuffle; set `shuffle=True` for cross-sectional experiments.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train/val/test sizes must sum to 1")

    n = len(df)
    if shuffle:
        df = df.sample(frac=1, random_state=seed)

    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


