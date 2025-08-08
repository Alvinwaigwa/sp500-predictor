import yfinance as yf
import pandas as pd
import warnings

def load_sp500_data(start_date="2020-01-01", end_date="2023-01-01"):
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


