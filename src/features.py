import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple


def create_lag_features(df: pd.DataFrame, col: str = 'close', lags: int = 5) -> pd.DataFrame:
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, col: str = 'close', windows=(3, 7, 14)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f'{col}_ma_{w}'] = df[col].rolling(w).mean()
        df[f'{col}_std_{w}'] = df[col].rolling(w).std()
    return df


def prepare_features(df: pd.DataFrame, target_col: str = 'close', lags: int = 5, windows=(3, 7, 14)) -> pd.DataFrame:
    df = create_lag_features(df, col=target_col, lags=lags)
    df = create_rolling_features(df, col=target_col, windows=windows)
    df = df.dropna()
    return df


def fit_scaler(X: pd.DataFrame, scaler_path: str = None) -> Tuple[StandardScaler, pd.DataFrame]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return scaler, X_scaled


def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)
