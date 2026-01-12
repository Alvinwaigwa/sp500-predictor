from typing import Tuple
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


from .utils import save_model, load_model


def _prepare_xy(df: pd.DataFrame, target_col: str = 'close'):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def train_xgboost(df: pd.DataFrame, target_col: str = 'close', params: dict = None, num_boost_round: int = 100):
    try:
        import xgboost as xgb
    except Exception as e:
        raise ImportError('xgboost is required for train_xgboost') from e

    X, y = _prepare_xy(df, target_col)
    params = params or {'objective': 'reg:squarederror', 'verbosity': 0}
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    preds = model.predict(dtrain)
    mse = mean_squared_error(y, preds)
    return model, {'mse': float(mse)}


def train_mlp(df: pd.DataFrame, target_col: str = 'close', hidden_layer_sizes=(64, 32), random_state: int = 0) -> Tuple[MLPRegressor, dict]:
    X, y = _prepare_xy(df, target_col)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return model, {'mse': float(mse)}


def predict_with_model(model, X: pd.DataFrame) -> pd.Series:
    if hasattr(model, 'predict'):
        preds = model.predict(X)
        return pd.Series(preds, index=X.index)
    # xgboost Booster
    try:
        import xgboost as xgb

        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat)
        return pd.Series(preds, index=X.index)
    except Exception:
        raise ValueError('Unsupported model type for prediction')
