import argparse
import json
from pathlib import Path

import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import mlflow
    import mlflow.sklearn
except Exception:
    mlflow = None

from src.data_loader import load_sp500_data, train_val_test_split
from src.features import prepare_features


def prepare_data(lags: int = 5, windows=(3, 7, 14)):
    df = load_sp500_data()
    feat = prepare_features(df, target_col='close', lags=lags, windows=windows)
    train, val, test = train_val_test_split(feat, train_size=0.7, val_size=0.15, test_size=0.15)
    X_train = train.drop(columns=['close'])
    y_train = train['close']
    X_val = val.drop(columns=['close'])
    y_val = val['close']

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)

    return X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled, scaler


def run_optuna_study(model_type: str = 'xgboost', n_trials: int = 20, output_dir: str = 'outputs/experiments', use_mlflow: bool = True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled, scaler = prepare_data()

    def objective(trial: optuna.Trial):
        if model_type == 'xgboost':
            if xgb is None:
                raise RuntimeError('xgboost not installed')
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'verbosity': 0,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            rmse = mean_squared_error(y_val, preds, squared=False)
            trial.set_user_attr('params', params)
            return rmse

        elif model_type == 'mlp':
            hidden1 = trial.suggest_int('hidden1', 16, 256)
            hidden2 = trial.suggest_int('hidden2', 8, 128)
            lr = trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1)
            params = {'hidden_layer_sizes': (hidden1, hidden2), 'learning_rate_init': lr}
            model = MLPRegressor(hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'], max_iter=500)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
            rmse = mean_squared_error(y_val, preds, squared=False)
            trial.set_user_attr('params', params)
            return rmse

        else:
            raise ValueError('Unsupported model_type')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    result = {'best_value': best.value, 'best_params': best.params}
    (Path(output_dir) / 'study_result.json').write_text(json.dumps(result, indent=2))

    # Refit best model on train+val and save
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    if model_type == 'xgboost':
        params = {k: best.params[k] for k in best.params}
        # ensure types
        params['n_estimators'] = int(params.get('n_estimators', 100))
        params['max_depth'] = int(params.get('max_depth', 6))
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_full, y_train_full)
        model_path = Path(output_dir) / f'xgb_best.joblib'
        if mlflow and use_mlflow:
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metric('best_rmse', float(best.value))
                mlflow.sklearn.log_model(model, 'model')
        else:
            import joblib

            joblib.dump(model, model_path)

    elif model_type == 'mlp':
        params = {'hidden_layer_sizes': (int(best.params['hidden1']), int(best.params['hidden2'])), 'learning_rate_init': float(best.params['learning_rate_init'])}
        model = MLPRegressor(hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'], max_iter=1000)
        X_full_scaled = pd.DataFrame(scaler.fit_transform(X_train_full), index=X_train_full.index, columns=X_train_full.columns)
        model.fit(X_full_scaled, y_train_full)
        model_path = Path(output_dir) / f'mlp_best.joblib'
        if mlflow and use_mlflow:
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metric('best_rmse', float(best.value))
                mlflow.sklearn.log_model(model, 'model')
        else:
            import joblib

            joblib.dump({'model': model, 'scaler': scaler}, model_path)

    return study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['xgboost', 'mlp'], default='xgboost')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--out', default='outputs/experiments')
    parser.add_argument('--no-mlflow', dest='mlflow', action='store_false')
    args = parser.parse_args()
    study = run_optuna_study(model_type=args.model, n_trials=args.trials, output_dir=args.out, use_mlflow=args.mlflow)
    print('Study completed. Best value:', study.best_value)


if __name__ == '__main__':
    main()
