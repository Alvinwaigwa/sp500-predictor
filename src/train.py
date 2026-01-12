import os
import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_sp500_data, train_val_test_split
from src.features import prepare_features, fit_scaler
from src.models.baselines import train_xgboost, train_mlp, predict_with_model
from src.models.utils import save_model


def main(output_dir: str = 'outputs'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models_dir = Path(output_dir) / 'models'
    models_dir.mkdir(exist_ok=True)

    # Load data
    df = load_sp500_data()

    # Feature engineering
    feat = prepare_features(df, target_col='close', lags=5, windows=(3,7,14))

    # Split
    train, val, test = train_val_test_split(feat, train_size=0.7, val_size=0.15, test_size=0.15)

    # Prepare X/y
    X_train = train.drop(columns=['close'])
    y_train = train['close']
    X_val = val.drop(columns=['close'])
    y_val = val['close']
    X_test = test.drop(columns=['close'])
    y_test = test['close']

    # Scale
    scaler, X_train_scaled = fit_scaler(X_train)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    metrics = {}

    # Train MLP
    mlp_model, mlp_stats = train_mlp(pd.concat([X_train_scaled, y_train], axis=1), target_col='close')
    save_model(mlp_model, str(models_dir / 'mlp.joblib'))
    metrics['mlp'] = mlp_stats

    # Train XGBoost if available
    try:
        xgb_model, xgb_stats = train_xgboost(pd.concat([X_train, y_train], axis=1), target_col='close', num_boost_round=100)
        save_model(xgb_model, str(models_dir / 'xgboost.model'))
        metrics['xgboost'] = xgb_stats
    except Exception as e:
        metrics['xgboost'] = {'error': str(e)}

    # Save metrics
    with open(Path(output_dir) / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Training complete. Models and metrics saved to', output_dir)


if __name__ == '__main__':
    main()
