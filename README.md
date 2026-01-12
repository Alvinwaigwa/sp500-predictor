Professional RSI Strategy Backtest — extended

Overview
This repository started as an RSI-based backtest for the S&P 500. It has been extended with a reproducible data pipeline, feature engineering, baseline machine learning models, deep learning models, hyperparameter tuning and walk-forward evaluation so the codebase can be used for systematic model development and reproducible experiments.

What’s included
- Data loading and deterministic splits: `src/data_loader.py` provides downloading and a `train/val/test` splitter for time-series workflows.
- Feature engineering: `src/features.py` contains lag and rolling-window feature builders and scaler utilities.
- Baseline models: `src/models/` includes XGBoost and a scikit-learn MLP with training helpers and save/load utilities.
- Deep learning: `src/models/deep.py` provides simple LSTM and Transformer model definitions and a minimal PyTorch training loop with checkpoint support.
- Experiments: `src/experiments.py` contains an Optuna study for tuning XGBoost and MLP hyperparameters and can optionally log runs with MLflow.
- Training script: `src/train.py` runs a quick end-to-end baseline training (features → train → save models and `metrics.json`).
- Backtesting and evaluation: `src/backtest.py` includes a vectorized backtester, prediction-to-signal helper, regression metrics (MAE, RMSE, MAPE) and a walk-forward evaluation helper.

Where outputs go
- Models and training artifacts: `outputs/models/`
- Metrics and study results: `outputs/metrics.json` and `outputs/experiments/study_result.json`

Quickstart (local)
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run a baseline training run (saves models and `metrics.json`):

```bash
python3 -m src.train
```

4. Run a quick Optuna experiment (example):

```bash
python3 -m src.experiments --model xgboost --trials 20
```

Notes for resumes and interviews
- Describe the pipeline: deterministic data splits, feature engineering, baseline → deep learning progression, and walk-forward evaluation.
- Highlight concrete artifacts: trained model files in `outputs/models/`, `metrics.json`, and Optuna study results under `outputs/experiments/`.
- Be ready to show the notebook walkthrough (`notebooks/analysis.ipynb`) or reproduce a quick demo using `python3 -m src.train` during an interview.

If you want, I can add a short model card and a polished `notebooks/modeling.ipynb` that walks through the full experiment and produces the key figures you'd include on a resume.