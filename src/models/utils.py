import joblib
from typing import Any


def save_model(model: Any, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
