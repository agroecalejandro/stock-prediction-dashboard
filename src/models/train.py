"""Orquesta el entrenamiento de todos los modelos (ARIMA, RF, XGBoost, LightGBM,
GRU, Transformer, TimesFM) para todos los tickers. Cada modelo, además de evaluarse
sobre el set de validación, genera y guarda un pronóstico real a futuro (ver
FUTURE_FORECAST_MONTHS en config.py)."""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import TICKERS
from src.models import (
    arima_model, rf_model, xgb_model, lgbm_model, gru_model, transformer_model, timesfm_model,
)

MODEL_RUNNERS = {
    "ARIMA": arima_model.train_and_predict,
    "RandomForest": rf_model.train_and_predict,
    "XGBoost": xgb_model.train_and_predict,
    "LightGBM": lgbm_model.train_and_predict,
    "GRU": gru_model.train_and_predict,
    "Transformer": transformer_model.train_and_predict,
    "TimesFM": timesfm_model.train_and_predict,
}


def train_all(tickers=None, models=None, optimize: bool = True):
    tickers = tickers or TICKERS
    models = models or list(MODEL_RUNNERS.keys())
    results = {}
    for ticker in tickers:
        for model_name in models:
            runner = MODEL_RUNNERS[model_name]
            try:
                _, metrics = runner(ticker, optimize=optimize)
                results[(ticker, model_name)] = metrics
            except Exception as exc:
                print(f"[ERROR] {model_name} en {ticker}: {exc}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None, choices=list(MODEL_RUNNERS.keys()))
    parser.add_argument("--no-optimize", action="store_true", help="Usa hiperparámetros fijos, sin Optuna (más rápido).")
    args = parser.parse_args()

    train_all(tickers=args.tickers, models=args.models, optimize=not args.no_optimize)
