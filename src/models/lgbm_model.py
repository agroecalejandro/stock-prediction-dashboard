"""Modelo LightGBM (gradient boosting, alternativa moderna y muy usada a XGBoost)
para predicción de precio de cierre (t+1)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import optuna
from lightgbm import LGBMRegressor

from config import N_OPTUNA_TRIALS
from src.features.build_features import build_feature_dataframe
from src.models.common import (
    time_based_split, compute_metrics, save_predictions, save_metrics,
    recursive_tabular_forecast,
)
from src.models.rf_model import FEATURE_COLS, _prepare_xy


def _optimize_hyperparams(X_train, y_train, X_val, y_val, n_trials=N_OPTUNA_TRIALS):
    """Optimización bayesiana (Optuna) de hiperparámetros de LightGBM."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = compute_metrics(y_val, preds)["rmse"]
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_and_predict(ticker: str, optimize: bool = True):
    """Entrena LightGBM para un ticker, predice sobre el set de validación y guarda resultados."""
    df = build_feature_dataframe(ticker)
    train_df, val_df = time_based_split(df)

    X_train, y_train = _prepare_xy(train_df)
    X_val, y_val = _prepare_xy(val_df)

    if optimize:
        best_params = _optimize_hyperparams(X_train, y_train, X_val, y_val)
    else:
        best_params = {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05}
    best_params.update({"random_state": 42, "n_jobs": -1, "verbosity": -1})

    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    pred_returns = model.predict(X_val)

    pred_prices = val_df["close"].values * (1 + pred_returns)
    actual_prices = val_df["close_next"].values

    metrics = compute_metrics(actual_prices, pred_prices)
    save_predictions(ticker, "LightGBM", val_df["target_date"], pred_prices, actual_prices)
    save_metrics(ticker, "LightGBM", metrics)

    # Modelo final (re-entrenado con TODOS los datos) para el pronóstico real a futuro.
    X_all, y_all = _prepare_xy(df)
    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_all, y_all)
    future = recursive_tabular_forecast(ticker, final_model, FEATURE_COLS)
    if future:
        future_dates, future_prices = zip(*future)
        save_predictions(ticker, "LightGBM", future_dates, future_prices)

    print(f"[LightGBM][{ticker}] {metrics} best_params={best_params}")
    return model, metrics


if __name__ == "__main__":
    from config import TICKERS
    for t in TICKERS:
        train_and_predict(t)
