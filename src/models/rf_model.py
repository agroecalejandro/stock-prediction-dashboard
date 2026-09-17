"""Modelo Random Forest para predicción de precio de cierre (t+1)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import optuna
from sklearn.ensemble import RandomForestRegressor

from config import N_OPTUNA_TRIALS
from src.features.build_features import build_feature_dataframe
from src.models.common import (
    time_based_split, compute_metrics, save_predictions, save_metrics,
    recursive_tabular_forecast,
)

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "month", "day_of_week", "day_of_month", "quarter", "is_month_end",
    "sma_5", "sma_10", "sma_20", "rsi_14", "macd", "macd_signal",
    "bb_high", "bb_low", "return_1d", "return_5d", "volatility_10d",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
]
# Se predice el RETORNO porcentual a t+1 (no el precio absoluto): los modelos de
# árboles no pueden extrapolar fuera del rango de precios visto en entrenamiento,
# lo cual falla en acciones con fuerte tendencia (ej. NVDA, GOOGL). El retorno es
# estacionario y no sufre ese problema; el precio se reconstruye después.
TARGET_COL = "return_next"


def _prepare_xy(df):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def _optimize_hyperparams(X_train, y_train, X_val, y_val, n_trials=N_OPTUNA_TRIALS):
    """Optimización bayesiana (Optuna) de hiperparámetros de Random Forest."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = compute_metrics(y_val, preds)["rmse"]
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_and_predict(ticker: str, optimize: bool = True):
    """Entrena Random Forest para un ticker, predice sobre el set de validación y guarda resultados."""
    df = build_feature_dataframe(ticker)
    train_df, val_df = time_based_split(df)

    X_train, y_train = _prepare_xy(train_df)
    X_val, y_val = _prepare_xy(val_df)

    if optimize:
        best_params = _optimize_hyperparams(X_train, y_train, X_val, y_val)
    else:
        best_params = {"n_estimators": 300, "max_depth": 10}
    best_params.update({"random_state": 42, "n_jobs": -1})

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    pred_returns = model.predict(X_val)

    # Reconstruye el precio a partir del retorno predicho: precio_t * (1 + retorno_predicho)
    pred_prices = val_df["close"].values * (1 + pred_returns)
    actual_prices = val_df["close_next"].values

    metrics = compute_metrics(actual_prices, pred_prices)
    save_predictions(ticker, "RandomForest", val_df["target_date"], pred_prices, actual_prices)
    save_metrics(ticker, "RandomForest", metrics)

    # Modelo final (re-entrenado con TODOS los datos) para el pronóstico real a futuro.
    X_all, y_all = _prepare_xy(df)
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X_all, y_all)
    future = recursive_tabular_forecast(ticker, final_model, FEATURE_COLS)
    if future:
        future_dates, future_prices = zip(*future)
        save_predictions(ticker, "RandomForest", future_dates, future_prices)

    print(f"[RandomForest][{ticker}] {metrics} best_params={best_params}")
    return model, metrics


if __name__ == "__main__":
    from config import TICKERS
    for t in TICKERS:
        train_and_predict(t)
