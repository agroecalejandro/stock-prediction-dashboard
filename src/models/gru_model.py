"""Modelo GRU (deep learning) para predicción de precio de cierre (t+1).

Igual que RandomForest/XGBoost/LightGBM, predice el RETORNO porcentual a t+1 (no
el precio absoluto): entrenar sobre la serie de precios directamente y escalar
con MinMax fijo en train hace que el modelo no pueda generar salidas fuera del
rango de entrenamiento, lo cual falla en acciones con fuerte tendencia (ej. NVDA,
GOOGL) exactamente igual que le pasaba a los árboles. El retorno es estacionario
(oscila siempre alrededor de 0 sin importar el nivel de precio) y no sufre esto.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import optuna
import torch
import torch.nn as nn

from config import GRU_LOOKBACK
from src.features.build_features import build_feature_dataframe, load_prices
from src.models.common import (
    time_based_split, compute_metrics, save_predictions, save_metrics,
    recursive_sequence_forecast,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optuna para GRU es más costoso (cada trial entrena una red), así que usamos menos trials/epochs.
GRU_N_TRIALS = 8
GRU_SEARCH_EPOCHS = 15
GRU_FINAL_EPOCHS = 60


class GRUNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def _make_target_sequences(returns_arr: np.ndarray, targets_arr: np.ndarray, lookback: int):
    """Ventanas de `lookback` retornos pasados (incluyendo el del día actual) -> retorno del día siguiente."""
    X, y = [], []
    for i in range(lookback - 1, len(returns_arr)):
        X.append(returns_arr[i - lookback + 1: i + 1])
        y.append(targets_arr[i])
    return np.array(X), np.array(y)


def _train_model(model, X_train, y_train, epochs, lr):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
    return model


def _predict(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        return model(X_t).cpu().numpy().flatten()


def _optimize_hyperparams(X_train, y_train, X_val, y_val, n_trials=GRU_N_TRIALS):
    """Optimización bayesiana (Optuna) de hiperparámetros de la red GRU."""

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = GRUNet(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
        model = _train_model(model, X_train, y_train, epochs=GRU_SEARCH_EPOCHS, lr=lr)
        preds = _predict(model, X_val)
        rmse = compute_metrics(y_val, preds)["rmse"]
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_and_predict(ticker: str, optimize: bool = True, lookback: int = GRU_LOOKBACK):
    """Entrena una red GRU para un ticker, predice sobre el set de validación y guarda resultados."""
    df = build_feature_dataframe(ticker)
    train_df, val_df = time_based_split(df)

    # return_1d = retorno del día anterior a hoy -> hoy (input); return_next = de hoy -> mañana (target).
    mean, std = train_df["return_1d"].mean(), train_df["return_1d"].std()
    def scale(x):
        return (x - mean) / std

    def unscale(x):
        return x * std + mean

    full_input_scaled = scale(df["return_1d"]).values
    full_target_scaled = scale(df["return_next"]).values
    n_train = len(train_df)

    X_train, y_train = _make_target_sequences(full_input_scaled[:n_train], full_target_scaled[:n_train], lookback)

    val_start = n_train - lookback + 1
    X_val, y_val = _make_target_sequences(full_input_scaled[val_start:], full_target_scaled[val_start:], lookback)

    if optimize:
        best_params = _optimize_hyperparams(X_train, y_train, X_val, y_val)
    else:
        # A diferencia de RF/XGBoost, el GRU es muy sensible al learning rate:
        # con valores bajos (ej. 0.001) no converge en pocas épocas y su error
        # se dispara. Este default replica lo que Optuna suele elegir como mejor.
        best_params = {"hidden_size": 64, "num_layers": 1, "lr": 0.005}

    model = GRUNet(
        input_size=1,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
    )
    model = _train_model(model, X_train, y_train, epochs=GRU_FINAL_EPOCHS, lr=best_params["lr"])
    pred_returns_scaled = _predict(model, X_val)
    pred_returns = unscale(pred_returns_scaled)

    pred_prices = val_df["close"].values * (1 + pred_returns)
    actual_prices = val_df["close_next"].values

    metrics = compute_metrics(actual_prices, pred_prices)
    save_predictions(ticker, "GRU", val_df["target_date"], pred_prices, actual_prices)
    save_metrics(ticker, "GRU", metrics)

    # Modelo final (re-entrenado con TODO el historial real, incluido el último día
    # que build_feature_dataframe descarta) para el pronóstico real a futuro.
    full_close = load_prices(ticker)["close"]
    full_returns = full_close.pct_change().dropna()
    full_returns_scaled_all = scale(full_returns).values
    X_full, y_full = _make_target_sequences(full_returns_scaled_all[:-1], full_returns_scaled_all[1:], lookback)

    final_model = GRUNet(
        input_size=1, hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"],
    )
    final_model = _train_model(final_model, X_full, y_full, epochs=GRU_FINAL_EPOCHS, lr=best_params["lr"])

    def _predict_one(window_1d):
        x = window_1d.reshape(1, -1)
        return float(_predict(final_model, x)[0])

    future_dates, future_scaled = recursive_sequence_forecast(
        full_returns_scaled_all[-lookback:], _predict_one, lookback, full_close.index[-1],
    )
    if len(future_dates):
        future_returns = unscale(future_scaled)
        future_prices = float(full_close.iloc[-1]) * np.cumprod(1 + future_returns)
        save_predictions(ticker, "GRU", future_dates, future_prices)

    print(f"[GRU][{ticker}] {metrics} best_params={best_params}")
    return model, metrics


if __name__ == "__main__":
    from config import TICKERS
    for t in TICKERS:
        train_and_predict(t)
