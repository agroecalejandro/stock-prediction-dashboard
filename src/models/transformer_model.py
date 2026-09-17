"""Modelo Transformer (arquitectura de atención, el estado del arte actual en
secuencias) para predicción de precio de cierre (t+1). Mismo enfoque que GRU:
predice el RETORNO a t+1 a partir de una ventana de retornos pasados (evita el
problema de extrapolación de nivel de precio en acciones con fuerte tendencia),
pero usando self-attention en vez de recurrencia."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import optuna
import torch
import torch.nn as nn

from config import TRANSFORMER_LOOKBACK
from src.features.build_features import build_feature_dataframe, load_prices
from src.models.common import (
    time_based_split, compute_metrics, save_predictions, save_metrics,
    recursive_sequence_forecast,
)
from src.models.gru_model import _make_target_sequences, DEVICE

TRANSFORMER_N_TRIALS = 8
TRANSFORMER_SEARCH_EPOCHS = 15
TRANSFORMER_FINAL_EPOCHS = 60


class TransformerNet(nn.Module):
    def __init__(self, lookback, input_size=1, d_model=32, nhead=4, num_layers=1,
                 dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, lookback, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


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


def _optimize_hyperparams(X_train, y_train, X_val, y_val, lookback, n_trials=TRANSFORMER_N_TRIALS):
    """Optimización bayesiana (Optuna) de hiperparámetros del Transformer."""

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = TransformerNet(lookback=lookback, d_model=d_model, nhead=4, num_layers=num_layers)
        model = _train_model(model, X_train, y_train, epochs=TRANSFORMER_SEARCH_EPOCHS, lr=lr)
        preds = _predict(model, X_val)
        rmse = compute_metrics(y_val, preds)["rmse"]
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_and_predict(ticker: str, optimize: bool = True, lookback: int = TRANSFORMER_LOOKBACK):
    """Entrena un Transformer para un ticker, predice sobre el set de validación y guarda resultados."""
    df = build_feature_dataframe(ticker)
    train_df, val_df = time_based_split(df)

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
        best_params = _optimize_hyperparams(X_train, y_train, X_val, y_val, lookback)
    else:
        best_params = {"d_model": 32, "num_layers": 1, "lr": 0.005}

    model = TransformerNet(
        lookback=lookback, d_model=best_params["d_model"], nhead=4, num_layers=best_params["num_layers"],
    )
    model = _train_model(model, X_train, y_train, epochs=TRANSFORMER_FINAL_EPOCHS, lr=best_params["lr"])
    pred_returns_scaled = _predict(model, X_val)
    pred_returns = unscale(pred_returns_scaled)

    pred_prices = val_df["close"].values * (1 + pred_returns)
    actual_prices = val_df["close_next"].values

    metrics = compute_metrics(actual_prices, pred_prices)
    save_predictions(ticker, "Transformer", val_df["target_date"], pred_prices, actual_prices)
    save_metrics(ticker, "Transformer", metrics)

    # Modelo final (re-entrenado con TODO el historial real) para el pronóstico a futuro.
    full_close = load_prices(ticker)["close"]
    full_returns = full_close.pct_change().dropna()
    full_returns_scaled_all = scale(full_returns).values
    X_full, y_full = _make_target_sequences(full_returns_scaled_all[:-1], full_returns_scaled_all[1:], lookback)

    final_model = TransformerNet(
        lookback=lookback, d_model=best_params["d_model"], nhead=4, num_layers=best_params["num_layers"],
    )
    final_model = _train_model(final_model, X_full, y_full, epochs=TRANSFORMER_FINAL_EPOCHS, lr=best_params["lr"])

    def _predict_one(window_1d):
        x = window_1d.reshape(1, -1)
        return float(_predict(final_model, x)[0])

    future_dates, future_scaled = recursive_sequence_forecast(
        full_returns_scaled_all[-lookback:], _predict_one, lookback, full_close.index[-1],
    )
    if len(future_dates):
        future_returns = unscale(future_scaled)
        future_prices = float(full_close.iloc[-1]) * np.cumprod(1 + future_returns)
        save_predictions(ticker, "Transformer", future_dates, future_prices)

    print(f"[Transformer][{ticker}] {metrics} best_params={best_params}")
    return model, metrics


if __name__ == "__main__":
    from config import TICKERS
    for t in TICKERS:
        train_and_predict(t)
