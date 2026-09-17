"""Utilidades compartidas por todos los modelos: split temporal, métricas, y guardado en DB."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import VALIDATION_FRACTION, FUTURE_FORECAST_MONTHS
from src.db.db_utils import get_session
from src.db.models import Prediction, ModelMetric


def time_based_split(df: pd.DataFrame, val_fraction: float = VALIDATION_FRACTION):
    """Divide un DataFrame ordenado por fecha en train/validación (sin mezclar, por ser serie de tiempo)."""
    n = len(df)
    split_idx = int(n * (1 - val_fraction))
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]
    return train, val


def compute_metrics(y_true, y_pred) -> dict:
    """Calcula MAE, RMSE y MAPE entre valores reales y predichos."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_predictions(ticker: str, model_name: str, dates, predicted, actual=None):
    """Guarda (o actualiza) predicciones de un modelo para un ticker en la DB."""
    actual = actual if actual is not None else [None] * len(dates)
    with get_session() as session:
        for date, pred, act in zip(dates, predicted, actual):
            date_ = pd.to_datetime(date).date()
            existing = (
                session.query(Prediction)
                .filter_by(ticker=ticker, model_name=model_name, target_date=date_)
                .first()
            )
            if existing:
                existing.predicted_close = float(pred)
                existing.actual_close = float(act) if act is not None else existing.actual_close
            else:
                session.add(Prediction(
                    ticker=ticker, model_name=model_name, target_date=date_,
                    predicted_close=float(pred),
                    actual_close=float(act) if act is not None else None,
                ))


def get_future_business_dates(last_date, months: int = FUTURE_FORECAST_MONTHS) -> pd.DatetimeIndex:
    """Días hábiles desde el día siguiente a `last_date` hasta `months` meses después."""
    start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    end = pd.Timestamp(last_date) + relativedelta(months=months)
    return pd.bdate_range(start=start, end=end)


def recursive_tabular_forecast(ticker: str, model, feature_cols: list, months: int = FUTURE_FORECAST_MONTHS):
    """Pronóstico recursivo a futuro para modelos tabulares (RF/XGBoost/LightGBM) que
    predicen el retorno a t+1.

    Como no conocemos el open/high/low/volumen reales de los días futuros, se
    aproximan con el propio cierre pronosticado (open=high=low=close) y el último
    volumen conocido; los indicadores técnicos se recalculan en cada paso sobre la
    serie extendida (real + pronosticado). Es una simplificación razonable para un
    horizonte corto (~1 mes): el error se acumula con cada paso recursivo.
    """
    from src.features.build_features import load_prices, add_calendar_features, add_technical_features

    working = load_prices(ticker)
    future_dates = get_future_business_dates(working.index[-1], months=months)

    results = []
    for target_date in future_dates:
        feat = add_technical_features(add_calendar_features(working))
        X = feat[feature_cols].iloc[[-1]]
        pred_return = float(model.predict(X)[0])
        last_close = float(working["close"].iloc[-1])
        pred_close = last_close * (1 + pred_return)
        results.append((target_date, pred_close))

        new_row = pd.DataFrame({
            "open": [pred_close], "high": [pred_close], "low": [pred_close],
            "close": [pred_close], "volume": [working["volume"].iloc[-1]],
        }, index=[target_date])
        working = pd.concat([working, new_row])

    return results


def recursive_sequence_forecast(initial_window, predict_fn, lookback: int, last_date,
                                 months: int = FUTURE_FORECAST_MONTHS):
    """Pronóstico recursivo a futuro para modelos secuenciales (GRU, Transformer).

    `initial_window` son los últimos `lookback` valores ESCALADOS reales conocidos.
    `predict_fn(window_1d)` debe regresar el siguiente valor escalado (float). En
    cada paso, el valor predicho se agrega a la ventana para generar el siguiente
    (igual que ARIMA rolling, pero con valores propios en vez de reales, ya que no
    hay "futuro real" que incorporar).
    """
    future_dates = get_future_business_dates(last_date, months=months)
    window = list(initial_window)
    preds = []
    for _ in future_dates:
        next_val = predict_fn(np.array(window[-lookback:]))
        preds.append(next_val)
        window.append(next_val)
    return future_dates, np.array(preds)


def save_metrics(ticker: str, model_name: str, metrics: dict):
    """Guarda (o actualiza) las métricas de validación de un modelo para un ticker.

    `trained_at` se refresca en CADA llamada (no solo al insertar la primera vez):
    es lo único que el dashboard usa para saber cuándo fue el último reentrenamiento
    ("Last training"), así que debe reflejar la corrida más reciente, no la fecha de
    creación original de la fila.
    """
    with get_session() as session:
        existing = (
            session.query(ModelMetric)
            .filter_by(ticker=ticker, model_name=model_name)
            .first()
        )
        if existing:
            existing.mae = metrics["mae"]
            existing.rmse = metrics["rmse"]
            existing.mape = metrics["mape"]
            existing.trained_at = datetime.utcnow()
        else:
            session.add(ModelMetric(
                ticker=ticker, model_name=model_name,
                mae=metrics["mae"], rmse=metrics["rmse"], mape=metrics["mape"],
                trained_at=datetime.utcnow(),
            ))
