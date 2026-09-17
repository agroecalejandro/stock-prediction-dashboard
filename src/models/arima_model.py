"""Modelo ARIMA (estadístico clásico) para predicción de precio de cierre."""
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from statsmodels.tsa.arima.model import ARIMA

from src.features.build_features import build_feature_dataframe, load_prices
from src.models.common import (
    time_based_split, compute_metrics, save_predictions, save_metrics,
    get_future_business_dates,
)

warnings.filterwarnings("ignore")

# Pequeña grilla de órdenes (p, d, q) candidatas; se elige la de menor AIC en train.
ORDER_CANDIDATES = [(1, 1, 0), (2, 1, 1), (5, 1, 0), (3, 1, 2), (2, 1, 2)]


def _select_best_order(train_series):
    best_order, best_aic, best_fit = None, float("inf"), None
    for order in ORDER_CANDIDATES:
        try:
            fit = ARIMA(train_series, order=order).fit()
            if fit.aic < best_aic:
                best_order, best_aic, best_fit = order, fit.aic, fit
        except Exception:
            continue
    return best_order, best_fit


def train_and_predict(ticker: str, optimize: bool = True):
    """Entrena ARIMA (rolling one-step-ahead) y guarda predicciones/métricas.

    ARIMA se re-ajusta en cada paso incorporando el valor real anterior (rolling forecast),
    que es la forma estándar de evaluar modelos de series de tiempo univariados.
    """
    df = build_feature_dataframe(ticker)
    train_df, val_df = time_based_split(df)

    history = list(train_df["close"].values)
    order, _ = _select_best_order(train_df["close"]) if optimize else ((5, 1, 0), None)
    order = order or (5, 1, 0)

    preds = []
    for actual_close in val_df["close"].values:
        model = ARIMA(history, order=order).fit()
        forecast = model.forecast(steps=1)[0]
        preds.append(forecast)
        history.append(actual_close)  # rolling: incorpora el valor real observado

    metrics = compute_metrics(val_df["close"].values, preds)
    save_predictions(ticker, "ARIMA", val_df.index, preds, val_df["close"].values)
    save_metrics(ticker, "ARIMA", metrics)

    # Pronóstico real a futuro: re-ajusta con TODO el historial real (incluido el
    # último día, que build_feature_dataframe descarta) y proyecta varios pasos a
    # la vez (ARIMA sí puede hacer esto de forma nativa, a diferencia de los
    # modelos tabulares que requieren simular cada día recursivamente).
    full_close = load_prices(ticker)["close"]
    future_dates = get_future_business_dates(full_close.index[-1])
    if len(future_dates):
        final_model = ARIMA(full_close.values, order=order).fit()
        future_preds = final_model.forecast(steps=len(future_dates))
        save_predictions(ticker, "ARIMA", future_dates, future_preds)

    print(f"[ARIMA][{ticker}] order={order} {metrics}")
    return None, metrics


if __name__ == "__main__":
    from config import TICKERS
    for t in TICKERS:
        train_and_predict(t)
