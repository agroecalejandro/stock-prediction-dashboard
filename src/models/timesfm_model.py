"""TimesFM (Google Research): modelo fundacional de series de tiempo, preentrenado
sobre millones de series de dominios muy distintos, usado aquí en modo *zero-shot*.

A diferencia de los otros seis modelos, TimesFM no se entrena por ticker ni tiene
hiperparámetros que optimizar con Optuna (`optimize` se ignora, se mantiene solo por
consistencia de interfaz con los demás modelos). Predice directamente el NIVEL de
precio, no el retorno: el propio modelo normaliza cada serie de entrada de forma
relativa a sí misma antes de pronosticar (`normalize_inputs=True`), que es precisamente
lo que evita el problema de extrapolación que sí afecta a los árboles y redes pequeñas
entrenadas desde cero en este proyecto (ver metodologia_workflow.html, sección 2).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.build_features import build_feature_dataframe, load_prices
from src.models.common import (
    time_based_split, compute_metrics, save_predictions, save_metrics,
    get_future_business_dates,
)

_model = None
CONTEXT_LEN = 512  # ventana de historia (días) que se le da al modelo en cada pronóstico
MAX_HORIZON = 40


def _get_model():
    """Carga el checkpoint preentrenado de forma perezosa (se descarga una sola vez, ~800MB)."""
    global _model
    if _model is None:
        import timesfm
        from timesfm import ForecastConfig
        _model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        _model.compile(ForecastConfig(
            max_context=1024, max_horizon=MAX_HORIZON,
            normalize_inputs=True, use_continuous_quantile_head=True,
        ))
    return _model


def _context_window(full_close, end_pos: int):
    """Últimos CONTEXT_LEN valores de `full_close` terminando en la posición `end_pos` (incluida)."""
    start = max(0, end_pos - CONTEXT_LEN + 1)
    return full_close.values[start:end_pos + 1]


def train_and_predict(ticker: str, optimize: bool = True):
    """Backtest de validación (zero-shot, 1 paso) + pronóstico real a futuro (multi-paso nativo)."""
    df = build_feature_dataframe(ticker)
    _, val_df = time_based_split(df)

    full_close = load_prices(ticker)["close"]
    full_dates = full_close.index
    model = _get_model()

    # Todos los contextos usan la MISMA longitud (CONTEXT_LEN): esto evita que el
    # modelo recompile su grafo por cada forma distinta y hace el batch mucho más rápido.
    contexts, actual_targets, target_dates = [], [], []
    for row_date, row in val_df.iterrows():
        pos = full_dates.get_loc(row_date)
        contexts.append(_context_window(full_close, pos))
        actual_targets.append(row["close_next"])
        target_dates.append(row["target_date"])

    point_forecast, _ = model.forecast(horizon=1, inputs=contexts)
    preds = point_forecast[:, 0]

    metrics = compute_metrics(actual_targets, preds)
    save_predictions(ticker, "TimesFM", target_dates, preds, actual_targets)
    save_metrics(ticker, "TimesFM", metrics)

    # Pronóstico real a futuro: un solo forward multi-horizonte (nativo, no recursivo).
    future_dates = get_future_business_dates(full_close.index[-1])
    if len(future_dates):
        ctx = _context_window(full_close, len(full_close) - 1)
        future_forecast, _ = model.forecast(horizon=len(future_dates), inputs=[ctx])
        save_predictions(ticker, "TimesFM", future_dates, future_forecast[0])

    print(f"[TimesFM][{ticker}] {metrics} (zero-shot, sin hiperparámetros)")
    return model, metrics


if __name__ == "__main__":
    from config import TICKERS
    for t in TICKERS:
        train_and_predict(t)
