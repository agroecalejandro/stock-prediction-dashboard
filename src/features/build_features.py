"""Construye features técnicas y de calendario a partir de precios crudos."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from src.db.db_utils import get_session
from src.db.models import StockPrice


def load_prices(ticker: str) -> pd.DataFrame:
    """Carga el histórico de precios de un ticker desde la DB, ordenado por fecha."""
    with get_session() as session:
        rows = (
            session.query(StockPrice)
            .filter(StockPrice.ticker == ticker)
            .order_by(StockPrice.date)
            .all()
        )
        data = [{
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in rows]
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"No hay precios en la DB para {ticker}. Corre fetch_prices primero.")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables de calendario: mes, día de la semana, día del mes, etc."""
    df = df.copy()
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["quarter"] = df.index.quarter
    df["is_month_end"] = df.index.is_month_end.astype(int)
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega indicadores técnicos comunes: medias móviles, RSI, MACD, Bollinger."""
    df = df.copy()
    close = df["close"]

    df["sma_5"] = SMAIndicator(close, window=5).sma_indicator()
    df["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    df["sma_20"] = SMAIndicator(close, window=20).sma_indicator()

    df["rsi_14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    bb = BollingerBands(close, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["volatility_10d"] = close.pct_change().rolling(10).std()

    # Lags del precio de cierre (útiles para modelos ML tabulares)
    for lag in (1, 2, 3, 5, 10):
        df[f"close_lag_{lag}"] = close.shift(lag)

    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las columnas objetivo para pronóstico a t+1 (día siguiente).

    Todas las features de una fila (día t) usan solo información disponible al
    cierre del día t; el objetivo es el cierre del día siguiente (t+1), evitando
    que el modelo "vea" el open/high/low del día que está prediciendo.
    """
    df = df.copy()
    df["target_date"] = df.index.to_series().shift(-1)
    df["close_next"] = df["close"].shift(-1)
    df["return_next"] = df["close_next"] / df["close"] - 1
    return df


def build_feature_dataframe(ticker: str) -> pd.DataFrame:
    """Pipeline completo: carga precios -> agrega features -> agrega target t+1 -> limpia NaNs."""
    df = load_prices(ticker)
    df = add_calendar_features(df)
    df = add_technical_features(df)
    df = add_target_columns(df)
    df = df.dropna()
    return df


if __name__ == "__main__":
    from config import TICKERS
    for t in TICKERS:
        feats = build_feature_dataframe(t)
        print(f"[{t}] filas con features completas: {len(feats)}, columnas: {list(feats.columns)}")
