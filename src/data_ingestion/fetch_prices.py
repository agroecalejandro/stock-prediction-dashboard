"""Descarga precios históricos de acciones desde Yahoo Finance y los guarda en la DB."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import yfinance as yf
import pandas as pd

from config import TICKERS, HISTORY_PERIOD, HISTORY_INTERVAL
from src.db.db_utils import get_session, init_db
from src.db.models import StockPrice


def fetch_ticker_history(ticker: str, period: str = HISTORY_PERIOD,
                          interval: str = HISTORY_INTERVAL) -> pd.DataFrame:
    """Descarga el histórico de un ticker y lo regresa como DataFrame limpio."""
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    return df[["date", "open", "high", "low", "close", "volume"]]


def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    """Inserta precios nuevos en la DB, ignorando los que ya existen (ticker+fecha)."""
    inserted = 0
    with get_session() as session:
        existing_dates = {
            d for (d,) in session.query(StockPrice.date).filter(StockPrice.ticker == ticker)
        }
        for row in df.itertuples(index=False):
            if row.date in existing_dates:
                continue
            session.add(StockPrice(
                ticker=ticker, date=row.date, open=row.open, high=row.high,
                low=row.low, close=row.close, volume=row.volume,
            ))
            inserted += 1
    return inserted


def update_all_prices(tickers=None):
    """Descarga y guarda el histórico de todos los tickers configurados."""
    init_db()
    tickers = tickers or TICKERS
    summary = {}
    for ticker in tickers:
        df = fetch_ticker_history(ticker)
        n = upsert_prices(ticker, df)
        summary[ticker] = {"rows_downloaded": len(df), "rows_inserted": n}
        print(f"[{ticker}] descargadas={len(df)} nuevas_insertadas={n}")
    return summary


if __name__ == "__main__":
    update_all_prices()
