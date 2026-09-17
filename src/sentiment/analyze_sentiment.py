"""Clasifica el sentimiento de titulares financieros usando FinBERT."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformers import pipeline

from config import FINBERT_MODEL, TICKERS
from src.db.db_utils import get_session, init_db
from src.db.models import NewsSentiment
from src.data_ingestion.fetch_news import fetch_all_news, fetch_historical_news_for_ticker

_classifier = None


def get_classifier():
    """Carga el pipeline de FinBERT de forma perezosa (se descarga una sola vez)."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline("text-classification", model=FINBERT_MODEL)
    return _classifier


def classify_headlines(headlines: list[str], batch_size: int = 32) -> list[dict]:
    """Clasifica una lista de titulares como positive/negative/neutral con score."""
    if not headlines:
        return []
    clf = get_classifier()
    results = clf(headlines, truncation=True, batch_size=batch_size)
    return results


def classify_and_save_items(items: list[dict]) -> int:
    """Clasifica una lista de noticias (dicts con ticker/title/link/published_at) y
    las guarda en la DB, evitando duplicados (ticker, title)."""
    if not items:
        return 0
    titles = [it["title"] for it in items]
    sentiments = classify_headlines(titles)

    inserted = 0
    with get_session() as session:
        existing = {
            (t, title) for (t, title) in
            session.query(NewsSentiment.ticker, NewsSentiment.title)
        }
        for item, sent in zip(items, sentiments):
            key = (item["ticker"], item["title"])
            if key in existing or not item["title"]:
                continue
            session.add(NewsSentiment(
                ticker=item["ticker"],
                published_at=item["published_at"],
                title=item["title"],
                link=item["link"],
                sentiment_label=sent["label"].lower(),
                sentiment_score=float(sent["score"]),
            ))
            existing.add(key)
            inserted += 1
    return inserted


def update_news_sentiment(tickers=None):
    """Descarga noticias recientes (RSS Yahoo Finance), las clasifica y guarda en la DB."""
    init_db()
    items = fetch_all_news(tickers)
    if not items:
        print("No se encontraron noticias.")
        return 0
    inserted = classify_and_save_items(items)
    print(f"Noticias nuevas insertadas: {inserted}")
    return inserted


def backfill_historical_news(tickers=None, start_date: str = None):
    """Backfill histórico: busca noticias desde `start_date` (default: config
    NEWS_HISTORY_START_DATE) por ticker vía Google News RSS, las clasifica con
    FinBERT y las guarda. Puede tardar varios minutos (descarga mes a mes).
    """
    init_db()
    tickers = tickers or TICKERS
    total_inserted = 0
    for ticker in tickers:
        items = fetch_historical_news_for_ticker(ticker, start_date=start_date)
        print(f"[{ticker}] {len(items)} titulares históricos encontrados, clasificando...")
        inserted = classify_and_save_items(items)
        print(f"[{ticker}] {inserted} noticias nuevas insertadas")
        total_inserted += inserted
    print(f"Total de noticias históricas insertadas: {total_inserted}")
    return total_inserted


if __name__ == "__main__":
    update_news_sentiment()
