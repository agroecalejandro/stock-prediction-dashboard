"""Backfill único de noticias históricas (desde config.NEWS_HISTORY_START_DATE hasta
hoy) vía Google News RSS, clasificadas con FinBERT. Correr una sola vez para poblar
el histórico; el pipeline diario (fetch_news_for_ticker, RSS de Yahoo) se encarga de
mantenerlo al día después.

Uso:
    venv\\Scripts\\python scripts\\backfill_historical_news.py
    venv\\Scripts\\python scripts\\backfill_historical_news.py --start 2024-06-01
    venv\\Scripts\\python scripts\\backfill_historical_news.py --tickers AMZN NVDA
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.sentiment.analyze_sentiment import backfill_historical_news

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--start", default=None, help="Fecha de inicio YYYY-MM-DD (default: config.NEWS_HISTORY_START_DATE)")
    args = parser.parse_args()

    backfill_historical_news(tickers=args.tickers, start_date=args.start)
