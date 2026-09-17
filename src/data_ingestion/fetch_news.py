"""Descarga titulares de noticias financieras por ticker.

Dos fuentes con propósitos distintos:
- RSS de Yahoo Finance (`fetch_news_for_ticker`): solo titulares recientes, sin
  rango de fechas, rápido — se usa en el pipeline diario.
- Búsqueda en Google News RSS (`fetch_historical_news_for_ticker`): admite
  `after:`/`before:`, no requiere API key — se usa para el backfill histórico.
"""
import sys
import time
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).resolve().parents[2]))

import feedparser

from config import TICKERS, COMPANIES, NEWS_RSS_TEMPLATE, GOOGLE_NEWS_RSS_TEMPLATE, NEWS_HISTORY_START_DATE


def fetch_news_for_ticker(ticker: str) -> list[dict]:
    """Obtiene titulares recientes para un ticker desde el feed RSS."""
    url = NEWS_RSS_TEMPLATE.format(ticker=ticker)
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        published = None
        if getattr(entry, "published_parsed", None):
            published = datetime(*entry.published_parsed[:6])
        items.append({
            "ticker": ticker,
            "title": entry.get("title", "").strip(),
            "link": entry.get("link"),
            "published_at": published,
        })
    return items


def _month_ranges(start_date: datetime, end_date: datetime):
    """Genera tuplas (inicio, fin) mensuales entre start_date y end_date."""
    cur = start_date.replace(day=1)
    ranges = []
    while cur < end_date:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        ranges.append((cur, min(nxt, end_date)))
        cur = nxt
    return ranges


def fetch_historical_news_for_ticker(ticker: str, start_date: str = None, end_date: datetime = None,
                                      pause_seconds: float = 1.0) -> list[dict]:
    """Busca noticias históricas de un ticker en Google News RSS, mes a mes.

    Usa el nombre de la compañía además del ticker para mejorar la relevancia
    (buscar solo "NVDA" trae mucho ruido no financiero).
    """
    ticker_to_name = {v: k for k, v in COMPANIES.items()}
    company_name = ticker_to_name.get(ticker, ticker)
    start = datetime.strptime(start_date or NEWS_HISTORY_START_DATE, "%Y-%m-%d")
    end = end_date or datetime.utcnow()

    items, seen_titles = [], set()
    for period_start, period_end in _month_ranges(start, end):
        query = f'("{ticker}" OR "{company_name}") stock after:{period_start:%Y-%m-%d} before:{period_end:%Y-%m-%d}'
        url = GOOGLE_NEWS_RSS_TEMPLATE.format(query=urllib.parse.quote(query))
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            published = None
            if getattr(entry, "published_parsed", None):
                published = datetime(*entry.published_parsed[:6])
            items.append({
                "ticker": ticker, "title": title,
                "link": entry.get("link"), "published_at": published,
            })
        time.sleep(pause_seconds)  # ser gentil con el servidor de Google News
    return items


def fetch_all_historical_news(tickers=None, start_date: str = None) -> list[dict]:
    tickers = tickers or TICKERS
    all_items = []
    for ticker in tickers:
        items = fetch_historical_news_for_ticker(ticker, start_date=start_date)
        print(f"[{ticker}] {len(items)} titulares históricos encontrados")
        all_items.extend(items)
    return all_items


def fetch_all_news(tickers=None) -> list[dict]:
    tickers = tickers or TICKERS
    all_items = []
    for ticker in tickers:
        items = fetch_news_for_ticker(ticker)
        print(f"[{ticker}] {len(items)} titulares encontrados")
        all_items.extend(items)
    return all_items


if __name__ == "__main__":
    news = fetch_all_news()
    for n in news[:5]:
        print(n)
