"""Daily pipeline: refreshes prices and news/sentiment.

Model training runs separately, on a weekly cadence (see run_weekly_training.py), so
that prices and news stay current every day without the cost of retraining every model
on each run. Intended to run once a day on a schedule (e.g. a cron job or a scheduled
task).
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import LOGS_DIR, TICKERS
from src.data_ingestion.fetch_prices import update_all_prices
from src.sentiment.analyze_sentiment import update_news_sentiment

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "daily_pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main():
    start = datetime.now()
    log.info("=== Iniciando pipeline diario (datos) ===")

    log.info("Paso 1/2: actualizando precios...")
    update_all_prices(TICKERS)

    log.info("Paso 2/2: actualizando noticias y sentimiento...")
    try:
        update_news_sentiment(TICKERS)
    except Exception as exc:
        log.warning(f"Fallo al actualizar noticias (se continúa igual): {exc}")

    elapsed = (datetime.now() - start).total_seconds()
    log.info(f"=== Pipeline diario completado en {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
