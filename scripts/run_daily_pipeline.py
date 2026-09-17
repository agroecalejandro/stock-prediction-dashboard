"""Pipeline diario: solo actualiza precios y noticias/sentimiento (rápido).

El reentrenamiento de modelos ya NO ocurre aquí — ver scripts/run_weekly_training.py.
Separar ambos permite mantener el precio real y las noticias frescos todos los días
sin pagar el costo de reentrenar 7 modelos x N compañías a diario (innecesario: los
modelos no cambian tanto día a día, y esto libera CPU en el VPS).

Pensado para ejecutarse una vez al día vía cron (en el VPS) o Task Scheduler (en local).
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
