"""Weekly retraining: runs all seven models for every company, with full Bayesian
hyperparameter optimization, and regenerates both the historical fit and the forward
forecast.

Intended to run once a week on a schedule (e.g. a cron job or a scheduled task). The
timestamp of this run is recorded in ModelMetric.trained_at, which the dashboard
displays as "Last training".
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import LOGS_DIR, TICKERS
from src.models.train import train_all

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "weekly_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main(optimize: bool):
    start = datetime.now()
    log.info("=== Iniciando reentrenamiento semanal ===")
    train_all(tickers=TICKERS, optimize=optimize)
    elapsed = (datetime.now() - start).total_seconds()
    log.info(f"=== Reentrenamiento semanal completado en {elapsed:.1f}s ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optimize", action="store_true",
                         help="Usa hiperparámetros fijos, sin Optuna (más rápido).")
    args = parser.parse_args()
    main(optimize=not args.no_optimize)
