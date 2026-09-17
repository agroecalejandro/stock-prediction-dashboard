"""Configuración central del proyecto."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# En esta máquina, un antivirus/proxy hace inspección SSL e inyecta su propio
# certificado raíz en el almacén de Windows. Librerías como curl_cffi (usada por
# yfinance) no confían en ese almacén por defecto, así que apuntamos explícitamente
# a un bundle de certificados exportado desde Windows (ver windows_ca_bundle.pem).
# En el VPS (Linux, sin ese antivirus) este archivo no existirá y esto no aplica.
_CA_BUNDLE = BASE_DIR / "windows_ca_bundle.pem"
if _CA_BUNDLE.exists():
    os.environ.setdefault("CURL_CA_BUNDLE", str(_CA_BUNDLE))
    os.environ.setdefault("SSL_CERT_FILE", str(_CA_BUNDLE))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_CA_BUNDLE))
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "db" / "stock_data.db"
LOGS_DIR = BASE_DIR / "logs"

# Empresas a monitorear: nombre visible -> ticker de Yahoo Finance
# Nota: SpaceX es privada (no cotiza en bolsa), por lo que no se incluye.
COMPANIES = {
    "Amazon": "AMZN",
    "Meta": "META",
    "Google": "GOOGL",
    "NVIDIA": "NVDA",
}

TICKERS = list(COMPANIES.values())

# Ventana histórica a descargar para entrenar
HISTORY_PERIOD = "5y"
HISTORY_INTERVAL = "1d"

# Split temporal: % de datos más recientes usados como validación/prueba
VALIDATION_FRACTION = 0.30

# Horizonte de predicción (días hábiles hacia adelante)
FORECAST_HORIZON = 1

# Ventana de secuencia para modelos secuenciales (GRU, Transformer): días pasados usados como input
GRU_LOOKBACK = 30
TRANSFORMER_LOOKBACK = 30

# Modelos disponibles en el pipeline (usados también para filtrar en el dashboard)
MODEL_NAMES = ["ARIMA", "RandomForest", "XGBoost", "LightGBM", "GRU", "Transformer", "TimesFM"]

# Optuna: número de trials para optimización bayesiana de hiperparámetros
N_OPTUNA_TRIALS = 25

# Cuántos meses hacia el futuro pronosticar más allá del último precio real conocido
FUTURE_FORECAST_MONTHS = 1

# Noticias recientes: feed RSS de Yahoo Finance por ticker (rápido, sin rango de fechas).
NEWS_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# Noticias históricas: búsqueda en Google News RSS, que sí admite rango de fechas
# (operadores `after:`/`before:`) y no requiere API key. Se usa para el backfill
# histórico y se consulta en bloques mensuales (cada consulta regresa ~100 resultados).
GOOGLE_NEWS_RSS_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
NEWS_HISTORY_START_DATE = "2025-01-01"

# Modelo de sentimiento financiero (HuggingFace)
FINBERT_MODEL = "ProsusAI/finbert"
