# Stock Prediction Dashboard

Daily monitoring and price forecasting for four tech stocks (Amazon, Meta, Alphabet/Google,
NVIDIA) comparing seven models across four modeling paradigms — statistical, machine
learning, deep learning, and a pretrained time-series foundation model — with an
interactive dashboard (Dash + Plotly) filterable by company and model.

A full methodology write-up (theory, math, and validation design behind every model) is
included in [`metodologia_workflow.html`](./metodologia_workflow.html) and is also
served as a second tab inside the running dashboard.

## Architecture

```
Yahoo Finance (prices) ────┐
                            ├─► Features (technical + calendar) ──► Models ──► Predictions (DB)
RSS news + FinBERT ─────────┘                                          │
                                                                        ▼
                                                               Dashboard (Dash/Plotly)
```

- **Data**: historical prices via `yfinance`; recent news via Yahoo Finance RSS (fast,
  daily) and historical news backfill via Google News RSS search (date-ranged, no API
  key needed).
- **Features**: technical indicators (SMA, RSI, MACD, Bollinger Bands, returns,
  volatility, price lags) + calendar variables (month, day of week, etc.).
- **Models** (predict the **next-day**, t+1, close, then run recursively/natively to
  project ~1 month ahead):
  - **ARIMA** (classical statistical model, rolling one-step forecast; native multi-step
    `forecast()` for the future projection).
  - **Random Forest**, **XGBoost**, **LightGBM** (tabular ML over technical features).
  - **GRU** and **Transformer** (deep learning, trained from scratch on sequences of past
    returns).
  - **TimesFM** (Google's pretrained time-series foundation model, used *zero-shot* — no
    per-ticker training, no hyperparameters).
  - **Six of the seven models predict the percentage *return* at t+1, not the absolute
    price** — the price is reconstructed afterward (`price_t * (1 + predicted_return)`).
    This matters: predicting the price level directly, tree ensembles and networks
    trained with a fixed scaler cannot produce outputs outside the range seen in
    training — which fails exactly on strongly trending stocks like NVIDIA or Google.
    This was a real bug found and fixed during development (see the methodology doc for
    the before/after numbers). TimesFM is the one exception: it targets the price level
    directly, since it internally normalizes each input window relative to itself.
  - All trainable models are tuned with **Optuna** (Bayesian hyperparameter
    optimization).
  - **Forward forecast**: after evaluation, each model is refit on the full available
    history and projects business days ahead (`FUTURE_FORECAST_MONTHS` in `config.py`,
    1 month by default). For the tabular models this is a recursive day-by-day rollout
    simulating each new close (a reasonable simplification at this horizon — error
    compounds with every step). These predictions are stored with `actual_close = NULL`
    until that day actually happens.
- **Validation**: chronological 70/30 split (most recent 30% held out — never shuffled,
  since this is time-series data).
- **Sentiment**: FinBERT (`ProsusAI/finbert`) classifies headlines as
  positive/negative/neutral.
- **Database**: SQLite (`data/db/stock_data.db`).
- **Dashboard**: Dash + Plotly —
  - Filter by **one or several companies** at once (color-coded per company) and by
    model(s) (distinct line pattern per model).
  - "Today" vertical line separating the historical fit from the forward forecast.
  - 100%-stacked background showing daily news sentiment (red = share of negative
    headlines, green = the rest; a day with no news shows 100% green), toggleable via
    checkbox.
  - Date range slider (drag the handles under the chart).
  - News list synced to the chart's visible date range — zoom/pan the chart to browse
    headlines from that period.
  - "Last training" indicator showing when the models currently shown were last refit.
  - Second tab embedding the full methodology write-up.

## Local setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

> If you hit SSL errors (`CERTIFICATE_VERIFY_FAILED`) installing packages or fetching
> data — typical when an antivirus does HTTPS inspection — also run
> `venv\Scripts\pip install pip-system-certs`. `yfinance` specifically uses `curl_cffi`,
> which doesn't automatically trust the Windows certificate store; if you still see SSL
> errors from it, export your Windows trusted roots to a PEM file and point
> `CURL_CA_BUNDLE`/`SSL_CERT_FILE` at it (see `config.py` for how this project picks up
> such a file automatically if present as `windows_ca_bundle.pem` in the project root).

## Usage

**1. Download historical prices:**
```bash
venv\Scripts\python -m src.data_ingestion.fetch_prices
```

**2. Download recent news and classify sentiment** (downloads FinBERT the first time, ~440MB):
```bash
venv\Scripts\python -m src.sentiment.analyze_sentiment
```

**2b. (Optional) Backfill historical news** further back than "recent" (default: since 2025-01-01):
```bash
venv\Scripts\python scripts\backfill_historical_news.py
```

**3. Train all models and generate predictions:**
```bash
venv\Scripts\python -m src.models.train
# faster, without Bayesian optimization:
venv\Scripts\python -m src.models.train --no-optimize
# only some tickers/models:
venv\Scripts\python -m src.models.train --tickers AMZN NVDA --models ARIMA GRU TimesFM
```

**4. Launch the dashboard:**
```bash
venv\Scripts\python -m src.dashboard.app
```
This starts a local development server; open the URL it prints in your browser.

**Production cadence** (two separate scripts, meant to be scheduled independently — daily
data refresh is cheap, weekly retraining is the expensive step):
```bash
venv\Scripts\python scripts\run_daily_pipeline.py     # prices + news only, fast, run daily
venv\Scripts\python scripts\run_weekly_training.py    # retrains all 7 models, run weekly
```

## Project layout

```
config.py                        # tickers, paths, model/feature settings
src/
  data_ingestion/                # yfinance prices, RSS/Google News scraping
  features/                      # technical indicators + calendar features
  sentiment/                     # FinBERT classification
  models/                        # one file per model (arima, rf, xgb, lgbm, gru,
                                  # transformer, timesfm) + shared train/eval utilities
  db/                            # SQLAlchemy models + session handling
  dashboard/                     # Dash app (app.py)
scripts/                         # daily/weekly pipeline entry points, news backfill
metodologia_workflow.html        # full methodology write-up (theory + math)
```
