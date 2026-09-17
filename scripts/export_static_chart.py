"""Exports a standalone, interactive Plotly chart (actual prices + every model's fit
and forecast, for all four companies) as a single self-contained HTML file. Unlike the
live Dash dashboard, this file needs no running server: it can be opened directly in a
browser or published as a static page (e.g. GitHub Pages), with a "Company" dropdown
(built on Plotly's client-side `updatemenus`) to switch which company is shown.

Usage:
    venv\\Scripts\\python scripts\\export_static_chart.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go

from config import COMPANIES, MODEL_NAMES, TICKERS
from src.db.db_utils import get_session
from src.db.models import StockPrice, Prediction

TICKER_TO_NAME = {v: k for k, v in COMPANIES.items()}
TICKER_COLORS = {
    "AMZN": "#1f77b4", "META": "#ff7f0e", "GOOGL": "#2ca02c", "NVDA": "#d62728",
}
MODEL_DASH = {
    "ARIMA": "dot",
    "RandomForest": "dash",
    "XGBoost": "longdash",
    "LightGBM": "dashdot",
    "GRU": "longdashdot",
    "Transformer": "2px,6px,2px,6px",
    "TimesFM": "8px,3px,1px,3px",
}


def load_actual(ticker: str):
    with get_session() as session:
        rows = (
            session.query(StockPrice)
            .filter(StockPrice.ticker == ticker)
            .order_by(StockPrice.date)
            .all()
        )
        return [r.date for r in rows], [r.close for r in rows]


def load_predictions(ticker: str, model_name: str):
    with get_session() as session:
        rows = (
            session.query(Prediction)
            .filter_by(ticker=ticker, model_name=model_name)
            .order_by(Prediction.target_date)
            .all()
        )
        return [r.target_date for r in rows], [r.predicted_close for r in rows]


def build_figure() -> go.Figure:
    fig = go.Figure()
    trace_ticker = []  # which ticker each trace belongs to, used to build visibility masks

    for ticker in TICKERS:
        color = TICKER_COLORS.get(ticker, "#888888")
        name = TICKER_TO_NAME.get(ticker, ticker)

        x, y = load_actual(ticker)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{name} · actual",
                                  line=dict(color=color, width=2.2)))
        trace_ticker.append(ticker)

        for model_name in MODEL_NAMES:
            px, py = load_predictions(ticker, model_name)
            if not px:
                continue
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="lines", name=f"{name} · {model_name}",
                line=dict(color=color, dash=MODEL_DASH.get(model_name, "dot"), width=1.4),
                opacity=0.85,
            ))
            trace_ticker.append(ticker)

    buttons = []
    for ticker in TICKERS:
        visible = [t == ticker for t in trace_ticker]
        buttons.append(dict(label=TICKER_TO_NAME.get(ticker, ticker), method="update",
                             args=[{"visible": visible}]))
    buttons.append(dict(label="All companies", method="update",
                         args=[{"visible": [True] * len(trace_ticker)}]))

    default_ticker = TICKERS[0]
    for i, t in enumerate(trace_ticker):
        fig.data[i].visible = (t == default_ticker)

    fig.update_layout(
        title="Actual price vs. forecast — all models",
        xaxis_title="Date", yaxis_title="Close price (USD)",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
        updatemenus=[dict(
            buttons=buttons, direction="down", showactive=True,
            x=1.0, xanchor="right", y=1.15, yanchor="top",
        )],
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.08)),
        margin=dict(t=90),
    )
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_path = Path(__file__).resolve().parents[1] / "dashboard_preview.html"
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out_path}")
