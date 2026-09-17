"""Dashboard (Dash + Plotly) para visualizar precios reales vs predicción (incluyendo
pronóstico a futuro), sentimiento de noticias y métricas, filtrable por compañía(s)
y modelo(s)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from flask import send_from_directory
from sqlalchemy import func
import dash_bootstrap_components as dbc

from config import COMPANIES, MODEL_NAMES, TICKERS, BASE_DIR
from src.db.db_utils import get_session
from src.db.models import StockPrice, Prediction, ModelMetric, NewsSentiment

TICKER_TO_NAME = {v: k for k, v in COMPANIES.items()}

# Un color por compañía (consistente entre la línea real y sus predicciones) y un
# patrón de línea distinto por modelo, para poder distinguir ambos ejes a la vez
# cuando se seleccionan varias compañías y varios modelos al mismo tiempo.
TICKER_COLORS = {
    "AMZN": "#1f77b4", "META": "#ff7f0e", "GOOGL": "#2ca02c", "NVDA": "#d62728",
}
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#17becf", "#bcbd22"]

MODEL_DASH = {
    "ARIMA": "dot",
    "RandomForest": "dash",
    "XGBoost": "longdash",
    "LightGBM": "dashdot",
    "GRU": "longdashdot",
    "Transformer": "2px,6px,2px,6px",
    "TimesFM": "8px,3px,1px,3px",
}

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Stock Prediction Dashboard"
server = app.server  # exposes the underlying Flask app for a WSGI server


@server.route("/metodologia")
def serve_metodologia():
    """Sirve la nota metodológica (HTML estático) para mostrarla en un iframe."""
    return send_from_directory(BASE_DIR, "metodologia_workflow.html")


def ticker_color(ticker: str, idx: int) -> str:
    return TICKER_COLORS.get(ticker, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def load_actual_prices(ticker: str) -> pd.DataFrame:
    with get_session() as session:
        rows = (
            session.query(StockPrice)
            .filter(StockPrice.ticker == ticker)
            .order_by(StockPrice.date)
            .all()
        )
        data = [{"date": r.date, "close": r.close} for r in rows]
    return pd.DataFrame(data)


def load_predictions(ticker: str, model_name: str) -> pd.DataFrame:
    with get_session() as session:
        rows = (
            session.query(Prediction)
            .filter_by(ticker=ticker, model_name=model_name)
            .order_by(Prediction.target_date)
            .all()
        )
        data = [{
            "date": r.target_date, "predicted": r.predicted_close, "actual": r.actual_close,
        } for r in rows]
    return pd.DataFrame(data)


def load_last_training_date():
    """Fecha del reentrenamiento más reciente (MAX(trained_at) en ModelMetric)."""
    with get_session() as session:
        latest = session.query(func.max(ModelMetric.trained_at)).scalar()
    return latest


def load_metrics(ticker: str) -> pd.DataFrame:
    with get_session() as session:
        rows = session.query(ModelMetric).filter_by(ticker=ticker).all()
        data = [{
            "Ticker": ticker, "Model": r.model_name, "MAE": round(r.mae, 3),
            "RMSE": round(r.rmse, 3), "MAPE (%)": round(r.mape, 2),
        } for r in rows]
    return pd.DataFrame(data)


def load_news(ticker: str, limit: int = 3000) -> pd.DataFrame:
    with get_session() as session:
        rows = (
            session.query(NewsSentiment)
            .filter_by(ticker=ticker)
            .order_by(NewsSentiment.published_at.desc())
            .limit(limit)
            .all()
        )
        data = [{
            "ticker": ticker, "title": r.title, "label": r.sentiment_label,
            "score": round(r.sentiment_score, 3) if r.sentiment_score else None,
            "published_at": r.published_at, "link": r.link,
        } for r in rows]
    return pd.DataFrame(data)


def load_daily_negative_ratio(tickers: list) -> dict:
    """Para cada día, qué fracción de las noticias (de las compañías dadas) fueron
    negativas. Un día sin noticias no aparece aquí (se interpreta como 0% negativo,
    es decir 100% "verde", en el punto donde se usa este diccionario)."""
    if not tickers:
        return {}
    with get_session() as session:
        rows = (
            session.query(NewsSentiment.published_at, NewsSentiment.sentiment_label)
            .filter(NewsSentiment.ticker.in_(tickers))
            .all()
        )
    data = [{"date": pub.date(), "label": label} for pub, label in rows if pub is not None]
    if not data:
        return {}
    df = pd.DataFrame(data)
    counts = df.groupby(["date", "label"]).size().unstack(fill_value=0)
    for col in ("negative", "positive", "neutral"):
        if col not in counts.columns:
            counts[col] = 0
    total = counts.sum(axis=1)
    neg_ratio = (counts["negative"] / total.replace(0, 1)).clip(0, 1)
    return neg_ratio.to_dict()


dashboard_tab_content = html.Div([
    dbc.Row([
        dbc.Col([
            html.Label("Company"),
            dcc.Dropdown(
                id="ticker-dropdown",
                options=[{"label": name, "value": ticker} for name, ticker in COMPANIES.items()],
                value=[TICKERS[0]],
                multi=True,
            ),
        ], width=4),
        dbc.Col([
            html.Label("Model(s)"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": m, "value": m} for m in MODEL_NAMES],
                value=MODEL_NAMES,
                multi=True,
            ),
        ], width=6),
        dbc.Col([
            html.Label("Options"),
            dcc.Checklist(
                id="display-options",
                options=[
                    {"label": " News sentiment background", "value": "sentiment"},
                    {"label": " Historical fit (validation)", "value": "fit"},
                    {"label": " Forward forecast", "value": "future"},
                ],
                value=["sentiment", "fit", "future"],
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block"},
            ),
        ], width=2),
    ], className="mb-4"),

    dcc.Graph(id="price-chart"),
    html.P(
        "The dotted area to the right of the \"Today\" line is the forward forecast "
        "(no actual price yet). Drag the handles of the range slider under the chart "
        "to move the visible date range. Use the \"Options\" checkboxes to show only "
        "the historical fit, only the forward forecast, both, or neither (actual "
        "price only).",
        className="text-muted small",
    ),

    html.H4("Model performance metrics (validation set)", className="mt-4"),
    html.Div(id="metrics-table"),

    html.H4("Recent news and sentiment", className="mt-4"),
    html.Div(id="news-list"),
], className="pt-4")

theory_tab_content = html.Div([
    html.Iframe(
        src="/metodologia",
        style={"width": "100%", "height": "85vh", "border": "none", "display": "block"},
    ),
], className="pt-3")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("📈 Stock Prediction Dashboard", className="my-3"), width="auto"),
        dbc.Col(
            html.Div(id="last-training-badge", className="text-muted small"),
            width="auto", className="ms-auto d-flex align-items-center",
        ),
    ], className="align-items-center", justify="between"),
    dcc.Tabs(id="main-tabs", value="tab-dashboard", children=[
        dcc.Tab(label="Dashboard", value="tab-dashboard", children=[dashboard_tab_content]),
        dcc.Tab(label="Theory / Methodology", value="tab-theory", children=[theory_tab_content]),
    ]),
], fluid=True)


@app.callback(
    Output("price-chart", "figure"),
    Input("ticker-dropdown", "value"),
    Input("model-dropdown", "value"),
    Input("display-options", "value"),
)
def update_chart(tickers, selected_models, display_options):
    tickers = tickers or []
    selected_models = selected_models or []
    display_options = display_options or []
    sentiment_on = "sentiment" in display_options
    fit_on = "fit" in display_options
    future_on = "future" in display_options
    fig = go.Figure()

    all_dates = set()
    last_actual_date = None
    per_ticker_actual = {}
    per_ticker_preds = {}

    for ticker in tickers:
        actual_df = load_actual_prices(ticker)
        per_ticker_actual[ticker] = actual_df
        all_dates.update(actual_df["date"])
        if not actual_df.empty:
            ticker_last = actual_df["date"].max()
            last_actual_date = ticker_last if last_actual_date is None else max(last_actual_date, ticker_last)

        per_ticker_preds[ticker] = {}
        for model_name in selected_models:
            pred_df = load_predictions(ticker, model_name)
            per_ticker_preds[ticker][model_name] = pred_df
            all_dates.update(pred_df["date"])

    # Fondo: % de noticias negativas (rojo) vs. el resto (verde) por día, apilado al 100%.
    if sentiment_on and tickers and all_dates:
        neg_ratio_by_day = load_daily_negative_ratio(tickers)
        bg_dates = sorted(all_dates)
        neg_pct = [neg_ratio_by_day.get(d, 0.0) * 100 for d in bg_dates]
        fig.add_trace(go.Scatter(
            x=bg_dates, y=neg_pct, mode="lines", line=dict(width=0),
            fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.25)",
            yaxis="y2", hoverinfo="skip", showlegend=False, name="Negative news",
        ))
        fig.add_trace(go.Scatter(
            x=bg_dates, y=[100] * len(bg_dates), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(44, 160, 44, 0.20)",
            yaxis="y2", hoverinfo="skip", showlegend=False, name="Non-negative news",
        ))

    for idx, ticker in enumerate(tickers):
        color = ticker_color(ticker, idx)
        name = TICKER_TO_NAME.get(ticker, ticker)
        actual_df = per_ticker_actual[ticker]
        fig.add_trace(go.Scatter(
            x=actual_df["date"], y=actual_df["close"],
            mode="lines", name=f"{name} · actual", line=dict(color=color, width=2.2),
        ))

        for model_name in selected_models:
            pred_df = per_ticker_preds[ticker][model_name]
            if pred_df.empty:
                continue
            # "Ajuste" = predicciones sobre el set de validación (ya sabemos el precio
            # real, sirve para evaluar qué tan bien ajusta el modelo). "Pronóstico
            # futuro" = fechas más allá de "Hoy", donde `actual` todavía es NULL.
            parts = []
            if fit_on:
                parts.append(pred_df[pred_df["actual"].notna()])
            if future_on:
                parts.append(pred_df[pred_df["actual"].isna()])
            if not parts:
                continue
            plot_df = pd.concat(parts).sort_values("date")
            if plot_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df["predicted"],
                mode="lines", name=f"{name} · {model_name}",
                line=dict(color=color, dash=MODEL_DASH.get(model_name, "dot"), width=1.6),
                opacity=0.85,
            ))

    if last_actual_date is not None:
        fig.add_vline(
            x=pd.Timestamp(last_actual_date), line_width=1, line_dash="dash",
            line_color="gray", annotation_text=f"Today ({pd.Timestamp(last_actual_date):%Y-%m-%d})",
            annotation_position="top",
        )

    title = "Actual price vs. forecast" if len(tickers) != 1 else (
        f"{TICKER_TO_NAME.get(tickers[0], tickers[0])} ({tickers[0]}) — Actual price vs. forecast"
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date", yaxis_title="Close price (USD)",
        template="plotly_white", legend=dict(orientation="h", y=-0.25),
        yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False, showticklabels=False),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.08)),
        margin=dict(b=10),
    )
    return fig


@app.callback(Output("last-training-badge", "children"), Input("ticker-dropdown", "value"))
def update_last_training_badge(_tickers):
    last_trained = load_last_training_date()
    if last_trained is None:
        return "Last training: N/A"
    return f"Last training: {last_trained:%Y-%m-%d}"


@app.callback(Output("metrics-table", "children"), Input("ticker-dropdown", "value"))
def update_metrics_table(tickers):
    tickers = tickers or []
    dfs = [load_metrics(t) for t in tickers]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return html.P("No metrics yet. Run model training first.")
    df = pd.concat(dfs, ignore_index=True)
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)


def extract_visible_range(relayout_data):
    """Extrae (inicio, fin) del rango de fechas visible en la gráfica a partir del
    `relayoutData` de Plotly (se llena al arrastrar el selector de rango o hacer
    zoom). Regresa None si la gráfica está en su vista completa (sin acotar)."""
    if not relayout_data:
        return None
    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        return relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]
    rng = relayout_data.get("xaxis.range")
    if rng and len(rng) == 2:
        return rng[0], rng[1]
    return None


@app.callback(
    Output("news-list", "children"),
    Input("ticker-dropdown", "value"),
    Input("price-chart", "relayoutData"),
)
def update_news_list(tickers, relayout_data):
    tickers = tickers or []
    dfs = [load_news(t) for t in tickers]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return html.P("No news downloaded yet for these companies.")

    df = pd.concat(dfs, ignore_index=True).sort_values("published_at", ascending=False)

    date_range = extract_visible_range(relayout_data)
    range_note = None
    if date_range:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df[(df["published_at"] >= start) & (df["published_at"] <= end)]
        range_note = (
            f"Showing news between {start:%Y-%m-%d} and {end:%Y-%m-%d} "
            f"(the range currently visible on the chart) — {len(df)} found."
        )
    else:
        df = df.head(30)
        range_note = (
            "Showing the 30 most recent news items. Drag the date range slider "
            "under the chart back in time (history goes back to 2025) to see "
            "news from that period here."
        )

    badge_color = {"positive": "success", "negative": "danger", "neutral": "secondary"}
    items = []
    for _, row in df.head(200).iterrows():
        date_str = row["published_at"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["published_at"]) else "unknown date"
        items.append(dbc.ListGroupItem([
            html.Span(date_str, className="text-muted small me-2", style={"whiteSpace": "nowrap"}),
            dbc.Badge(row["ticker"], color="dark", className="me-2"),
            dbc.Badge(row["label"] or "N/A", color=badge_color.get(row["label"], "light"), className="me-2"),
            html.A(row["title"], href=row["link"], target="_blank") if row["link"] else html.Span(row["title"]),
        ]))
    return html.Div([
        html.P(range_note, className="text-muted small"),
        dbc.ListGroup(items),
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8050, use_reloader=False, threaded=True)
