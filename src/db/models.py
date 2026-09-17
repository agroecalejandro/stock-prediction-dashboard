"""Esquema de base de datos (SQLAlchemy ORM)."""
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class StockPrice(Base):
    """Precio histórico/diario de una acción."""

    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)

    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)


class Prediction(Base):
    """Predicción de precio generada por un modelo para una fecha objetivo."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False, index=True)
    target_date = Column(Date, nullable=False, index=True)
    predicted_close = Column(Float, nullable=False)
    actual_close = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "model_name", "target_date", name="uq_pred"),
    )


class ModelMetric(Base):
    """Métricas de desempeño de un modelo sobre el set de validación."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False, index=True)
    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    trained_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "model_name", name="uq_metric"),
    )


class NewsSentiment(Base):
    """Noticia financiera con su sentimiento clasificado."""

    __tablename__ = "news_sentiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False, index=True)
    published_at = Column(DateTime, nullable=True, index=True)
    title = Column(String, nullable=False)
    link = Column(String, nullable=True)
    sentiment_label = Column(String)  # positive / negative / neutral
    sentiment_score = Column(Float)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("ticker", "title", name="uq_news"),)
