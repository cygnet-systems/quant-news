"""SQLAlchemy ORM models for the persistence layer.

Tables:
    data_snapshots       — content hash of input data per (symbol, date, data_type)
    report_catalog       — metadata + S3 key for rendered reports
    recommendation_runs  — cached recommendation synthesis results
    stock_prices         — cached OHLCV price data
    stock_info           — cached company fundamentals
    cache_metadata       — cache freshness tracking per (symbol, data_type)
    news_articles        — cached real-time news articles
    model_predictions    — model prediction results with evaluation tracking
    historical_news      — Alpha Vantage historical news for feature building
    strategy_evaluations — per-prediction strategy evaluation results
    strategy_metrics     — aggregate strategy performance metrics
    trading_agent_reports — TradingAgents research reports
"""

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared base for all ORM models."""

    type_annotation_map = {dict: JSONB}


# ---------------------------------------------------------------------------
# Existing tables (unchanged from 001)
# ---------------------------------------------------------------------------


class DataSnapshot(Base):
    """Content hash of the input data used for a prediction or report.

    Cache key: (symbol, trade_date, data_type).
    If the content_hash matches, downstream results are still valid.
    """

    __tablename__ = "data_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)
    data_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    record_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", "data_type", name="uq_snapshot_key"),
        Index("ix_snapshot_lookup", "symbol", "trade_date", "data_type"),
    )


class ReportCatalog(Base):
    """Metadata for a rendered report stored in object storage.

    report_type: 'ai_report', 'full_analysis', 'trading_agents_research'
    storage_key: S3 object key (e.g. 'reports/EIX/2026-07-13/full_analysis.md')
    """

    __tablename__ = "report_catalog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str | None] = mapped_column(String(16))
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)
    report_type: Mapped[str] = mapped_column(String(64), nullable=False)
    input_data_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    file_format: Mapped[str] = mapped_column(String(8), nullable=False, default="md")
    size_bytes: Mapped[int | None] = mapped_column(Integer)

    metadata_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Ownership/visibility: NULL owner or is_public=True means visible to
    # everyone (the pre-auth default). owner_uid is the Cygnet SSO uid.
    owner_uid: Mapped[str | None] = mapped_column(String(64))
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    __table_args__ = (
        UniqueConstraint(
            "symbol", "trade_date", "report_type", "input_data_hash",
            name="uq_report_cache",
        ),
        Index("ix_report_lookup", "symbol", "trade_date", "report_type"),
        Index("ix_report_owner", "owner_uid"),
    )


class RecommendationRun(Base):
    """Cached recommendation synthesis from the second-pass LLM.

    Keyed by (trade_date, input_data_hash) — one recommendation per
    Full Analysis run across all symbols in that run.
    """

    __tablename__ = "recommendation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)
    symbols_csv: Mapped[str] = mapped_column(Text, nullable=False)
    input_data_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model_used: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_used: Mapped[str] = mapped_column(String(32), nullable=False)
    result_json: Mapped[dict | None] = mapped_column(JSONB)

    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    owner_uid: Mapped[str | None] = mapped_column(String(64))
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    __table_args__ = (
        UniqueConstraint(
            "trade_date", "input_data_hash", name="uq_recommendation_cache"
        ),
        Index("ix_recommendation_lookup", "trade_date"),
        Index("ix_recommendation_owner", "owner_uid"),
    )


# ---------------------------------------------------------------------------
# New tables (migrated from DuckDB)
# ---------------------------------------------------------------------------


class StockPrice(Base):
    """Cached OHLCV price data per (symbol, date)."""

    __tablename__ = "stock_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float | None] = mapped_column(Double)
    high: Mapped[float | None] = mapped_column(Double)
    low: Mapped[float | None] = mapped_column(Double)
    close: Mapped[float | None] = mapped_column(Double)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    dividends: Mapped[float | None] = mapped_column(Double)
    stock_splits: Mapped[float | None] = mapped_column(Double)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_stock_price_key"),
        Index("ix_stock_price_lookup", "symbol", "date"),
    )


class StockInfo(Base):
    """Cached company fundamentals — one row per symbol."""

    __tablename__ = "stock_info"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(String)
    sector: Mapped[str | None] = mapped_column(String)
    industry: Mapped[str | None] = mapped_column(String)
    market_cap: Mapped[int | None] = mapped_column(BigInteger)
    current_price: Mapped[float | None] = mapped_column(Double)
    previous_close: Mapped[float | None] = mapped_column(Double)
    pe_ratio: Mapped[float | None] = mapped_column(Double)
    dividend_yield: Mapped[float | None] = mapped_column(Double)
    fifty_two_week_high: Mapped[float | None] = mapped_column(Double)
    fifty_two_week_low: Mapped[float | None] = mapped_column(Double)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    avg_volume: Mapped[int | None] = mapped_column(BigInteger)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CacheMetadata(Base):
    """Cache freshness tracking per (symbol, data_type)."""

    __tablename__ = "cache_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    data_type: Mapped[str] = mapped_column(String(32), nullable=False)
    period: Mapped[str | None] = mapped_column(String(8))
    last_updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    record_count: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("symbol", "data_type", name="uq_cache_metadata_key"),
    )


class NewsArticle(Base):
    """Cached real-time news articles per symbol."""

    __tablename__ = "news_articles"

    id: Mapped[str] = mapped_column(String, primary_key=True, autoincrement=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[str | None] = mapped_column(Text)
    sentiment: Mapped[str | None] = mapped_column(String(16))
    sentiment_score: Mapped[float | None] = mapped_column(Double)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    topics_json: Mapped[str | None] = mapped_column(Text)
    overall_sentiment_score: Mapped[float | None] = mapped_column(Double)
    overall_sentiment_label: Mapped[str | None] = mapped_column(String(32))
    ticker_relevance_score: Mapped[float | None] = mapped_column(Double)
    impact: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        Index("ix_news_article_lookup", "symbol", "fetched_at"),
    )


class ModelPrediction(Base):
    """Model prediction result with evaluation tracking.

    Replaces both the old PredictionRun (cache invalidation via
    input_data_hash) and the DuckDB model_predictions table (evaluation
    tracking with actual_close, was_correct, pnl).

    Primary key: string composite ID, e.g. "AAPL_kronos_mini_20260713".
    """

    __tablename__ = "model_predictions"

    id: Mapped[str] = mapped_column(String, primary_key=True, autoincrement=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    prediction_date: Mapped[date] = mapped_column(Date, nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    decision: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Double)
    up_probability: Mapped[float | None] = mapped_column(Double)
    predicted_close: Mapped[float | None] = mapped_column(Double)
    previous_close: Mapped[float | None] = mapped_column(Double)

    # Evaluation fields (filled after target_date has passed)
    actual_close: Mapped[float | None] = mapped_column(Double)
    was_correct: Mapped[bool | None] = mapped_column(Boolean)
    pnl_dollars: Mapped[float | None] = mapped_column(Double)

    # Model metadata
    model_version: Mapped[str | None] = mapped_column(String(32))
    training_samples: Mapped[int | None] = mapped_column(Integer)
    feature_values_json: Mapped[str | None] = mapped_column(Text)
    details_json: Mapped[dict | None] = mapped_column(JSONB)

    # Cache invalidation
    input_data_hash: Mapped[str | None] = mapped_column(String(64))

    # Timing
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Ownership/visibility (default public — pre-auth behavior unchanged)
    owner_uid: Mapped[str | None] = mapped_column(String(64))
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    __table_args__ = (
        Index("ix_model_pred_sym_model_date", "symbol", "model_name", "prediction_date"),
        Index("ix_model_pred_sym_date", "symbol", "prediction_date"),
        Index("ix_model_pred_hash", "input_data_hash"),
        Index("ix_model_pred_owner", "owner_uid"),
    )


class HistoricalNews(Base):
    """Alpha Vantage historical news articles for feature building."""

    __tablename__ = "historical_news"

    id: Mapped[str] = mapped_column(String, primary_key=True, autoincrement=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    published_date: Mapped[date | None] = mapped_column(Date)
    title: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)
    topics_json: Mapped[str | None] = mapped_column(Text)
    overall_sentiment_score: Mapped[float | None] = mapped_column(Double)
    overall_sentiment_label: Mapped[str | None] = mapped_column(String(32))
    ticker_sentiment_score: Mapped[float | None] = mapped_column(Double)
    ticker_relevance_score: Mapped[float | None] = mapped_column(Double)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_historical_news_lookup", "symbol", "published_date"),
    )


class StrategyEvaluation(Base):
    """Per-prediction strategy evaluation result.

    One-to-many with ModelPrediction: each prediction can be evaluated
    by multiple strategies.
    """

    __tablename__ = "strategy_evaluations"

    id: Mapped[str] = mapped_column(String, primary_key=True, autoincrement=False)
    prediction_id: Mapped[str] = mapped_column(
        String, ForeignKey("model_predictions.id"), nullable=False
    )
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[str | None] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(8), nullable=False)
    position_size: Mapped[float] = mapped_column(Double, default=1000.0)
    entry_price: Mapped[float | None] = mapped_column(Double)
    exit_price: Mapped[float | None] = mapped_column(Double)
    pnl_dollars: Mapped[float | None] = mapped_column(Double)
    was_correct: Mapped[bool | None] = mapped_column(Boolean)
    signal_metadata_json: Mapped[str | None] = mapped_column(Text)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_strategy_eval_lookup", "strategy_name", "prediction_id"),
    )


class StrategyMetrics(Base):
    """Aggregate strategy performance metrics (vectorbt results)."""

    __tablename__ = "strategy_metrics"

    id: Mapped[str] = mapped_column(String, primary_key=True, autoincrement=False)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(16))
    period: Mapped[str] = mapped_column(String(16), nullable=False)
    sharpe_ratio: Mapped[float | None] = mapped_column(Double)
    sortino_ratio: Mapped[float | None] = mapped_column(Double)
    max_drawdown: Mapped[float | None] = mapped_column(Double)
    win_rate: Mapped[float | None] = mapped_column(Double)
    total_pnl: Mapped[float | None] = mapped_column(Double)
    total_trades: Mapped[int | None] = mapped_column(Integer)
    metrics_json: Mapped[dict | None] = mapped_column(JSONB)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_strategy_metrics_lookup", "strategy_name", "symbol"),
    )


class TradingAgentReport(Base):
    """TradingAgents research report (LLM-generated)."""

    __tablename__ = "trading_agent_reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, autoincrement=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    decision: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Double)
    report_text: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    owner_uid: Mapped[str | None] = mapped_column(String(64))
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    __table_args__ = (
        Index("ix_trading_agent_report_lookup", "symbol", "trade_date"),
        Index("ix_trading_agent_report_owner", "owner_uid"),
    )


class ActivityLog(Base):
    """Durable audit trail of pipeline activity, one row per emitted event.

    The diskcache progress feed is a 300-event rolling window that each new
    run clears -- fine for driving the live panel, useless as a record. This
    is the record: append-only, queryable, and survives cache wipes and
    restarts. `run_id` groups the events of a single Predict/Full Analysis
    invocation; `user_id` is a single local identity today but is indexed so
    per-user filtering works unchanged when there is more than one.
    """

    __tablename__ = "activity_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="local")
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    run_title: Mapped[str | None] = mapped_column(Text)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_activity_user_time", "user_id", "created_at"),
        Index("ix_activity_run", "run_id"),
    )
