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

    run_id: Mapped[str | None] = mapped_column(String(36))
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
        Index("ix_recommendation_run", "run_id"),
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

    # Whether the news the models were meant to read actually arrived:
    # "ok" (articles), "empty" (source answered, window genuinely quiet), or
    # "unavailable" (the source failed). A prediction made blind scores the
    # same as any other, so without this column the scoreboard silently mixes
    # supported calls with unsupported ones. NULL means the run predates this
    # column, which is not the same as "ok".
    news_status: Mapped[str | None] = mapped_column(String(16))

    # Which combination method produced an ensemble call. NULL for every other
    # model, and for ensemble rows written before it was recorded.
    ensemble_method: Mapped[str | None] = mapped_column(String(32))

    # Cache invalidation
    input_data_hash: Mapped[str | None] = mapped_column(String(64))

    # The pipeline run that produced this row (activity_log/llm_usage share
    # the id). NULL for rows written before runs were stamped.
    run_id: Mapped[str | None] = mapped_column(String(36))

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
        Index("ix_model_pred_news_status", "news_status", "prediction_date"),
        Index("ix_model_pred_ens_method", "ensemble_method", "prediction_date"),
        Index("ix_model_pred_run", "run_id"),
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
    # Full timestamp (migration 013): the overnight window needs the hour.
    # NULL on rows written before it existed.
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_historical_news_lookup", "symbol", "published_date"),
        Index("ix_historical_news_published_at", "symbol", "published_at"),
    )


class NewsCoverage(Base):
    """Which (symbol, day) pairs have been fetched from the news vendor.

    The ledger behind the point-in-time news store: a run computes the days
    of its window that are NOT here and fetches only those. A day within the
    live edge is re-fetched when its fetched_at is older than the live TTL,
    because the vendor keeps indexing a session's articles for a while.
    """

    __tablename__ = "news_coverage"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


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
    run_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    owner_uid: Mapped[str | None] = mapped_column(String(64))
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    __table_args__ = (
        Index("ix_trading_agent_report_lookup", "symbol", "trade_date"),
        Index("ix_trading_agent_report_owner", "owner_uid"),
        Index("ix_trading_agent_report_run", "run_id"),
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
    # Structured facts behind the message (counts, windows, hashes) for the
    # Trace view. Lives ONLY here — the diskcache feed events stay small.
    payload: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_activity_user_time", "user_id", "created_at"),
        Index("ix_activity_run", "run_id"),
    )


class ScheduledJob(Base):
    """A recurring job the app runs itself, editable from the dashboard.

    The schedule lives in the database rather than in config so it survives a
    redeploy and can be changed without one. The app owns the clock: there is
    no external cron, and nothing here depends on the machine the container
    happens to be running on beyond it being up.

    ``last_success_date`` is what makes catch-up safe. On startup a job whose
    window has passed today and that has no success recorded for today runs
    once; a redeploy loop cannot turn that into repeated runs.
    """

    __tablename__ = "scheduled_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False,
                                          server_default=text("true"))

    # Local-time schedule. Stored as parts rather than a cron string so the UI
    # can present a time picker and a weekday choice without parsing.
    hour: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    minute: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    days_of_week: Mapped[str] = mapped_column(String(32), nullable=False,
                                              server_default="mon-fri")
    timezone: Mapped[str] = mapped_column(String(64), nullable=False,
                                          server_default="US/Eastern")

    symbols_csv: Mapped[str | None] = mapped_column(Text)
    params_json: Mapped[dict | None] = mapped_column(JSONB)

    # Denormalised outcome of the most recent run — the UI reads these on
    # every render, and catch-up needs last_success_date without scanning
    # the history table.
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(String(16))
    last_detail: Mapped[str | None] = mapped_column(Text)
    last_duration_ms: Mapped[int | None] = mapped_column(Integer)
    last_success_date: Mapped[str | None] = mapped_column(String(10))

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    owner_uid: Mapped[str | None] = mapped_column(String(64))
    # Private schedules exist per user; public ones are visible to everyone
    # (and to the Home jobs strip). Legacy rows stay public.
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False,
                                            server_default=text("true"),
                                            default=True)


class JobRun(Base):
    """One execution of a scheduled job — the audit trail behind the status."""

    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # schedule | manual | catchup
    trigger: Mapped[str] = mapped_column(String(16), nullable=False, default="schedule")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    detail: Mapped[str | None] = mapped_column(Text)
    owner_uid: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        Index("ix_job_runs_job_time", "job_id", "started_at"),
    )


class LLMUsage(Base):
    """One row per LLM API call: tokens, cost, and what the call was for.

    Internal telemetry — nothing here is ever fed back into a prompt. Token
    counts come from the provider response and are exact. Cost is derived,
    so the RATES USED are stored on the row: a later price change cannot
    silently rewrite the cost of calls already made.

    ``stage`` is what the call was for (research / ai_report /
    recommendations / …) so spend can be attributed to the thing that
    produced it rather than to a model name alone.
    """

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(String(36))
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    symbol: Mapped[str | None] = mapped_column(String(16))
    trade_date: Mapped[str | None] = mapped_column(String(10))

    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(64))

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Dollars per MILLION tokens, as applied to this row.
    input_rate_per_mtok: Mapped[float | None] = mapped_column(Double)
    output_rate_per_mtok: Mapped[float | None] = mapped_column(Double)
    cost_usd: Mapped[float | None] = mapped_column(Double)

    duration_ms: Mapped[int | None] = mapped_column(Integer)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    error: Mapped[str | None] = mapped_column(Text)

    owner_uid: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        Index("ix_llm_usage_time", "created_at"),
        Index("ix_llm_usage_stage", "stage", "created_at"),
        Index("ix_llm_usage_run", "run_id"),
    )


class LLMTrace(Base):
    """Full request/response capture for one PHYSICAL LLM API call.

    One row per attempt that reached a provider — retries and failover
    attempts each get their own row with a shared context and an incremented
    ``attempt``. Captured BEFORE any parsing, so an unparseable or truncated
    response is preserved exactly as the provider sent it.

    Division of labour with ``llm_usage``: that table remains the SOLE
    token/cost record; this one holds the bodies (system prompt, prompt, raw
    response) and the request parameters actually sent. ``usage_id`` links
    the paired llm_usage row when the call also produced one.

    ``section`` narrows ``stage`` to the report section the call served
    (e.g. "research:BE", "ai_report:overall", "recommendations").
    """

    __tablename__ = "llm_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    usage_id: Mapped[int | None] = mapped_column(Integer)
    run_id: Mapped[str | None] = mapped_column(String(36))
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    section: Mapped[str | None] = mapped_column(String(64))
    symbol: Mapped[str | None] = mapped_column(String(16))
    trade_date: Mapped[date | None] = mapped_column(Date)

    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(64))

    system_prompt: Mapped[str | None] = mapped_column(Text)
    prompt: Mapped[str | None] = mapped_column(Text)
    response: Mapped[str | None] = mapped_column(Text)
    # The request parameters actually sent (temperature, max_tokens,
    # reasoning effort, thinking config, …) minus the bodies above.
    params_json: Mapped[dict | None] = mapped_column(JSONB)

    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    owner_uid: Mapped[str | None] = mapped_column(String(64))
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    __table_args__ = (
        Index("ix_llm_traces_run", "run_id"),
        Index("ix_llm_traces_time", "created_at"),
        Index("ix_llm_traces_owner", "owner_uid"),
    )


class WatchlistHistory(Base):
    """Every distinct symbol group a user has actually pulled data for.

    Previously this was five entries in the browser's localStorage, so it
    vanished with a cleared cache and never reached a second machine. Rows are
    never deleted: the subset-merge that keeps the recent-chips list readable
    (REX superseding REX+WGO's prefix) is a presentation rule, and applying it
    to storage would throw away the history it exists to preserve.

    Uniqueness is enforced by an expression index on COALESCE(owner_uid, '')
    rather than a UniqueConstraint, because Postgres treats NULL owners as
    distinct and anonymous sessions would never bump use_count.
    """

    __tablename__ = "watchlist_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_uid: Mapped[str | None] = mapped_column(String(64))
    # Sorted and comma-joined, so the same set always keys the same row.
    symbols_csv: Mapped[str] = mapped_column(Text, nullable=False)

    first_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    use_count: Mapped[int] = mapped_column(Integer, nullable=False,
                                           server_default=text("1"))

    __table_args__ = (
        Index("ix_watchlist_history_recent", "owner_uid", "last_used_at"),
    )
