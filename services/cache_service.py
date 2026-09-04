"""PostgreSQL caching service for stock data and model predictions.

Provides a caching layer using PostgreSQL via SQLAlchemy to minimize API calls
and enable fast local data access with SQL query capabilities.

Migrated from DuckDB, same public API, Postgres backend.
"""

import hashlib
import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import delete, func, select, text, update

from config import APP
from db.session import get_session, get_engine

logger = logging.getLogger(__name__)


def _sanitize_json(obj):
    """Recursively replace NaN/Inf floats with None for JSONB columns.

    json.dumps(float('nan')) emits the literal `NaN`, which Postgres JSONB
    rejects. The Dash UI path survives only because dcc.Store's encoder
    converts NaN to null in transit; any server-side caller (scripts,
    benchmarks) would fail on insert. Features may now legitimately be NaN
    ("missing"), so sanitize at the persistence boundary.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def _usable_price(v) -> bool:
    """A price a row can be scored against: present, finite, positive.

    NaN slips past every ``<= 0`` / ``is None`` guard (in Python NaN fails all
    comparisons; in Postgres NaN sorts ABOVE infinity, so ``close > 0`` is
    true for it). One pre-market vendor fetch on 2026-08-18 wrote NaN closes
    into stock_prices, which store_prediction copied into previous_close for
    an entire day's predictions and the evaluator then crashed serializing
    NaN into JSONB, killing the whole run's transaction.
    """
    return v is not None and not isinstance(v, str) \
        and math.isfinite(v) and v > 0


def _parse_topics(topics_json) -> list:
    """Deserialize a HistoricalNews.topics_json value to the list form the
    feature builder consumes. The column may hold a JSON string, an already-
    decoded list (JSONB), or NULL."""
    if isinstance(topics_json, list):
        return topics_json
    if isinstance(topics_json, str) and topics_json:
        try:
            parsed = json.loads(topics_json)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []
    return []


def _pred_to_dict(r, include_details: bool = False) -> dict:
    """Serialize a ModelPrediction row for the UI.

    A superset of what list_all_predictions returns: evaluated_at and
    target_date are what let a caller tell "not scored yet" apart from
    "scored as a HOLD", which was_correct alone cannot express.
    """
    out = {
        "id": r.id,
        "symbol": r.symbol,
        "model_name": r.model_name,
        "prediction_date": str(r.prediction_date),
        "target_date": str(r.target_date),
        "decision": r.decision,
        "confidence": r.confidence,
        "up_probability": r.up_probability,
        "predicted_close": r.predicted_close,
        "previous_close": r.previous_close,
        "actual_close": r.actual_close,
        "was_correct": r.was_correct,
        "pnl_dollars": r.pnl_dollars,
        "model_version": r.model_version,
        "duration_ms": r.duration_ms,
        # "ok" | "empty" | "unavailable" | None (predates the column). A call
        # made while the news source was down is not comparable to one made on
        # a full window, so the UI marks it rather than averaging it in.
        "news_status": r.news_status,
        # The window the run was made with, so the input-data export can
        # rebuild exactly those inputs (None = predates the stamp).
        "news_window_days": ((r.details_json or {}).get("news_window_days")
                             if isinstance(r.details_json, dict) else None),
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "evaluated_at": r.evaluated_at.isoformat() if r.evaluated_at else None,
    }
    if include_details:
        out["details"] = r.details_json or {}
    return out


def _current_uid():
    """Uid to attribute writes/reads to: the signed-in user in a request
    thread, or the owning user of a scheduled run (QUANTNEWS_RUN_OWNER in
    the run subprocess). None means anonymous/public."""
    try:
        from services.auth_service import effective_uid
        return effective_uid()
    except Exception:
        return None


def _default_public() -> bool:
    """Visibility for new rows: public unless this process is a private
    scheduled run (QUANTNEWS_RUN_PUBLIC=0)."""
    try:
        from services.auth_service import run_is_public
        return run_is_public()
    except Exception:
        return True


def _visible(model_cls):
    """Row-visibility clause: public rows (or legacy NULL-owner rows) plus
    the signed-in user's own. With everything public today this filters
    nothing; it becomes load-bearing the moment private rows exist."""
    cond = model_cls.is_public.is_(True) | model_cls.owner_uid.is_(None)
    uid = _current_uid()
    if uid:
        cond = cond | (model_cls.owner_uid == uid)
    return cond


def _by_run_kind(q, kind: str | None):
    """Restrict a model_predictions select to rows written by runs of one
    kind (scheduled | manual), through a LEFT JOIN on analysis_runs.

    A prediction with no run row (run_id NULL, or an id from before
    analysis_runs existed) counts as scheduled: every prediction written
    before that table was the daily job's, and the Home board must keep
    showing them under the scheduled cutoffs. None leaves the query alone.
    """
    if not kind:
        return q
    from db.models import AnalysisRun, ModelPrediction
    return (q.outerjoin(AnalysisRun, AnalysisRun.run_id == ModelPrediction.run_id)
             .where(func.coalesce(AnalysisRun.kind, "scheduled") == kind))


# The text-SQL twin of _by_run_kind, for the readers that hand-write SQL
# (calibration is on the ORM path; the evening mail and the Alpha Lab are
# not). Both must keep agreeing that a prediction with no run row is the
# daily job's. It is a correlated subquery rather than a join so it drops
# into an existing WHERE without touching the caller's FROM.
SCHEDULED_ONLY_SQL = (
    "coalesce((select r.kind from analysis_runs r "
    "where r.run_id = model_predictions.run_id), 'scheduled') = 'scheduled'"
)


def _run_kind(session, run_id: str | None) -> str:
    """The kind of the run a write belongs to.

    No run row means scheduled, the same COALESCE `_by_run_kind` reads with:
    the headless writers and every row written before analysis_runs existed
    belong to the daily board.
    """
    if not run_id:
        return "scheduled"
    from db.models import AnalysisRun
    kind = session.execute(
        select(AnalysisRun.kind).where(AnalysisRun.run_id == run_id)
    ).scalar_one_or_none()
    return kind or "scheduled"


def _prediction_id(symbol: str, model_name: str, pred_date: date,
                   run_id: str | None, kind: str) -> str:
    """Primary key for a prediction row, run-scoped for manual runs.

    An ad-hoc run of a watchlist name lands on the same (symbol, model,
    cutoff) as the daily job, and one shared id made the merge REPLACE the
    scheduled row: its run_id was re-stamped manual, so `_by_run_kind`
    dropped it from the Scheduled board (the name read "not run" for a day
    the job did call it) and the merge's deliberate score reset threw away
    that day's evaluation. Manual rows therefore get their own id space and
    the scheduled row stays exactly as the job wrote it. Scheduled ids keep
    the historic shape, so every stored row and its evaluation stay where
    the evaluator and the backtests already look for them.
    """
    pred_id = f"{symbol}_{model_name}_{pred_date:%Y%m%d}"
    if kind == "manual" and run_id:
        # The whole id, not a prefix: this is a primary key, and a collision
        # would silently overwrite another run's call.
        return f"{pred_id}_{run_id}"
    return pred_id


def _fallback_run_id(what: str):
    """Run identity for a write whose caller passed no run_id.

    Only the headless paths (benchmark, scripts) may land here: every UI and
    scheduled writer threads the analysis_runs id explicitly, because the
    process-local guess below cannot name a run that lives in another
    process (the server persisting the model subprocess's output stamped
    everything "adhoc", and two users' runs shared one id before that).
    The warning is the tell that a caller was missed.
    """
    try:
        from services import progress_service as prog
        run_id = prog.current_run_id()
    except Exception:
        run_id = None
    logger.warning("%s written without an explicit run_id; falling back to "
                   "the process-local run %r", what, run_id)
    return run_id


class CacheService:
    """Postgres-backed caching service for stock data.

    Provides methods to cache and retrieve stock prices, company info,
    and news articles with manual refresh capability.
    """

    def __init__(self) -> None:
        Path("cache/exports").mkdir(parents=True, exist_ok=True)

    def _period_to_days(self, period: str) -> int:
        if period == "ytd":
            jan1 = datetime(datetime.now().year, 1, 1)
            # Floor of a week so early-January YTD still loads a usable window.
            return max((datetime.now() - jan1).days, 7)
        period_map = {
            "5d": 7, "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "max": 7300,
        }
        if period not in period_map:
            logger.warning(f"unknown price period {period!r}: treating as 1y")
        return period_map.get(period, 365)

    def is_cached(self, symbol: str, data_type: str = "prices") -> bool:
        from db.models import CacheMetadata
        with get_session() as session:
            stmt = select(func.count()).select_from(CacheMetadata).where(
                CacheMetadata.symbol == symbol.upper(),
                CacheMetadata.data_type == data_type,
            )
            return session.execute(stmt).scalar() > 0

    def get_cache_info(self, symbol: str, data_type: str = "prices") -> Optional[dict]:
        from db.models import CacheMetadata
        with get_session() as session:
            row = session.execute(
                select(CacheMetadata).where(
                    CacheMetadata.symbol == symbol.upper(),
                    CacheMetadata.data_type == data_type,
                )
            ).scalar_one_or_none()
            if row:
                return {
                    "period": row.period,
                    "last_updated": row.last_updated,
                    "record_count": row.record_count,
                }
        return None

    # =========================================================================
    # News cache
    # =========================================================================

    def get_cached_news(self, symbol: str, max_age_minutes: int = 15) -> Optional[list[dict]]:
        from db.models import NewsArticle
        symbol = symbol.upper().strip()
        cutoff_time = datetime.now() - pd.Timedelta(minutes=max_age_minutes)

        with get_session() as session:
            rows = session.execute(
                select(NewsArticle)
                .where(NewsArticle.symbol == symbol, NewsArticle.fetched_at >= cutoff_time)
                .order_by(NewsArticle.published_at.desc())
            ).scalars().all()

            if not rows:
                return None

            return [
                {
                    "id": r.id,
                    "symbol": r.symbol,
                    "title": r.title,
                    "source": r.source,
                    "url": r.url,
                    "published_at": r.published_at,
                    "summary": r.summary,
                    "sentiment": r.sentiment,
                    "sentiment_score": r.sentiment_score,
                    "fetched_at": r.fetched_at,
                    "topics": json.loads(r.topics_json) if r.topics_json else None,
                    "overall_sentiment_score": r.overall_sentiment_score,
                    "overall_sentiment_label": r.overall_sentiment_label,
                    "ticker_relevance_score": r.ticker_relevance_score,
                    "impact": r.impact,
                }
                for r in rows
            ]

    def cache_news(self, symbol: str, articles: list) -> None:
        from db.models import NewsArticle, CacheMetadata
        symbol = symbol.upper().strip()
        now = datetime.now(timezone.utc)

        with get_session() as session:
            session.execute(delete(NewsArticle).where(NewsArticle.symbol == symbol))

            for article in articles:
                if hasattr(article, "id"):
                    a = article
                    row = NewsArticle(
                        id=a.id, symbol=symbol, title=a.title, source=a.source,
                        url=a.url, published_at=a.published_at, summary=a.summary,
                        sentiment=a.sentiment, sentiment_score=a.sentiment_score,
                        fetched_at=now,
                        topics_json=json.dumps(a.topics) if a.topics else None,
                        overall_sentiment_score=a.overall_sentiment_score,
                        overall_sentiment_label=a.overall_sentiment_label,
                        ticker_relevance_score=a.ticker_relevance_score,
                        impact=a.impact,
                    )
                else:
                    row = NewsArticle(
                        id=article.get("id"), symbol=symbol,
                        title=article.get("title"), source=article.get("source"),
                        url=article.get("url"), published_at=article.get("published_at"),
                        summary=article.get("summary"), sentiment=article.get("sentiment"),
                        sentiment_score=article.get("sentiment_score"), fetched_at=now,
                        topics_json=json.dumps(article.get("topics")) if article.get("topics") else None,
                        overall_sentiment_score=article.get("overall_sentiment_score"),
                        overall_sentiment_label=article.get("overall_sentiment_label"),
                        ticker_relevance_score=article.get("ticker_relevance_score"),
                        impact=article.get("impact"),
                    )
                session.merge(row)

            existing_meta = session.execute(
                select(CacheMetadata).where(
                    CacheMetadata.symbol == symbol,
                    CacheMetadata.data_type == "news",
                )
            ).scalar_one_or_none()
            if existing_meta:
                existing_meta.last_updated = now
                existing_meta.record_count = len(articles)
            else:
                session.add(CacheMetadata(
                    symbol=symbol, data_type="news", period=None,
                    last_updated=now, record_count=len(articles),
                ))

    # =========================================================================
    # Stock prices
    # =========================================================================

    def get_stock_prices(
        self, symbol: str, period: str = APP.DEFAULT_PERIOD, force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        from db.models import StockPrice
        symbol = symbol.upper().strip()
        metadata = {"from_cache": False, "api_error": None, "cache_time": None}

        days = self._period_to_days(period)
        start_date = datetime.now() - pd.Timedelta(days=days)

        if not force_refresh and self.is_cached(symbol, "prices"):
            with get_session() as session:
                oldest, newest = session.execute(
                    select(func.min(StockPrice.date), func.max(StockPrice.date))
                    .where(StockPrice.symbol == symbol)
                ).one()

            # Coverage at BOTH ends. The old check only tested the oldest
            # bar, so a cache whose newest bar was a week old was served as
            # fresh (from_cache=True, no api_error). Every metric block and
            # feature vector then ran on stale prices. 4 calendar days
            # allows a long weekend.
            reaches_present = (
                newest is not None
                and pd.to_datetime(newest) >= datetime.now() - pd.Timedelta(days=4))
            if oldest and reaches_present:
                oldest_dt = pd.to_datetime(oldest)
                if oldest_dt <= (start_date + pd.Timedelta(days=7)):
                    df = pd.read_sql(
                        select(
                            StockPrice.date, StockPrice.open, StockPrice.high,
                            StockPrice.low, StockPrice.close, StockPrice.volume,
                            StockPrice.dividends, StockPrice.stock_splits,
                        ).where(
                            StockPrice.symbol == symbol,
                            StockPrice.date >= start_date.strftime("%Y-%m-%d"),
                        ).order_by(StockPrice.date),
                        get_engine(),
                    )
                    if not df.empty:
                        df.set_index("date", inplace=True)
                        df.index = pd.to_datetime(df.index)
                        df = df.rename(columns={
                            "open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume",
                            "dividends": "Dividends", "stock_splits": "Stock Splits",
                        })
                        cache_info = self.get_cache_info(symbol, "prices")
                        metadata["from_cache"] = True
                        metadata["cache_time"] = cache_info.get("last_updated") if cache_info else None
                        return df, metadata
            elif oldest and not reaches_present:
                logger.info(f"{symbol}: cached prices end {newest}: refetching")

        from services.stock_data import fetch_stock_data
        try:
            df = fetch_stock_data(symbol, period)
            self._cache_prices(symbol, df, period)
            metadata["from_cache"] = False
            return df, metadata
        except Exception as e:
            if self.is_cached(symbol, "prices"):
                fallback_df = pd.read_sql(
                    select(
                        StockPrice.date, StockPrice.open, StockPrice.high,
                        StockPrice.low, StockPrice.close, StockPrice.volume,
                        StockPrice.dividends, StockPrice.stock_splits,
                    ).where(
                        StockPrice.symbol == symbol,
                        StockPrice.date >= start_date.strftime("%Y-%m-%d"),
                    ).order_by(StockPrice.date),
                    get_engine(),
                )
                if not fallback_df.empty:
                    fallback_df.set_index("date", inplace=True)
                    fallback_df.index = pd.to_datetime(fallback_df.index)
                    fallback_df = fallback_df.rename(columns={
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume",
                        "dividends": "Dividends", "stock_splits": "Stock Splits",
                    })
                    cache_info = self.get_cache_info(symbol, "prices")
                    metadata["from_cache"] = True
                    metadata["api_error"] = str(e)
                    metadata["cache_time"] = cache_info.get("last_updated") if cache_info else None
                    return fallback_df, metadata
            raise

    def _cache_prices(self, symbol: str, df: pd.DataFrame, period: str) -> None:
        from db.models import StockPrice, CacheMetadata
        if df.empty:
            return

        cache_df = df.reset_index()
        column_mapping = {}
        for col in cache_df.columns:
            col_lower = str(col).lower()
            if col_lower in ("date", "index"):
                column_mapping[col] = "date"
            elif col_lower == "open":
                column_mapping[col] = "open"
            elif col_lower == "high":
                column_mapping[col] = "high"
            elif col_lower == "low":
                column_mapping[col] = "low"
            elif col_lower == "close":
                column_mapping[col] = "close"
            elif col_lower == "volume":
                column_mapping[col] = "volume"
            elif col_lower == "dividends":
                column_mapping[col] = "dividends"
            elif col_lower in ("stock splits", "stock_splits"):
                column_mapping[col] = "stock_splits"

        cache_df = cache_df.rename(columns=column_mapping)
        required_cols = ["date", "open", "high", "low", "close", "volume", "dividends", "stock_splits"]
        for col in required_cols:
            if col not in cache_df.columns:
                cache_df[col] = 0.0 if col != "date" else None
        cache_df = cache_df[required_cols]

        # A bar with no real close is a bar that never happened. Pre-market
        # fetches can return today's (or even yesterday's) row with NaN OHLC;
        # the delete-then-insert below would REPLACE a good stored close with
        # that NaN, which then poisons previous_close on every prediction
        # stored that morning (2026-08-18: all 139 rows, evaluation crashed).
        finite_close = pd.to_numeric(cache_df["close"], errors="coerce")
        bad = ~(finite_close.notna() & (finite_close > 0)
                & (finite_close != float("inf")))
        if bad.any():
            logger.warning(
                f"Dropping {int(bad.sum())} bar(s) with unusable close for "
                f"{symbol}: {[str(d)[:10] for d in cache_df.loc[bad, 'date'].tolist()]}")
            cache_df = cache_df[~bad]
            if cache_df.empty:
                return

        now = datetime.now(timezone.utc)

        with get_session() as session:
            # Replace only the dates this fetch actually covers. Deleting the
            # whole symbol first meant any short fetch, the evaluator's 3mo
            # backfill, or a truncated yfinance response, destroyed the full
            # stored history and silently replaced it with the shorter frame.
            new_dates = [d for d in cache_df["date"].tolist() if d is not None]
            session.execute(delete(StockPrice).where(
                StockPrice.symbol == symbol,
                StockPrice.date.in_(new_dates),
            ))

            for _, row in cache_df.iterrows():
                session.add(StockPrice(
                    symbol=symbol,
                    date=row["date"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row["volume"]) if pd.notna(row["volume"]) else 0,
                    dividends=row["dividends"],
                    stock_splits=row["stock_splits"],
                    fetched_at=now,
                ))

            existing_meta = session.execute(
                select(CacheMetadata).where(
                    CacheMetadata.symbol == symbol,
                    CacheMetadata.data_type == "prices",
                )
            ).scalar_one_or_none()
            # Count what the table actually holds. The merge above means the
            # stored history can be longer than this fetch.
            total_rows = session.execute(
                select(func.count()).select_from(StockPrice)
                .where(StockPrice.symbol == symbol)
            ).scalar()
            if existing_meta:
                existing_meta.period = period
                existing_meta.last_updated = now
                existing_meta.record_count = total_rows
            else:
                session.add(CacheMetadata(
                    symbol=symbol, data_type="prices", period=period,
                    last_updated=now, record_count=total_rows,
                ))

    # =========================================================================
    # Stock info
    # =========================================================================

    def get_stock_info(self, symbol: str, force_refresh: bool = False) -> Optional[dict]:
        from db.models import StockInfo
        symbol = symbol.upper().strip()

        if not force_refresh and self.is_cached(symbol, "info"):
            with get_session() as session:
                row = session.execute(
                    select(StockInfo).where(StockInfo.symbol == symbol)
                ).scalar_one_or_none()
                if row:
                    return {
                        "symbol": symbol,
                        "name": row.name,
                        "sector": row.sector,
                        "industry": row.industry,
                        "market_cap": row.market_cap,
                        "current_price": row.current_price,
                        "previous_close": row.previous_close,
                        "pe_ratio": row.pe_ratio,
                        "dividend_yield": row.dividend_yield,
                        "fifty_two_week_high": row.fifty_two_week_high,
                        "fifty_two_week_low": row.fifty_two_week_low,
                        "volume": row.volume,
                        "avg_volume": row.avg_volume,
                    }

        from services.stock_data import get_stock_info
        info = get_stock_info(symbol)
        self._cache_info(info)
        return {
            "symbol": info.symbol,
            "name": info.name, "sector": info.sector, "industry": info.industry,
            "market_cap": info.market_cap, "current_price": info.current_price,
            "previous_close": info.previous_close, "pe_ratio": info.pe_ratio,
            "dividend_yield": info.dividend_yield,
            "fifty_two_week_high": info.fifty_two_week_high,
            "fifty_two_week_low": info.fifty_two_week_low,
            "volume": info.volume, "avg_volume": info.avg_volume,
        }

    def _cache_info(self, info) -> None:
        from db.models import StockInfo, CacheMetadata
        with get_session() as session:
            session.merge(StockInfo(
                symbol=info.symbol, name=info.name, sector=info.sector,
                industry=info.industry, market_cap=info.market_cap,
                current_price=info.current_price, previous_close=info.previous_close,
                pe_ratio=info.pe_ratio, dividend_yield=info.dividend_yield,
                fifty_two_week_high=info.fifty_two_week_high,
                fifty_two_week_low=info.fifty_two_week_low,
                volume=info.volume, avg_volume=info.avg_volume,
                fetched_at=datetime.now(timezone.utc),
            ))
            existing_meta = session.execute(
                select(CacheMetadata).where(
                    CacheMetadata.symbol == info.symbol,
                    CacheMetadata.data_type == "info",
                )
            ).scalar_one_or_none()
            if existing_meta:
                existing_meta.last_updated = datetime.now(timezone.utc)
                existing_meta.record_count = 1
            else:
                session.add(CacheMetadata(
                    symbol=info.symbol, data_type="info", period=None,
                    last_updated=datetime.now(timezone.utc), record_count=1,
                ))

    # =========================================================================
    # Multiple stocks
    # =========================================================================

    def get_multiple_stocks(
        self, symbols: list[str], period: str = APP.DEFAULT_PERIOD,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        result = {}
        for symbol in symbols:
            try:
                df = self.get_stock_prices(symbol, period, force_refresh)
                result[symbol.upper()] = df
            except ValueError:
                continue
        return result

    # =========================================================================
    # Parquet export (uses pandas, not DuckDB)
    # =========================================================================

    def export_to_parquet(self, symbol: str, output_path: Optional[str] = None) -> str:
        from db.models import StockPrice
        symbol = symbol.upper()
        if not self.is_cached(symbol, "prices"):
            raise ValueError(f"No cached data for {symbol}")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = f"cache/exports/{symbol}_{timestamp}.parquet"

        df = pd.read_sql(
            select(StockPrice).where(StockPrice.symbol == symbol),
            get_engine(),
        )
        df.to_parquet(output_path, index=False)
        return output_path

    # =========================================================================
    # Cache status / management
    # =========================================================================

    def get_cache_status(self) -> list[dict]:
        from db.models import CacheMetadata
        with get_session() as session:
            rows = session.execute(
                select(CacheMetadata).order_by(CacheMetadata.symbol, CacheMetadata.data_type)
            ).scalars().all()
            return [
                {
                    "symbol": r.symbol,
                    "data_type": r.data_type,
                    "period": r.period,
                    "last_updated": r.last_updated,
                    "record_count": r.record_count,
                }
                for r in rows
            ]

    def get_all_cached_symbols(self) -> list[str]:
        from db.models import CacheMetadata
        with get_session() as session:
            rows = session.execute(
                select(CacheMetadata.symbol)
                .where(CacheMetadata.data_type == "prices")
                .distinct()
                .order_by(CacheMetadata.symbol)
            ).scalars().all()
            return list(rows)

    def clear_symbol(self, symbol: str) -> None:
        from db.models import StockPrice, StockInfo, NewsArticle, CacheMetadata
        symbol = symbol.upper()
        with get_session() as session:
            for model in [StockPrice, NewsArticle]:
                session.execute(delete(model).where(model.symbol == symbol))
            session.execute(delete(StockInfo).where(StockInfo.symbol == symbol))
            session.execute(delete(CacheMetadata).where(CacheMetadata.symbol == symbol))

    def clear_all(self) -> None:
        from db.models import StockPrice, StockInfo, NewsArticle, CacheMetadata
        with get_session() as session:
            for model in [StockPrice, StockInfo, NewsArticle, CacheMetadata]:
                session.execute(delete(model))

    def get_raw_data(self, symbol: str) -> pd.DataFrame:
        from db.models import StockPrice
        return pd.read_sql(
            select(
                StockPrice.date, StockPrice.open, StockPrice.high,
                StockPrice.low, StockPrice.close, StockPrice.volume,
                StockPrice.fetched_at,
            ).where(StockPrice.symbol == symbol.upper())
            .order_by(StockPrice.date.desc()),
            get_engine(),
        )

    # =========================================================================
    # Model Predictions
    # =========================================================================

    def store_prediction(
        self, symbol: str, model_name: str, result: dict,
        prediction_date_str: str | None = None,
        run_id: str | None = None,
    ) -> None:
        from db.models import ModelPrediction, StockPrice
        from utils.trading_calendar import get_next_trading_day

        symbol = symbol.upper()
        pred_date = date.fromisoformat(prediction_date_str) if prediction_date_str else date.today()
        target = get_next_trading_day(pred_date)
        if run_id is None:
            run_id = _fallback_run_id(
                f"prediction {symbol}/{model_name} at {pred_date}")

        details = result.get("details", {})
        feature_values = _sanitize_json(details.get("feature_values"))
        training_samples = details.get("training_samples")

        data_hash = hashlib.sha256(
            json.dumps({
                "symbol": symbol, "model": model_name,
                "date": str(pred_date),
                "features": feature_values,
            }, sort_keys=True, default=str).encode()
        ).hexdigest()

        with get_session() as session:
            pred_id = _prediction_id(symbol, model_name, pred_date, run_id,
                                     _run_kind(session, run_id))

            # Most recent USABLE close, not merely the most recent row: a
            # partial pre-market bar can sit at the top with a NaN close, and
            # storing that as previous_close makes the row unscorable (and
            # once crashed the evaluator wholesale). Scan a few and take the
            # first real one; NULL when none is, the evaluator skips NULLs.
            recent_closes = session.execute(
                select(StockPrice.close)
                .where(StockPrice.symbol == symbol, StockPrice.date <= str(pred_date))
                .order_by(StockPrice.date.desc())
                .limit(5)
            ).scalars().all()
            prev_close_row = next(
                (c for c in recent_closes if _usable_price(c)), None)

            # Re-storing an existing id keeps its owner and visibility. The
            # merge used to take them from the CURRENT writer, so a private
            # job re-running a symbol hid a public UI prediction (and vice
            # versa): the row changed hands as a side effect of a rerun.
            existing = session.get(ModelPrediction, pred_id)
            owner_uid = existing.owner_uid if existing else _current_uid()
            is_public = (bool(existing.is_public) if existing
                         else _default_public())

            session.merge(ModelPrediction(
                id=pred_id,
                symbol=symbol,
                model_name=model_name,
                prediction_date=pred_date,
                target_date=target,
                decision=result.get("decision", "HOLD"),
                confidence=result.get("confidence"),
                up_probability=result.get("up_probability"),
                predicted_close=result.get("predicted_close"),
                previous_close=prev_close_row,
                model_version=result.get("model_version") or details.get("model_version"),
                training_samples=training_samples,
                feature_values_json=json.dumps(feature_values) if feature_values else None,
                details_json=_sanitize_json(details) if details else None,
                # "ok" | "empty" | "unavailable". NULL for rows written before
                # this was tracked, which is not the same as "ok".
                news_status=(result.get("news_status")
                             or details.get("news_status")),
                ensemble_method=details.get("method"),
                input_data_hash=data_hash,
                # The run that produced this row, and how long the model took
                #: what lets the Trace page join a run to its predictions.
                run_id=run_id,
                duration_ms=result.get("duration_ms"),
                created_at=datetime.now(timezone.utc),
                # Re-storing a prediction id invalidates any prior evaluation:
                # merge() used to keep old actual_close/was_correct/pnl from a
                # DIFFERENT decision (12% of rows had scores contradicting
                # their own decision), and the evaluator never revisits rows
                # with actual_close set. Explicit Nones force a re-score.
                actual_close=None,
                was_correct=None,
                pnl_dollars=None,
                evaluated_at=None,
                owner_uid=owner_uid,
                is_public=is_public,
            ))

    def has_predictions_for_today(self, symbol: str) -> bool:
        from db.models import ModelPrediction
        today = date.today()
        with get_session() as session:
            count = session.execute(
                select(func.count()).select_from(ModelPrediction).where(
                    ModelPrediction.symbol == symbol.upper(),
                    ModelPrediction.prediction_date == today,
                )
            ).scalar()
            return count > 0

    def get_predictions_for_today(self, symbol: str,
                                  prediction_date: date | None = None) -> dict:
        """Stored predictions for a symbol, keyed by model name.

        ``prediction_date`` defaults to today; pass the run's data cutoff to
        ask "has this exact analysis already been done?": which is what lets
        a repeated run reuse work instead of re-paying for it.
        """
        from db.models import ModelPrediction
        target_day = prediction_date or date.today()
        with get_session() as session:
            rows = session.execute(
                select(ModelPrediction).where(
                    ModelPrediction.symbol == symbol.upper(),
                    ModelPrediction.prediction_date == target_day,
                )
                # A scheduled and a manual run each keep their own row for
                # one (symbol, model, cutoff), so the reuse check has to
                # choose: oldest first means the newest analysis wins the
                # slot below. Legacy rows carry no stamp and lose.
                .order_by(ModelPrediction.created_at.asc().nullsfirst())
            ).scalars().all()

            results = {}
            for r in rows:
                results[r.model_name] = {
                    "model_name": r.model_name,
                    "decision": r.decision,
                    "confidence": r.confidence,
                    "up_probability": r.up_probability,
                    "predicted_close": r.predicted_close,
                    "details": r.details_json or {},
                    "model_version": r.model_version,
                    # The bar this call was made against. A reuse check needs
                    # it to tell "already done" from "done on stale data".
                    "previous_close": r.previous_close,
                    "error": None,
                }
            return results

    def evaluate_predictions(self) -> int:
        import pytz

        from db.models import ModelPrediction, StockPrice
        from models.base import compute_pnl
        from utils.trading_calendar import get_previous_trading_day, is_market_open_today

        # Exchange-local date, not the host's: on the UTC container the host
        # date rolls over at 20:00 ET, which advanced the cutoff a day early.
        today = datetime.now(pytz.timezone("US/Eastern")).date()
        cutoff = get_previous_trading_day(today) if is_market_open_today() else today

        # Backfill: predictions can't score without the target date's close.
        # The price cache may be stale (TTL served old rows), so force-refresh
        # any symbol whose stored prices don't reach its pending target dates.
        with get_session() as session:
            pending_pairs = session.execute(
                select(ModelPrediction.symbol, func.max(ModelPrediction.target_date))
                .where(
                    ModelPrediction.actual_close.is_(None),
                    ModelPrediction.target_date <= cutoff,
                )
                .group_by(ModelPrediction.symbol)
            ).all()

        for symbol, max_target in pending_pairs:
            with get_session() as session:
                latest = session.execute(
                    select(func.max(StockPrice.date)).where(StockPrice.symbol == symbol)
                ).scalar()
            # <= rather than <: when the stored bar for the target date was
            # written DURING that session it is a partial intraday print, and
            # scoring against it is scoring against a price that never closed.
            # One refresh per pending symbol buys a settled bar.
            if latest is None or str(latest) <= str(max_target):
                try:
                    self.get_stock_prices(symbol, period="3mo", force_refresh=True)
                    logger.info(f"Backfilled prices for {symbol} through {max_target}")
                except Exception as e:
                    logger.warning(f"Price backfill failed for {symbol}: {e}")

        with get_session() as session:
            pending = session.execute(
                select(ModelPrediction).where(
                    ModelPrediction.actual_close.is_(None),
                    ModelPrediction.target_date <= cutoff,
                )
            ).scalars().all()

            evaluated = 0
            skipped_no_price: list[str] = []
            skipped_no_prev = 0
            hold_bands: dict = {}
            for pred in pending:
                actual_row = session.execute(
                    select(StockPrice.close).where(
                        StockPrice.symbol == pred.symbol,
                        StockPrice.date == str(pred.target_date),
                    )
                ).scalar_one_or_none()

                if not _usable_price(actual_row):
                    skipped_no_price.append(
                        f"{pred.symbol}@{str(pred.target_date)[:10]}")
                    continue
                if not _usable_price(pred.previous_close):
                    skipped_no_prev += 1
                    continue

                actual_close = actual_row
                price_went_up = actual_close > pred.previous_close

                if pred.decision == "BUY":
                    was_correct = price_went_up
                elif pred.decision == "SELL":
                    was_correct = not price_went_up
                else:
                    # A HOLD is right when standing aside was right: the move
                    # stayed inside the symbol's own no-trade band. Leaving
                    # this as None made HOLD unfalsifiable, so a model that
                    # holds most of the time never showed a wrong call.
                    move = abs(actual_close - pred.previous_close) / pred.previous_close
                    band = self._hold_band(
                        session, pred.symbol, hold_bands, pred.target_date)
                    was_correct = move <= band
                    # The band drifts as history accrues; without recording
                    # what this row was judged against, the verdict can never
                    # be audited or reproduced.
                    pred.details_json = _sanitize_json({
                        **(pred.details_json or {}),
                        "hold_eval": {"band": band, "move": move},
                    })

                pnl = compute_pnl(pred.decision, pred.previous_close, actual_close)

                pred.actual_close = actual_close
                pred.was_correct = was_correct
                pred.pnl_dollars = pnl
                pred.evaluated_at = datetime.now(timezone.utc)
                evaluated += 1

            if evaluated > 0:
                logger.info(f"Evaluated {evaluated} predictions")
            # A skip is not a benign outcome: the row stays unevaluated and
            # nothing else will ever come back for it, so say so loudly.
            if skipped_no_price:
                logger.warning(
                    f"Evaluation skipped {len(skipped_no_price)} prediction(s) "
                    f"with no close for their target session: "
                    f"{', '.join(sorted(set(skipped_no_price))[:10])}")
            if skipped_no_prev:
                logger.warning(
                    f"Evaluation skipped {skipped_no_prev} prediction(s) with "
                    f"no usable previous_close, these can never score")

        return evaluated

    def evaluation_backlog(self) -> dict:
        """Mature predictions still unevaluated, the number that should be
        zero after a healthy evaluation run, and the alarm when it isn't."""
        import pytz

        from db.models import ModelPrediction
        from utils.trading_calendar import get_previous_trading_day, is_market_open_today

        today = datetime.now(pytz.timezone("US/Eastern")).date()
        cutoff = get_previous_trading_day(today) if is_market_open_today() else today
        with get_session() as session:
            rows = session.execute(
                select(ModelPrediction.target_date, func.count())
                .where(
                    ModelPrediction.actual_close.is_(None),
                    ModelPrediction.target_date <= cutoff,
                    # A row with no usable previous_close can never score
                    # (the evaluator skips it every night). Counting it made
                    # the evaluation job "partial" forever, which unstamped
                    # its success date, flagged it overdue and mailed daily.
                    ModelPrediction.previous_close.isnot(None),
                    ModelPrediction.previous_close > 0,
                    ModelPrediction.previous_close != float("nan"),
                )
                .group_by(ModelPrediction.target_date)
            ).all()
            unscorable = session.execute(
                select(func.count()).select_from(ModelPrediction).where(
                    ModelPrediction.actual_close.is_(None),
                    ModelPrediction.target_date <= cutoff,
                    (ModelPrediction.previous_close.is_(None))
                    | (ModelPrediction.previous_close <= 0)
                    | (ModelPrediction.previous_close == float("nan")),
                )
            ).scalar() or 0
        return {
            "pending_mature": sum(n for _, n in rows),
            "by_target_date": {str(d)[:10]: n for d, n in sorted(rows)},
            # Reported, not counted: nothing will ever score these.
            "unscorable": int(unscorable),
        }

    def _hold_band(self, session, symbol: str, _cache: dict,
                   before_date=None) -> float:
        """No-trade band for a symbol: its own typical absolute daily move.

        Judging every name against one fixed percentage compares a utility
        against a biotech. This uses the symbol's median absolute daily return
        so "the session was quiet for this stock" means the same thing
        everywhere. Falls back to the fixed band when history is too short.

        ``before_date`` bounds the history to bars strictly before the
        prediction's target session, the band a HOLD is judged against must
        not be derived from prices that hadn't printed yet.
        """
        from db.models import StockPrice
        from config import MODEL

        key = (symbol, str(before_date)[:10] if before_date else None)
        if key in _cache:
            return _cache[key]

        band = MODEL.HOLD_BAND_PCT
        try:
            stmt = (
                select(StockPrice.close)
                .where(StockPrice.symbol == symbol)
                .order_by(StockPrice.date.asc())
            )
            if before_date is not None:
                stmt = stmt.where(StockPrice.date < str(before_date)[:10])
            closes = session.execute(stmt).scalars().all()
            if len(closes) > MODEL.HOLD_BAND_MIN_HISTORY:
                moves = [
                    abs(closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(1, len(closes))
                    if _usable_price(closes[i - 1]) and _usable_price(closes[i])
                ]
                if moves:
                    moves.sort()
                    mid = len(moves) // 2
                    median = (moves[mid] if len(moves) % 2
                              else (moves[mid - 1] + moves[mid]) / 2)
                    if median > 0:
                        band = median * MODEL.HOLD_BAND_VOL_MULTIPLE
        except Exception as e:
            logger.debug("hold band fallback for %s: %s", symbol, e)

        _cache[key] = band
        return band

    def backfill_hold_scores(self) -> int:
        """Score HOLD rows that were resolved before HOLDs were scorable.

        `evaluate_predictions` only considers rows with `actual_close IS NULL`,
        so rows already resolved under the old rule. Where a HOLD got
        `was_correct = None`, are permanently invisible to it. Any future
        change to how a decision is scored needs a pass like this one, or the
        history keeps whatever verdict the rule at the time produced.

        Idempotent: only touches HOLDs that have a resolved price and no
        verdict yet. Never rewrites a BUY/SELL.
        """
        from db.models import ModelPrediction

        updated = 0
        bands: dict = {}
        with get_session() as session:
            rows = session.execute(
                select(ModelPrediction).where(
                    ModelPrediction.decision == "HOLD",
                    ModelPrediction.was_correct.is_(None),
                    ModelPrediction.actual_close.isnot(None),
                )
            ).scalars().all()

            for pred in rows:
                if not _usable_price(pred.previous_close) \
                        or not _usable_price(pred.actual_close):
                    continue
                move = abs(pred.actual_close - pred.previous_close) / pred.previous_close
                band = self._hold_band(session, pred.symbol, bands, pred.target_date)
                pred.was_correct = move <= band
                pred.details_json = _sanitize_json({
                    **(pred.details_json or {}),
                    "hold_eval": {"band": band, "move": move},
                })
                if pred.pnl_dollars is None:
                    pred.pnl_dollars = 0.0
                pred.evaluated_at = datetime.now(timezone.utc)
                updated += 1

        if updated:
            logger.info(f"Backfilled {updated} HOLD scores across "
                        f"{len(bands)} symbol bands")
        return updated

    def get_prediction_history(
        self, symbol: str, model_name: Optional[str] = None, limit: int = 50,
    ) -> list[dict]:
        from db.models import ModelPrediction
        with get_session() as session:
            stmt = select(ModelPrediction).where(ModelPrediction.symbol == symbol.upper())
            if model_name:
                stmt = stmt.where(ModelPrediction.model_name == model_name)
            stmt = stmt.order_by(ModelPrediction.prediction_date.desc()).limit(limit)

            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id, "symbol": r.symbol, "model_name": r.model_name,
                    "prediction_date": r.prediction_date, "target_date": r.target_date,
                    "decision": r.decision, "confidence": r.confidence,
                    "up_probability": r.up_probability, "predicted_close": r.predicted_close,
                    "previous_close": r.previous_close, "actual_close": r.actual_close,
                    "was_correct": r.was_correct, "pnl_dollars": r.pnl_dollars,
                    "model_version": r.model_version,
                    "evaluated_at": r.evaluated_at, "created_at": r.created_at,
                }
                for r in rows
            ]

    # =========================================================================
    # TradingAgents report persistence
    # =========================================================================

    def save_trading_agent_report(
        self, symbol: str, trade_date: str, decision: str, confidence: float,
        report_text: str, model_name: str = "", input_tokens: int = 0,
        output_tokens: int = 0, run_id: str | None = None,
    ) -> None:
        import uuid
        from db.models import TradingAgentReport

        # Callers pass the ISO string the run works in; the column is a
        # Date. psycopg2 coerces the string, SQLite (the tests) does not.
        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date[:10])
        with get_session() as session:
            session.add(TradingAgentReport(
                id=str(uuid.uuid4()),
                symbol=symbol.upper(),
                trade_date=trade_date,
                decision=decision,
                confidence=confidence,
                report_text=report_text,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                run_id=(run_id if run_id is not None
                        else _fallback_run_id(f"research report {symbol}")),
                owner_uid=_current_uid(),
                is_public=_default_public(),
            ))

    def get_trading_agent_reports(self, symbol: str, limit: int = 20) -> list[dict]:
        from db.models import TradingAgentReport
        with get_session() as session:
            rows = session.execute(
                select(TradingAgentReport)
                .where(TradingAgentReport.symbol == symbol.upper())
                .where(_visible(TradingAgentReport))
                .order_by(TradingAgentReport.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "id": r.id, "symbol": r.symbol,
                    "trade_date": str(r.trade_date),
                    "decision": r.decision, "confidence": r.confidence,
                    "report_text": r.report_text, "model_name": r.model_name,
                    "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                    "created_at": str(r.created_at),
                    "owner_uid": r.owner_uid,
                }
                for r in rows
            ]

    def get_prior_trading_agent_report(
        self, symbol: str, as_of: str,
        generated_before: "datetime | None" = None,
        exclude_id: str | None = None,
    ) -> dict | None:
        """The stance a reader last saw for ``symbol``, or None.

        Two separate bounds, because trade_date alone cannot do both jobs:

        * ``trade_date <= as_of`` is the lookahead guard. A report for a LATER
          session is never visible, so a walk-forward run can only ever see
          stances for sessions that had already happened.
        * ``created_at < generated_before`` is the self-exclusion guard, and it
          is what makes a same-cutoff re-run work. Every report this app writes
          carries trade_date = the run's DATA CUTOFF, and the run dialog caps
          the target at the next session, so re-running a name today produces
          a report with the identical trade_date as this morning's. A strict
          ``<`` on trade_date silently hid the predecessor in exactly the case
          a reader is checking continuity. Ordering by (trade_date, created_at)
          descending then picks the genuinely most recent earlier stance.

        ``exclude_id`` drops one specific row (a regeneration replacing a
        known report) without relying on clock resolution.
        """
        from db.models import TradingAgentReport
        with get_session() as session:
            q = (
                select(TradingAgentReport)
                .where(TradingAgentReport.symbol == symbol.upper())
                .where(TradingAgentReport.trade_date <= as_of)
                .where(_visible(TradingAgentReport))
            )
            if generated_before is not None:
                q = q.where(TradingAgentReport.created_at < generated_before)
            if exclude_id:
                q = q.where(TradingAgentReport.id != exclude_id)
            r = session.execute(
                q.order_by(TradingAgentReport.trade_date.desc(),
                           TradingAgentReport.created_at.desc())
                .limit(1)
            ).scalars().first()
            if r is None:
                return None
            return {
                "id": r.id, "symbol": r.symbol,
                "trade_date": str(r.trade_date),
                "decision": r.decision, "confidence": r.confidence,
                "report_text": r.report_text, "model_name": r.model_name,
                "created_at": str(r.created_at),
            }

    def latest_reports_by_symbol(self, symbols: list[str] | None = None) -> dict[str, dict]:
        """Newest research report per symbol, without the report body.

        One DISTINCT ON query instead of a query per symbol, the Home page
        needs "does this name have a report, and what did it say" for every
        row, and report_text would drag megabytes through the session for a
        headline that only needs the verdict fields.
        """
        from db.models import TradingAgentReport
        with get_session() as session:
            stmt = (
                select(TradingAgentReport.id, TradingAgentReport.symbol,
                       TradingAgentReport.trade_date, TradingAgentReport.decision,
                       TradingAgentReport.confidence, TradingAgentReport.model_name,
                       TradingAgentReport.created_at, TradingAgentReport.owner_uid)
                .where(_visible(TradingAgentReport))
                .distinct(TradingAgentReport.symbol)
                .order_by(TradingAgentReport.symbol,
                          TradingAgentReport.created_at.desc())
            )
            if symbols:
                stmt = stmt.where(TradingAgentReport.symbol.in_(
                    [s.upper() for s in symbols]))
            return {
                r.symbol: {
                    "id": r.id, "symbol": r.symbol,
                    "trade_date": str(r.trade_date),
                    "decision": r.decision, "confidence": r.confidence,
                    "model_name": r.model_name,
                    "created_at": str(r.created_at),
                    "owner_uid": r.owner_uid,
                }
                for r in session.execute(stmt)
            }

    def get_all_trading_agent_reports(self, limit: int | None = 50) -> list[dict]:
        from db.models import TradingAgentReport
        with get_session() as session:
            q = (select(TradingAgentReport)
                .where(_visible(TradingAgentReport))
                .order_by(TradingAgentReport.created_at.desc()))
            if limit:
                q = q.limit(limit)
            rows = session.execute(q).scalars().all()
            return [
                {
                    "id": r.id, "symbol": r.symbol,
                    "trade_date": str(r.trade_date),
                    "decision": r.decision, "confidence": r.confidence,
                    "report_text": r.report_text, "model_name": r.model_name,
                    "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                    "created_at": str(r.created_at),
                }
                for r in rows
            ]

    def get_trading_agent_report(self, report_id: str) -> dict | None:
        """One research report, body included, or None when it is missing
        or not visible to the viewer."""
        from db.models import TradingAgentReport
        if not report_id:
            return None
        with get_session() as session:
            r = session.execute(
                select(TradingAgentReport)
                .where(TradingAgentReport.id == str(report_id))
                .where(_visible(TradingAgentReport))
            ).scalars().first()
            if r is None:
                return None
            return {
                "id": r.id, "symbol": r.symbol,
                "trade_date": str(r.trade_date),
                "decision": r.decision, "confidence": r.confidence,
                "report_text": r.report_text, "model_name": r.model_name,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "run_id": r.run_id,
                "created_at": str(r.created_at),
                "owner_uid": r.owner_uid,
            }

    def get_trading_agent_reports_for_run(self, run_id: str) -> list[dict]:
        """Headline fields of every visible report a run wrote, newest
        first. No body: the run page lists verdicts and opens the reader
        for the text (ix_trading_agent_report_run)."""
        from db.models import TradingAgentReport
        if not run_id:
            return []
        with get_session() as session:
            rows = session.execute(
                select(TradingAgentReport.id, TradingAgentReport.symbol,
                       TradingAgentReport.trade_date, TradingAgentReport.decision,
                       TradingAgentReport.confidence, TradingAgentReport.model_name,
                       TradingAgentReport.created_at, TradingAgentReport.owner_uid)
                .where(TradingAgentReport.run_id == run_id)
                .where(_visible(TradingAgentReport))
                .order_by(TradingAgentReport.created_at.desc())
            ).all()
            return [
                {
                    "id": r.id, "symbol": r.symbol,
                    "trade_date": str(r.trade_date),
                    "decision": r.decision, "confidence": r.confidence,
                    "model_name": r.model_name,
                    "created_at": str(r.created_at),
                    "owner_uid": r.owner_uid,
                }
                for r in rows
            ]

    def get_model_accuracy(self, model_name: str, symbol: Optional[str] = None) -> dict:
        """Accuracy for a model.

        `total`/`correct`/`accuracy` cover ACTIVE (BUY/SELL) calls only, so the
        headline number keeps meaning "directional hit rate" now that HOLDs are
        also scored. HOLD accountability is reported alongside as hold_*, and
        `all_*` covers both, mixing them into one rate would compare a
        directional call against a no-trade-band call.
        """
        from db.models import ModelPrediction
        with get_session() as session:
            base_filter = [
                ModelPrediction.model_name == model_name,
                ModelPrediction.was_correct.isnot(None),
            ]
            if symbol:
                base_filter.append(ModelPrediction.symbol == symbol.upper())

            active = [*base_filter, ModelPrediction.decision != "HOLD"]
            held = [*base_filter, ModelPrediction.decision == "HOLD"]

            def _count(*where) -> int:
                return session.execute(
                    select(func.count()).select_from(ModelPrediction).where(*where)
                ).scalar() or 0

            total = _count(*active)
            correct = _count(*active, ModelPrediction.was_correct.is_(True))
            hold_total = _count(*held)
            hold_correct = _count(*held, ModelPrediction.was_correct.is_(True))

            pnl_total = session.execute(
                select(func.sum(ModelPrediction.pnl_dollars)).where(*base_filter)
            ).scalar() or 0.0

            distinct_symbols = session.execute(
                select(func.count(ModelPrediction.symbol.distinct())).where(*base_filter)
            ).scalar() or 0

            all_total = total + hold_total
            all_correct = correct + hold_correct
            return {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0.0,
                "hold_total": hold_total,
                "hold_correct": hold_correct,
                "hold_accuracy": hold_correct / hold_total if hold_total > 0 else 0.0,
                "all_total": all_total,
                "all_correct": all_correct,
                "all_accuracy": all_correct / all_total if all_total > 0 else 0.0,
                "pnl_total": float(pnl_total),
                "distinct_symbols": distinct_symbols,
            }

    # =========================================================================
    # Strategy Evaluations
    # =========================================================================

    def store_strategy_evaluations(self, evaluations: list[dict]) -> int:
        from db.models import StrategyEvaluation
        inserted = 0
        with get_session() as session:
            for ev in evaluations:
                try:
                    existing = session.execute(
                        select(StrategyEvaluation).where(StrategyEvaluation.id == ev["id"])
                    ).scalar_one_or_none()
                    if existing:
                        continue
                    session.add(StrategyEvaluation(
                        id=ev["id"],
                        prediction_id=ev["prediction_id"],
                        strategy_name=ev["strategy_name"],
                        strategy_version=ev.get("strategy_version"),
                        action=ev["action"],
                        position_size=ev.get("position_size", 1000.0),
                        entry_price=ev.get("entry_price"),
                        exit_price=ev.get("exit_price"),
                        pnl_dollars=ev.get("pnl_dollars"),
                        was_correct=ev.get("was_correct"),
                        signal_metadata_json=json.dumps(ev.get("metadata", {})) if ev.get("metadata") else None,
                        evaluated_at=datetime.now(timezone.utc),
                    ))
                    inserted += 1
                except Exception:
                    pass
        return inserted

    def get_unevaluated_predictions_for_strategy(self, strategy_name: str) -> list[dict]:
        """Scored predictions this strategy has not been run over yet.

        Scheduled rows only, like every other track record on this platform:
        an ad-hoc rerun of a call the daily job already made stores its own
        row, and each extra row would become another strategy_evaluations
        trade, inflating the per-symbol win rate and Sharpe on Analyze.
        """
        from db.models import ModelPrediction, StrategyEvaluation
        with get_session() as session:
            subq = select(StrategyEvaluation.prediction_id).where(
                StrategyEvaluation.strategy_name == strategy_name
            ).scalar_subquery()

            rows = session.execute(_by_run_kind(
                select(ModelPrediction)
                .where(
                    ModelPrediction.actual_close.isnot(None),
                    ModelPrediction.id.notin_(subq),
                ), "scheduled")
            ).scalars().all()

            return [
                {
                    "id": r.id, "symbol": r.symbol, "model_name": r.model_name,
                    "prediction_date": r.prediction_date, "target_date": r.target_date,
                    "decision": r.decision, "confidence": r.confidence,
                    "up_probability": r.up_probability, "predicted_close": r.predicted_close,
                    "previous_close": r.previous_close, "actual_close": r.actual_close,
                    "was_correct": r.was_correct, "pnl_dollars": r.pnl_dollars,
                    "feature_values_json": r.feature_values_json,
                    "details_json": json.dumps(r.details_json) if r.details_json else None,
                    "model_version": r.model_version,
                }
                for r in rows
            ]

    def get_strategy_evaluations(
        self, strategy_name: str, symbol: Optional[str] = None, limit: int = 200,
    ) -> list[dict]:
        from db.models import ModelPrediction, StrategyEvaluation
        with get_session() as session:
            stmt = (
                select(StrategyEvaluation, ModelPrediction)
                .join(ModelPrediction, ModelPrediction.id == StrategyEvaluation.prediction_id)
                .where(StrategyEvaluation.strategy_name == strategy_name)
            )
            if symbol:
                stmt = stmt.where(ModelPrediction.symbol == symbol.upper())
            stmt = stmt.order_by(ModelPrediction.target_date.desc()).limit(limit)

            rows = session.execute(stmt).all()
            return [
                {
                    "id": se.id, "prediction_id": se.prediction_id,
                    "strategy_name": se.strategy_name, "strategy_version": se.strategy_version,
                    "action": se.action, "position_size": se.position_size,
                    "entry_price": se.entry_price, "exit_price": se.exit_price,
                    "pnl_dollars": se.pnl_dollars, "was_correct": se.was_correct,
                    "signal_metadata_json": se.signal_metadata_json,
                    "evaluated_at": se.evaluated_at,
                    "symbol": mp.symbol, "model_name": mp.model_name,
                    "target_date": mp.target_date, "model_decision": mp.decision,
                }
                for se, mp in rows
            ]

    def get_evaluated_signal_series(self, strategy_name: str, symbol: str) -> list[dict]:
        from db.models import ModelPrediction, StrategyEvaluation
        with get_session() as session:
            rows = session.execute(
                select(
                    ModelPrediction.target_date,
                    StrategyEvaluation.action,
                    StrategyEvaluation.entry_price,
                    StrategyEvaluation.exit_price,
                    StrategyEvaluation.pnl_dollars,
                )
                .join(ModelPrediction, ModelPrediction.id == StrategyEvaluation.prediction_id)
                .where(
                    StrategyEvaluation.strategy_name == strategy_name,
                    ModelPrediction.symbol == symbol.upper(),
                    StrategyEvaluation.action != "SKIP",
                )
                .order_by(ModelPrediction.target_date.asc())
            ).all()
            return [
                {
                    "target_date": r[0], "action": r[1],
                    "entry_price": r[2], "exit_price": r[3],
                    "pnl_dollars": r[4],
                }
                for r in rows
            ]

    def get_strategy_symbols(self, strategy_name: str) -> list[str]:
        from db.models import ModelPrediction, StrategyEvaluation
        with get_session() as session:
            rows = session.execute(
                select(ModelPrediction.symbol.distinct())
                .select_from(StrategyEvaluation)
                .join(ModelPrediction, ModelPrediction.id == StrategyEvaluation.prediction_id)
                .where(
                    StrategyEvaluation.strategy_name == strategy_name,
                    StrategyEvaluation.action != "SKIP",
                )
            ).scalars().all()
            return list(rows)

    def store_strategy_metrics(
        self, strategy_name: str, symbol: Optional[str], period: str, metrics: dict,
    ) -> None:
        from db.models import StrategyMetrics
        metrics_id = f"{strategy_name}_{symbol or 'all'}_{period}"
        # Non-finite floats (inf Sharpe on a zero-variance series) serialize
        # to the invalid JSON token "Infinity" and fail the whole insert. 
        # sanitize at the persistence boundary regardless of caller hygiene.
        metrics = _sanitize_json(metrics)
        with get_session() as session:
            session.merge(StrategyMetrics(
                id=metrics_id,
                strategy_name=strategy_name,
                symbol=symbol,
                period=period,
                sharpe_ratio=metrics.get("sharpe_ratio"),
                sortino_ratio=metrics.get("sortino_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
                win_rate=metrics.get("win_rate"),
                total_pnl=metrics.get("total_pnl"),
                total_trades=metrics.get("total_trades"),
                metrics_json=metrics,
                computed_at=datetime.now(timezone.utc),
            ))

    def get_strategy_metrics(
        self, strategy_name: Optional[str] = None, symbol: Optional[str] = None,
    ) -> list[dict]:
        from db.models import StrategyMetrics
        with get_session() as session:
            stmt = select(StrategyMetrics)
            if strategy_name:
                stmt = stmt.where(StrategyMetrics.strategy_name == strategy_name)
            if symbol:
                stmt = stmt.where(StrategyMetrics.symbol == symbol.upper())
            stmt = stmt.order_by(StrategyMetrics.computed_at.desc())

            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id, "strategy_name": r.strategy_name,
                    "symbol": r.symbol, "period": r.period,
                    "sharpe_ratio": r.sharpe_ratio, "sortino_ratio": r.sortino_ratio,
                    "max_drawdown": r.max_drawdown, "win_rate": r.win_rate,
                    "total_pnl": r.total_pnl, "total_trades": r.total_trades,
                    "metrics_json": json.dumps(r.metrics_json) if r.metrics_json else None,
                    "computed_at": r.computed_at,
                }
                for r in rows
            ]

    def delete_strategy_evaluations(self, strategy_name: Optional[str] = None) -> int:
        from db.models import StrategyEvaluation
        with get_session() as session:
            stmt = select(func.count()).select_from(StrategyEvaluation)
            del_stmt = delete(StrategyEvaluation)
            if strategy_name:
                stmt = stmt.where(StrategyEvaluation.strategy_name == strategy_name)
                del_stmt = del_stmt.where(StrategyEvaluation.strategy_name == strategy_name)
            count = session.execute(stmt).scalar() or 0
            if count > 0:
                session.execute(del_stmt)
        return count

    # =========================================================================
    # Historical News (AV)
    # =========================================================================

    def store_historical_news(self, articles: list[dict]) -> int:
        from db.models import HistoricalNews
        inserted = 0
        with get_session() as session:
            for article in articles:
                article_id = article.get("id") or hashlib.md5(
                    (article.get("url", "") + article.get("published_date", "")).encode()
                ).hexdigest()

                existing = session.execute(
                    select(HistoricalNews).where(HistoricalNews.id == article_id)
                ).scalar_one_or_none()
                if existing:
                    continue

                try:
                    session.add(HistoricalNews(
                        id=article_id,
                        symbol=article.get("symbol", ""),
                        published_date=article.get("published_date"),
                        title=article.get("title"),
                        summary=article.get("summary"),
                        url=article.get("url"),
                        source=article.get("source"),
                        topics_json=json.dumps(article.get("topics", [])) if article.get("topics") else None,
                        overall_sentiment_score=article.get("overall_sentiment_score"),
                        overall_sentiment_label=article.get("overall_sentiment_label"),
                        ticker_sentiment_score=article.get("ticker_sentiment_score"),
                        ticker_relevance_score=article.get("ticker_relevance_score"),
                        fetched_at=datetime.now(timezone.utc),
                        published_at=article.get("published_at"),
                    ))
                    inserted += 1
                except Exception:
                    pass
        return inserted

    # ---- point-in-time news store: per-day vendor coverage -----------------

    def news_coverage(self, symbol: str, start: "date", end: "date") -> dict:
        """{day: fetched_at} for the (symbol, day) pairs already fetched."""
        from db.models import NewsCoverage
        with get_session() as session:
            rows = session.execute(
                select(NewsCoverage.day, NewsCoverage.fetched_at).where(
                    NewsCoverage.symbol == symbol.upper(),
                    NewsCoverage.day >= start, NewsCoverage.day <= end)
            ).all()
        return {d: t for d, t in rows}

    def mark_news_coverage(self, symbol: str, days: list) -> None:
        from db.models import NewsCoverage
        now = datetime.now(timezone.utc)
        with get_session() as session:
            for d in days:
                session.merge(NewsCoverage(symbol=symbol.upper(), day=d, fetched_at=now))

    def prune_news(self, retention_days: int) -> int:
        """Drop articles and coverage older than the retention window."""
        from db.models import HistoricalNews, NewsCoverage
        from sqlalchemy import delete
        cutoff = date.today() - timedelta(days=retention_days)
        with get_session() as session:
            n = session.execute(
                delete(HistoricalNews).where(HistoricalNews.published_date < cutoff)
            ).rowcount or 0
            session.execute(delete(NewsCoverage).where(NewsCoverage.day < cutoff))
        if n:
            logger.info(f"news store: pruned {n} articles older than {cutoff}")
        return int(n)

    def get_historical_news(
        self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
    ) -> list[dict]:
        from db.models import HistoricalNews
        with get_session() as session:
            stmt = select(HistoricalNews).where(HistoricalNews.symbol == symbol.upper())
            if start_date:
                stmt = stmt.where(HistoricalNews.published_date >= start_date)
            if end_date:
                stmt = stmt.where(HistoricalNews.published_date <= end_date)
            stmt = stmt.order_by(HistoricalNews.published_date.desc())

            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id, "symbol": r.symbol,
                    "published_date": r.published_date,
                    "title": r.title, "summary": r.summary,
                    "url": r.url, "source": r.source,
                    "topics_json": r.topics_json,
                    "topics": _parse_topics(r.topics_json),
                    "overall_sentiment_score": r.overall_sentiment_score,
                    "overall_sentiment_label": r.overall_sentiment_label,
                    "ticker_sentiment_score": r.ticker_sentiment_score,
                    "ticker_relevance_score": r.ticker_relevance_score,
                    "fetched_at": r.fetched_at,
                    "published_at": r.published_at,
                }
                for r in rows
            ]

    # =========================================================================
    # Historical Data Queries
    # =========================================================================

    def list_report_catalog(self, limit: int | None = 50) -> list[dict]:
        from db.models import ReportCatalog
        with get_session() as session:
            q = (select(ReportCatalog)
                .where(_visible(ReportCatalog))
                .order_by(ReportCatalog.created_at.desc()))
            if limit:
                q = q.limit(limit)
            rows = session.execute(q).scalars().all()
            return [
                {
                    "id": r.id,
                    "symbol": r.symbol,
                    "trade_date": r.trade_date,
                    "report_type": r.report_type,
                    "storage_key": r.storage_key,
                    "file_format": r.file_format,
                    "size_bytes": r.size_bytes,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def list_recommendation_runs(self, limit: int | None = 50) -> list[dict]:
        from db.models import RecommendationRun
        with get_session() as session:
            q = (select(RecommendationRun)
                .where(_visible(RecommendationRun))
                .order_by(RecommendationRun.created_at.desc()))
            if limit:
                q = q.limit(limit)
            rows = session.execute(q).scalars().all()
            return [
                {
                    "id": r.id,
                    "trade_date": r.trade_date,
                    "symbols_csv": r.symbols_csv,
                    "model_used": r.model_used,
                    "provider_used": r.provider_used,
                    "result_json": r.result_json,
                    "duration_ms": r.duration_ms,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def get_recommendation_for_run(self, run_id: str) -> dict | None:
        """The newest visible synthesis a run stored, or None
        (ix_recommendation_run)."""
        from db.models import RecommendationRun
        if not run_id:
            return None
        with get_session() as session:
            r = session.execute(
                select(RecommendationRun)
                .where(RecommendationRun.run_id == run_id)
                .where(_visible(RecommendationRun))
                .order_by(RecommendationRun.created_at.desc())
                .limit(1)
            ).scalars().first()
            if r is None:
                return None
            return {
                "id": r.id,
                "trade_date": r.trade_date,
                "symbols_csv": r.symbols_csv,
                "model_used": r.model_used,
                "provider_used": r.provider_used,
                "result_json": r.result_json,
                "duration_ms": r.duration_ms,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }

    def get_predictions_for_run(self, run_id: str,
                                include_details: bool = False) -> list[dict]:
        """Every visible prediction a run stored, by symbol then model
        (ix on model_predictions.run_id). The run page's one prediction
        read; the same dict shape get_predictions_between returns."""
        from db.models import ModelPrediction
        if not run_id:
            return []
        with get_session() as session:
            rows = session.execute(
                select(ModelPrediction)
                .where(ModelPrediction.run_id == run_id,
                       _visible(ModelPrediction))
                .order_by(ModelPrediction.symbol, ModelPrediction.model_name)
            ).scalars().all()
            return [_pred_to_dict(r, include_details) for r in rows]

    def list_all_predictions(
        self, symbol: str | None = None, limit: int = 100,
    ) -> list[dict]:
        from db.models import ModelPrediction
        with get_session() as session:
            q = select(ModelPrediction).where(_visible(ModelPrediction)).order_by(
                ModelPrediction.prediction_date.desc(),
                ModelPrediction.symbol,
                ModelPrediction.model_name,
            )
            if symbol:
                q = q.where(ModelPrediction.symbol == symbol.upper())
            rows = session.execute(q.limit(limit)).scalars().all()
            return [
                {
                    "id": r.id,
                    "symbol": r.symbol,
                    "model_name": r.model_name,
                    "prediction_date": str(r.prediction_date),
                    "target_date": str(r.target_date),
                    "decision": r.decision,
                    "confidence": r.confidence,
                    "up_probability": r.up_probability,
                    "previous_close": r.previous_close,
                    "actual_close": r.actual_close,
                    "was_correct": r.was_correct,
                    "pnl_dollars": r.pnl_dollars,
                    "model_version": r.model_version,
                    "duration_ms": r.duration_ms,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def get_latest_prediction_date(self, kind: str | None = None) -> "date | None":
        """The most recent data cutoff any visible prediction was made from.

        The launch screen is built around this cohort rather than around a run,
        because predictions carry no run id -- the date is the only grouping
        the schema actually guarantees. ``kind`` narrows it to cutoffs written
        by scheduled or manual runs (see _by_run_kind).
        """
        from db.models import ModelPrediction
        with get_session() as session:
            q = (select(func.max(ModelPrediction.prediction_date))
                 .where(_visible(ModelPrediction)))
            return session.execute(_by_run_kind(q, kind)).scalar()

    def get_prediction_dates(self, limit: int = 90,
                             kind: str | None = None) -> list["date"]:
        """Every distinct prediction cutoff, newest first.

        Feeds the Home cutoff selector: the board can be pointed at any past
        cutoff, not only the latest, so "what did we call that morning" is
        one dropdown away instead of a Performance-page filter safari.
        ``kind`` lists only the cutoffs a scheduled (or manual) run wrote.
        """
        from db.models import ModelPrediction
        with get_session() as session:
            q = (
                select(ModelPrediction.prediction_date)
                .where(_visible(ModelPrediction))
                .distinct()
                .order_by(ModelPrediction.prediction_date.desc())
                .limit(limit)
            )
            rows = session.execute(_by_run_kind(q, kind)).scalars().all()
            return list(rows)

    def get_predictions_between(
        self,
        start: "date",
        end: "date",
        symbols: list[str] | None = None,
        include_details: bool = False,
        kind: str | None = None,
    ) -> list[dict]:
        """Visible predictions with a cutoff in [start, end], inclusive.

        Date-scoped so callers stop pulling a fixed row count and filtering in
        Python; hits the (symbol, prediction_date) index. Serves both the
        single-day cohort (start == end) and trailing-window aggregates.

        ``symbols`` None means every symbol; a list restricts to those names,
        and an empty list is an empty answer (a board over an empty watchlist
        has no rows, not every row). ``kind`` keeps only rows written by
        scheduled or manual runs (see _by_run_kind).

        details_json is opt-in because it carries per-model payloads that are
        pure weight on a multi-week range scan.
        """
        from db.models import ModelPrediction
        if symbols is not None and not symbols:
            return []
        with get_session() as session:
            q = (
                select(ModelPrediction)
                .where(
                    _visible(ModelPrediction),
                    ModelPrediction.prediction_date >= start,
                    ModelPrediction.prediction_date <= end,
                )
                .order_by(
                    ModelPrediction.prediction_date.desc(),
                    ModelPrediction.symbol,
                    ModelPrediction.model_name,
                )
            )
            if symbols:
                q = q.where(ModelPrediction.symbol.in_([s.upper() for s in symbols]))
            return [_pred_to_dict(r, include_details) for r in
                    session.execute(_by_run_kind(q, kind)).scalars().all()]

    def query_predictions(
        self,
        symbols: list[str] | None = None,
        start: "date | None" = None,
        end: "date | None" = None,
        model: str | None = None,
        outcome: str | None = None,
        kind: str | None = None,
    ) -> list[dict]:
        """Every visible prediction matching the History filters, no row cap.

        Filters run in SQL (on target_date, the session predicted) so the
        page sees the whole matching set and paginates it, instead of the
        old newest-1000 slice that rendered any older date as an empty day.
        ``outcome``: pending (no verdict and no P&L), right, wrong.

        ``kind`` restricts to rows written by scheduled or manual runs (see
        _by_run_kind). The Performance page asks for "scheduled": an ad-hoc
        rerun writes its own row for a call the daily job already made, and
        counting both reports one call as two trades.
        """
        from db.models import ModelPrediction
        with get_session() as session:
            q = (select(ModelPrediction)
                 .where(_visible(ModelPrediction))
                 .order_by(ModelPrediction.target_date.desc(),
                           ModelPrediction.symbol, ModelPrediction.model_name))
            if symbols:
                q = q.where(ModelPrediction.symbol.in_([s.upper() for s in symbols]))
            if start is not None:
                q = q.where(ModelPrediction.target_date >= start)
            if end is not None:
                q = q.where(ModelPrediction.target_date <= end)
            if model and model != "all":
                q = q.where(ModelPrediction.model_name == model)
            if outcome == "pending":
                q = q.where(ModelPrediction.was_correct.is_(None),
                            ModelPrediction.pnl_dollars.is_(None))
            elif outcome == "right":
                q = q.where(ModelPrediction.was_correct.is_(True))
            elif outcome == "wrong":
                q = q.where(ModelPrediction.was_correct.is_(False))
            return [_pred_to_dict(r) for r in
                    session.execute(_by_run_kind(q, kind)).scalars().all()]

    def get_open_predictions(self, limit: int = 200) -> list[dict]:
        """Calls that have been made but cannot be scored yet.

        Unresolved means no actual_close: evaluation is deferred until the
        target session closes and the evaluator runs, so these are genuinely
        in flight rather than failed.
        """
        from db.models import ModelPrediction
        today = datetime.now().date()
        with get_session() as session:
            rows = session.execute(
                select(ModelPrediction)
                .where(
                    _visible(ModelPrediction),
                    ModelPrediction.actual_close.is_(None),
                    ModelPrediction.target_date >= today,
                )
                .order_by(
                    ModelPrediction.target_date,
                    ModelPrediction.symbol,
                    ModelPrediction.model_name,
                )
                .limit(limit)
            ).scalars().all()
            return [_pred_to_dict(r) for r in rows]

    def get_report_content(self, storage_key: str) -> str | None:
        try:
            from services.storage_service import download_report
            return download_report(storage_key)
        except Exception as e:
            logger.warning("Failed to download report %s: %s", storage_key, e)
            return None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        pass

    def __enter__(self) -> "CacheService":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Singleton instance
_cache_instance: Optional[CacheService] = None


def get_cache() -> CacheService:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance
