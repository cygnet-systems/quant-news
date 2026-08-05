"""PostgreSQL caching service for stock data and model predictions.

Provides a caching layer using PostgreSQL via SQLAlchemy to minimize API calls
and enable fast local data access with SQL query capabilities.

Migrated from DuckDB — same public API, Postgres backend.
"""

import hashlib
import json
import logging
from datetime import date, datetime
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


def _current_uid():
    """Cygnet SSO uid for this request, or None (anonymous / subprocess)."""
    try:
        from services.auth_service import current_uid
        return current_uid()
    except Exception:
        return None


def _visible(model_cls):
    """Row-visibility clause: public rows (or legacy NULL-owner rows) plus
    the signed-in user's own. With everything public today this filters
    nothing; it becomes load-bearing the moment private rows exist."""
    cond = model_cls.is_public.is_(True) | model_cls.owner_uid.is_(None)
    uid = _current_uid()
    if uid:
        cond = cond | (model_cls.owner_uid == uid)
    return cond


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
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825,
        }
        return period_map.get(period, 365)

    def is_cached(self, symbol: str, data_type: str = "prices", period: Optional[str] = None) -> bool:
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
        now = datetime.now()

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
                oldest = session.execute(
                    select(func.min(StockPrice.date)).where(StockPrice.symbol == symbol)
                ).scalar()

            if oldest:
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

        now = datetime.now()

        with get_session() as session:
            session.execute(delete(StockPrice).where(StockPrice.symbol == symbol))

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
            if existing_meta:
                existing_meta.period = period
                existing_meta.last_updated = now
                existing_meta.record_count = len(df)
            else:
                session.add(CacheMetadata(
                    symbol=symbol, data_type="prices", period=period,
                    last_updated=now, record_count=len(df),
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
                fetched_at=datetime.now(),
            ))
            existing_meta = session.execute(
                select(CacheMetadata).where(
                    CacheMetadata.symbol == info.symbol,
                    CacheMetadata.data_type == "info",
                )
            ).scalar_one_or_none()
            if existing_meta:
                existing_meta.last_updated = datetime.now()
                existing_meta.record_count = 1
            else:
                session.add(CacheMetadata(
                    symbol=info.symbol, data_type="info", period=None,
                    last_updated=datetime.now(), record_count=1,
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
    ) -> None:
        from db.models import ModelPrediction, StockPrice
        from utils.trading_calendar import get_next_trading_day

        symbol = symbol.upper()
        pred_date = date.fromisoformat(prediction_date_str) if prediction_date_str else date.today()
        target = get_next_trading_day(pred_date)
        pred_id = f"{symbol}_{model_name}_{pred_date:%Y%m%d}"

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
            prev_close_row = session.execute(
                select(StockPrice.close)
                .where(StockPrice.symbol == symbol, StockPrice.date <= str(pred_date))
                .order_by(StockPrice.date.desc())
                .limit(1)
            ).scalar_one_or_none()

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
                input_data_hash=data_hash,
                created_at=datetime.now(),
                # Re-storing a prediction id invalidates any prior evaluation:
                # merge() used to keep old actual_close/was_correct/pnl from a
                # DIFFERENT decision (12% of rows had scores contradicting
                # their own decision), and the evaluator never revisits rows
                # with actual_close set. Explicit Nones force a re-score.
                actual_close=None,
                was_correct=None,
                pnl_dollars=None,
                evaluated_at=None,
                # Ownership: stamped from the signed-in user; everything is
                # public by default until a privacy toggle exists.
                owner_uid=_current_uid(),
                is_public=True,
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
        ask "has this exact analysis already been done?" — which is what lets
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
                    # The bar this call was made against — a reuse check needs
                    # it to tell "already done" from "done on stale data".
                    "previous_close": r.previous_close,
                    "error": None,
                }
            return results

    def evaluate_predictions(self) -> int:
        from db.models import ModelPrediction, StockPrice
        from models.base import compute_pnl
        from utils.trading_calendar import get_previous_trading_day, is_market_open_today

        today = date.today()
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
            if latest is None or str(latest) < str(max_target):
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
            hold_bands: dict = {}
            for pred in pending:
                actual_row = session.execute(
                    select(StockPrice.close).where(
                        StockPrice.symbol == pred.symbol,
                        StockPrice.date == str(pred.target_date),
                    )
                ).scalar_one_or_none()

                if actual_row is None:
                    continue
                if pred.previous_close is None or pred.previous_close <= 0:
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
                    was_correct = move <= self._hold_band(
                        session, pred.symbol, hold_bands)

                pnl = compute_pnl(pred.decision, pred.previous_close, actual_close)

                pred.actual_close = actual_close
                pred.was_correct = was_correct
                pred.pnl_dollars = pnl
                pred.evaluated_at = datetime.now()
                evaluated += 1

            if evaluated > 0:
                logger.info(f"Evaluated {evaluated} predictions")

        return evaluated

    def _hold_band(self, session, symbol: str, _cache: dict) -> float:
        """No-trade band for a symbol: its own typical absolute daily move.

        Judging every name against one fixed percentage compares a utility
        against a biotech. This uses the symbol's median absolute daily return
        so "the session was quiet for this stock" means the same thing
        everywhere. Falls back to the fixed band when history is too short.
        """
        from db.models import StockPrice
        from config import MODEL

        if symbol in _cache:
            return _cache[symbol]

        band = MODEL.HOLD_BAND_PCT
        try:
            closes = session.execute(
                select(StockPrice.close)
                .where(StockPrice.symbol == symbol)
                .order_by(StockPrice.date.asc())
            ).scalars().all()
            if len(closes) > MODEL.HOLD_BAND_MIN_HISTORY:
                moves = [
                    abs(closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(1, len(closes))
                    if closes[i - 1]
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

        _cache[symbol] = band
        return band

    def backfill_hold_scores(self) -> int:
        """Score HOLD rows that were resolved before HOLDs were scorable.

        `evaluate_predictions` only considers rows with `actual_close IS NULL`,
        so rows already resolved under the old rule — where a HOLD got
        `was_correct = None` — are permanently invisible to it. Any future
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
                if not pred.previous_close or pred.previous_close <= 0:
                    continue
                move = abs(pred.actual_close - pred.previous_close) / pred.previous_close
                pred.was_correct = move <= self._hold_band(session, pred.symbol, bands)
                if pred.pnl_dollars is None:
                    pred.pnl_dollars = 0.0
                pred.evaluated_at = datetime.now()
                updated += 1

        if updated:
            logger.info(f"Backfilled {updated} HOLD scores across "
                        f"{len(bands)} symbol bands")
        return updated

    def rescore_hold_predictions(self) -> int:
        """Clear and recompute every HOLD verdict under the current band.

        Separate from `backfill_hold_scores`, which only fills gaps. Use this
        after changing HOLD_BAND_* so history reflects one consistent rule
        rather than a mix of whatever was configured when each row resolved.
        """
        from db.models import ModelPrediction

        updated = 0
        bands: dict = {}
        with get_session() as session:
            rows = session.execute(
                select(ModelPrediction).where(
                    ModelPrediction.decision == "HOLD",
                    ModelPrediction.actual_close.isnot(None),
                )
            ).scalars().all()
            for pred in rows:
                if not pred.previous_close or pred.previous_close <= 0:
                    continue
                move = abs(pred.actual_close - pred.previous_close) / pred.previous_close
                pred.was_correct = move <= self._hold_band(session, pred.symbol, bands)
                pred.evaluated_at = datetime.now()
                updated += 1

        if updated:
            logger.info(f"Rescored {updated} HOLD predictions across "
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
        output_tokens: int = 0,
    ) -> None:
        import uuid
        from db.models import TradingAgentReport

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
                owner_uid=_current_uid(),
                is_public=True,
            ))

    def get_trading_agent_reports(self, symbol: str, limit: int = 20) -> list[dict]:
        from db.models import TradingAgentReport
        with get_session() as session:
            rows = session.execute(
                select(TradingAgentReport)
                .where(TradingAgentReport.symbol == symbol.upper())
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
                }
                for r in rows
            ]

    def get_all_trading_agent_reports(self, limit: int = 50) -> list[dict]:
        from db.models import TradingAgentReport
        with get_session() as session:
            rows = session.execute(
                select(TradingAgentReport)
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
                }
                for r in rows
            ]

    def get_model_accuracy(self, model_name: str, symbol: Optional[str] = None) -> dict:
        """Accuracy for a model.

        `total`/`correct`/`accuracy` cover ACTIVE (BUY/SELL) calls only, so the
        headline number keeps meaning "directional hit rate" now that HOLDs are
        also scored. HOLD accountability is reported alongside as hold_*, and
        `all_*` covers both — mixing them into one rate would compare a
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
                        evaluated_at=datetime.now(),
                    ))
                    inserted += 1
                except Exception:
                    pass
        return inserted

    def get_unevaluated_predictions_for_strategy(self, strategy_name: str) -> list[dict]:
        from db.models import ModelPrediction, StrategyEvaluation
        with get_session() as session:
            subq = select(StrategyEvaluation.prediction_id).where(
                StrategyEvaluation.strategy_name == strategy_name
            ).scalar_subquery()

            rows = session.execute(
                select(ModelPrediction)
                .where(
                    ModelPrediction.actual_close.isnot(None),
                    ModelPrediction.id.notin_(subq),
                )
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
        # to the invalid JSON token "Infinity" and fail the whole insert —
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
                computed_at=datetime.now(),
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
                        fetched_at=datetime.now(),
                    ))
                    inserted += 1
                except Exception:
                    pass
        return inserted

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
                    "overall_sentiment_score": r.overall_sentiment_score,
                    "overall_sentiment_label": r.overall_sentiment_label,
                    "ticker_sentiment_score": r.ticker_sentiment_score,
                    "ticker_relevance_score": r.ticker_relevance_score,
                    "fetched_at": r.fetched_at,
                }
                for r in rows
            ]

    # =========================================================================
    # Historical Data Queries
    # =========================================================================

    def list_report_catalog(self, limit: int = 50) -> list[dict]:
        from db.models import ReportCatalog
        with get_session() as session:
            rows = session.execute(
                select(ReportCatalog)
                .where(_visible(ReportCatalog))
                .order_by(ReportCatalog.created_at.desc())
                .limit(limit)
            ).scalars().all()
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

    def list_recommendation_runs(self, limit: int = 50) -> list[dict]:
        from db.models import RecommendationRun
        with get_session() as session:
            rows = session.execute(
                select(RecommendationRun)
                .where(_visible(RecommendationRun))
                .order_by(RecommendationRun.created_at.desc())
                .limit(limit)
            ).scalars().all()
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
