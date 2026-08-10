"""Persistence service — orchestrates Postgres catalog + S3 object storage.

Cache invalidation strategy:
    1. Hash the input data (stock prices + news + features) for a (symbol, date).
    2. Store the hash in data_snapshots.
    3. Before running a prediction/report, check if a cached result exists
       with the SAME input_data_hash.
    4. If yes → return cached result (skip model run / LLM call).
    5. If no  → run the model, store the result + hash.

This means a report generated on 2026-07-13 for EIX stays valid as long as the
underlying stock data and news for that date haven't changed.
"""

import hashlib
import json
import logging
import time
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select

from db.models import DataSnapshot, ModelPrediction, RecommendationRun, ReportCatalog
from db.session import get_session
from services import storage_service

logger = logging.getLogger(__name__)


def _current_uid():
    """Uid to attribute writes to: the signed-in user in a request thread,
    or the owning user of a scheduled run (QUANTNEWS_RUN_OWNER). None means
    anonymous/public."""
    try:
        from services.auth_service import effective_uid
        return effective_uid()
    except Exception:
        return None


def _default_public() -> bool:
    """Public unless this process is a private scheduled run."""
    try:
        from services.auth_service import run_is_public
        return run_is_public()
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Input data hashing
# ---------------------------------------------------------------------------

def compute_data_hash(data: dict | list | str) -> str:
    """Compute a stable SHA-256 hash of input data.

    Sorts dict keys for determinism. Handles nested structures.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_prediction_input_hash(
    symbol: str,
    trade_date: str,
    stock_data: dict,
    news_data: dict | None = None,
) -> str:
    """Hash the inputs that feed into a prediction for a symbol+date.

    Includes stock prices and news so that if either changes, the
    prediction is invalidated.
    """
    payload = {
        "symbol": symbol,
        "trade_date": trade_date,
        "stock_data_hash": compute_data_hash(stock_data),
    }
    if news_data:
        payload["news_data_hash"] = compute_data_hash(news_data)
    return compute_data_hash(payload)


# ---------------------------------------------------------------------------
# Data snapshots (input data fingerprints)
# ---------------------------------------------------------------------------

def upsert_data_snapshot(
    symbol: str,
    trade_date: str,
    data_type: str,
    content_hash: str,
    record_count: int | None = None,
) -> bool:
    """Store or update the content hash for a (symbol, date, type) key.

    Returns True if the hash changed (data is new/updated).
    """
    with get_session() as session:
        stmt = select(DataSnapshot).where(
            DataSnapshot.symbol == symbol,
            DataSnapshot.trade_date == trade_date,
            DataSnapshot.data_type == data_type,
        )
        existing = session.execute(stmt).scalar_one_or_none()

        if existing:
            if existing.content_hash == content_hash:
                return False
            existing.content_hash = content_hash
            existing.record_count = record_count
            existing.created_at = datetime.utcnow()
            return True

        session.add(DataSnapshot(
            symbol=symbol,
            trade_date=trade_date,
            data_type=data_type,
            content_hash=content_hash,
            record_count=record_count,
        ))
        return True


def get_data_hash(symbol: str, trade_date: str, data_type: str) -> Optional[str]:
    """Get the stored content hash for a snapshot key, or None."""
    with get_session() as session:
        stmt = select(DataSnapshot.content_hash).where(
            DataSnapshot.symbol == symbol,
            DataSnapshot.trade_date == trade_date,
            DataSnapshot.data_type == data_type,
        )
        return session.execute(stmt).scalar_one_or_none()


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

def get_cached_prediction(
    symbol: str,
    trade_date: str,
    model_name: str,
    input_data_hash: str,
) -> Optional[dict]:
    """Return cached prediction result if it exists for this exact input hash.

    Matches on (symbol, prediction_date, model_name, input_data_hash) in the
    merged model_predictions table.
    """
    pred_date = date.fromisoformat(trade_date)
    with get_session() as session:
        stmt = select(ModelPrediction).where(
            ModelPrediction.symbol == symbol,
            ModelPrediction.prediction_date == pred_date,
            ModelPrediction.model_name == model_name,
            ModelPrediction.input_data_hash == input_data_hash,
        )
        row = session.execute(stmt).scalars().first()
        if row and row.details_json:
            logger.info(f"Cache hit: {model_name}/{symbol}@{trade_date}")
            return row.details_json
    return None


def store_prediction(
    symbol: str,
    trade_date: str,
    model_name: str,
    input_data_hash: str,
    result: dict,
    model_version: str | None = None,
    duration_ms: int | None = None,
) -> None:
    """Store a prediction result using the merged ModelPrediction table.

    Uses session.merge() with the string ID format "SYMBOL_modelname_YYYYMMDD"
    so that re-running for the same (symbol, model, date) upserts.
    """
    pred_date = date.fromisoformat(trade_date)
    pred_id = f"{symbol}_{model_name}_{pred_date:%Y%m%d}"

    # target_date defaults to the next trading day; fall back to pred_date+1
    try:
        from utils.trading_calendar import get_next_trading_day
        target = get_next_trading_day(pred_date)
    except Exception:
        from datetime import timedelta
        target = pred_date + timedelta(days=1)

    with get_session() as session:
        session.merge(ModelPrediction(
            id=pred_id,
            symbol=symbol,
            model_name=model_name,
            prediction_date=pred_date,
            target_date=target,
            decision=result.get("decision", "HOLD"),
            confidence=result.get("confidence"),
            up_probability=result.get("up_probability"),
            details_json=result,
            input_data_hash=input_data_hash,
            model_version=model_version,
            duration_ms=duration_ms,
            # Re-storing an id invalidates any prior evaluation — same rule as
            # cache_service.store_prediction. Without these explicit Nones the
            # merge keeps the OLD verdict against the NEW decision, and the
            # evaluator never revisits a row whose actual_close survived.
            actual_close=None,
            was_correct=None,
            pnl_dollars=None,
            evaluated_at=None,
            # Ownership must ride the merge too: this writer used to omit it,
            # and merge() then NULLed owner_uid on rows the cache_service
            # writer had stamped — silently un-owning them.
            owner_uid=_current_uid(),
            is_public=_default_public(),
        ))
    logger.info(f"Stored prediction: {model_name}/{symbol}@{trade_date}")


# ---------------------------------------------------------------------------
# Report cache (catalog + S3)
# ---------------------------------------------------------------------------

def get_cached_report(
    symbol: str | None,
    trade_date: str,
    report_type: str,
    input_data_hash: str,
    raw: bool = False,
) -> Optional[str | bytes]:
    """Return cached report content if it exists for this exact input hash.

    Downloads from S3. Returns text by default, or raw bytes if raw=True.
    """
    with get_session() as session:
        stmt = select(ReportCatalog).where(
            ReportCatalog.trade_date == trade_date,
            ReportCatalog.report_type == report_type,
            ReportCatalog.input_data_hash == input_data_hash,
        )
        if symbol:
            stmt = stmt.where(ReportCatalog.symbol == symbol)
        # Newest match, not "the only match". One input hash can legitimately
        # have more than one catalog row (the storage-key layout changed, or
        # two processes stored concurrently), and scalar_one_or_none() RAISES
        # on that — turning a duplicate into a permanently broken cache for
        # that date instead of a cache hit.
        stmt = stmt.order_by(ReportCatalog.created_at.desc()).limit(1)
        row = session.execute(stmt).scalars().first()
        if not row:
            return None

    content = storage_service.download_report(row.storage_key)
    if content:
        logger.info(f"Cache hit: report {report_type}/{symbol}@{trade_date}")
        return content if raw else content.decode("utf-8")
    return None


def store_report(
    symbol: str | None,
    trade_date: str,
    report_type: str,
    input_data_hash: str,
    content: str | bytes,
    file_format: str = "md",
    metadata: dict | None = None,
) -> str:
    """Store a report in S3 and record its catalog entry in Postgres.

    Returns the S3 storage key.
    """
    sym_part = symbol or "portfolio"
    # The input hash is part of the path, not just the catalog row. Without it
    # every portfolio report for a date shared one key, so a run over a
    # DIFFERENT symbol set overwrote the previous one — the 20-symbol morning
    # report was replaced by a 2-symbol ad-hoc run, and only the newest scope
    # stayed cacheable. Distinct inputs now coexist and each stays retrievable.
    storage_key = (f"reports/{sym_part}/{trade_date}/"
                   f"{report_type}-{input_data_hash[:12]}.{file_format}")

    size_bytes = storage_service.upload_report(
        storage_key,
        content,
        content_type=_mime_for_format(file_format),
        metadata=metadata,
    )

    with get_session() as session:
        # storage_key is unique but not the PK — upsert by key so re-running
        # a report for the same symbol/date/type overwrites the catalog row.
        existing = session.execute(
            select(ReportCatalog).where(ReportCatalog.storage_key == storage_key)
        ).scalar_one_or_none()
        if existing:
            existing.input_data_hash = input_data_hash
            existing.file_format = file_format
            existing.size_bytes = size_bytes
            existing.metadata_json = metadata
        else:
            session.add(ReportCatalog(
                symbol=symbol,
                trade_date=trade_date,
                report_type=report_type,
                input_data_hash=input_data_hash,
                storage_key=storage_key,
                file_format=file_format,
                size_bytes=size_bytes,
                metadata_json=metadata,
                owner_uid=_current_uid(),
                is_public=_default_public(),
            ))

    logger.info(f"Stored report: {storage_key} ({size_bytes} bytes)")
    return storage_key


# ---------------------------------------------------------------------------
# Recommendation cache
# ---------------------------------------------------------------------------

def get_cached_recommendation(
    trade_date: str,
    input_data_hash: str,
) -> Optional[dict]:
    """Return cached recommendation if it exists for this exact input hash."""
    with get_session() as session:
        # store_recommendation() inserts (merge on a PK-less row is an
        # INSERT), so repeated identical runs can leave several rows for one
        # hash. Take the newest rather than raising on the second one.
        stmt = (select(RecommendationRun)
                .where(RecommendationRun.trade_date == trade_date,
                       RecommendationRun.input_data_hash == input_data_hash)
                .order_by(RecommendationRun.created_at.desc())
                .limit(1))
        row = session.execute(stmt).scalars().first()
        if row and row.result_json:
            logger.info(f"Cache hit: recommendation@{trade_date}")
            return row.result_json
    return None


def store_recommendation(
    trade_date: str,
    symbols: list[str],
    input_data_hash: str,
    result: dict,
    model_used: str,
    provider_used: str,
    duration_ms: int | None = None,
) -> None:
    """Store a recommendation result."""
    with get_session() as session:
        session.merge(RecommendationRun(
            trade_date=trade_date,
            symbols_csv=",".join(symbols),
            input_data_hash=input_data_hash,
            model_used=model_used,
            provider_used=provider_used,
            result_json=result,
            duration_ms=duration_ms,
            owner_uid=_current_uid(),
            is_public=_default_public(),
        ))
    logger.info(f"Stored recommendation: {trade_date} ({len(symbols)} symbols)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mime_for_format(fmt: str) -> str:
    return {
        "md": "text/markdown",
        "html": "text/html",
        "pdf": "application/pdf",
        "json": "application/json",
    }.get(fmt, "application/octet-stream")
