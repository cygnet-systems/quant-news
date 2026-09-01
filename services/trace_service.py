"""Per-run trace capture and retrieval.

Write side: :func:`record_llm_call` stores one ``llm_traces`` row per
physical LLM API call — the exact bodies and request parameters, captured
before any parsing so truncated/unparseable responses survive. Attribution
(run/stage/section/symbol) comes from the same ContextVar the cost telemetry
uses, so a trace row and its ``llm_usage`` twin always agree on what the call
was for. Token/cost math stays in ``llm_usage`` alone.

Read side: the queries behind the Trace page. The list query deliberately
never selects the Text bodies — a run can hold dozens of multi-kilobyte
prompts, and the list only needs the envelope; :func:`get_llm_call_bodies`
fetches one row's bodies when the user expands it.

Everything here is best-effort telemetry: a write must never break the call
path it observes, and a read failure renders as an empty page, not a crash.
"""

from __future__ import annotations

import json
import logging
from datetime import date as date_cls
from typing import Optional

logger = logging.getLogger(__name__)

# Cap per-body storage. Nothing in the pipeline legitimately exceeds this —
# a runaway payload should not be able to bloat the table unboundedly.
_MAX_BODY_CHARS = 200_000


def _clip(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    if len(text) > _MAX_BODY_CHARS:
        return text[:_MAX_BODY_CHARS] + f"\n…[truncated, {len(text)} chars total]"
    return text


def record_llm_call(
    *,
    provider: Optional[str],
    model: Optional[str],
    system_prompt: Optional[str],
    prompt: Optional[str],
    response: Optional[str],
    params: Optional[dict],
    attempt: int,
    ok: bool,
    error: Optional[str] = None,
    duration_ms: Optional[int] = None,
    usage_id: Optional[int] = None,
) -> None:
    """Write one llm_traces row. Never raises — same contract as usage.record."""
    try:
        from db.models import LLMTrace
        from db.session import get_session
        from services import progress_service as prog
        from services import usage_service

        ctx = usage_service.current()

        trade_date = None
        if ctx.trade_date:
            try:
                trade_date = date_cls.fromisoformat(str(ctx.trade_date)[:10])
            except ValueError:
                trade_date = None

        # Same owner/visibility conventions as every other output table.
        try:
            from services.auth_service import effective_uid, run_is_public
            owner_uid = effective_uid()
            is_public = run_is_public()
        except Exception:
            owner_uid, is_public = None, True

        # Params must be JSON-clean (provider SDKs put dicts/enums in here).
        params_json = None
        if params:
            try:
                params_json = json.loads(json.dumps(params, default=str))
            except (TypeError, ValueError):
                params_json = {"unserializable": str(params)[:500]}

        with get_session() as session:
            session.add(LLMTrace(
                usage_id=usage_id,
                run_id=prog.current_run_id(),
                stage=ctx.stage,
                section=ctx.section,
                symbol=ctx.symbol,
                trade_date=trade_date,
                provider=provider,
                model=model,
                system_prompt=_clip(system_prompt),
                prompt=_clip(prompt),
                response=_clip(response),
                params_json=params_json,
                attempt=int(attempt or 1),
                ok=bool(ok),
                error=(error or None) and str(error)[:2000],
                duration_ms=duration_ms,
                owner_uid=owner_uid,
                is_public=is_public,
            ))
    except Exception as e:
        logger.debug(f"llm trace write failed: {e}")


# ---------------------------------------------------------------------------
# Trace page queries
# ---------------------------------------------------------------------------


def _visible_clause():
    """Same visibility rule the activity log applies: admins see everything,
    everyone else sees public/legacy rows plus their own."""
    from sqlalchemy import true

    from db.models import LLMTrace
    from services import progress_service as prog

    if prog.viewer_is_admin():
        return true()
    cond = LLMTrace.is_public.is_(True) | LLMTrace.owner_uid.is_(None)
    try:
        from services.auth_service import current_uid
        uid = current_uid()
    except Exception:
        uid = None
    if uid:
        cond = cond | (LLMTrace.owner_uid == uid)
    return cond


def list_llm_calls(run_id: str, after_id: int = 0, limit: int = 500) -> list[dict]:
    """The envelope of every LLM call in a run — no Text bodies.

    ``after_id`` supports incremental polling (WHERE run_id = X AND id > N).
    ``error`` is truncated server-side so a stack trace cannot drag the whole
    body budget through the list query.
    """
    try:
        from sqlalchemy import func, select

        from db.models import LLMTrace, LLMUsage
        from db.session import get_session

        with get_session() as session:
            rows = session.execute(
                select(
                    LLMTrace.id, LLMTrace.usage_id, LLMTrace.stage,
                    LLMTrace.section, LLMTrace.symbol, LLMTrace.provider,
                    LLMTrace.model, LLMTrace.attempt, LLMTrace.ok,
                    func.left(LLMTrace.error, 200).label("error"),
                    LLMTrace.duration_ms, LLMTrace.created_at,
                    LLMUsage.input_tokens, LLMUsage.output_tokens,
                    LLMUsage.cost_usd,
                )
                .join(LLMUsage, LLMUsage.id == LLMTrace.usage_id, isouter=True)
                .where(LLMTrace.run_id == run_id,
                       LLMTrace.id > int(after_id or 0),
                       _visible_clause())
                .order_by(LLMTrace.id.asc())
                .limit(limit)
            ).all()
        return [{
            "id": r.id, "usage_id": r.usage_id, "stage": r.stage,
            "section": r.section, "symbol": r.symbol, "provider": r.provider,
            "model": r.model, "attempt": r.attempt, "ok": r.ok,
            "error": r.error, "duration_ms": r.duration_ms,
            "created_at": r.created_at,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd,
        } for r in rows]
    except Exception as e:
        logger.debug(f"llm trace list query failed: {e}")
        return []


def get_llm_call_bodies(trace_id: int) -> Optional[dict]:
    """The Text bodies of ONE call, fetched only when the user expands it."""
    try:
        from sqlalchemy import select

        from db.models import LLMTrace
        from db.session import get_session

        with get_session() as session:
            row = session.execute(
                select(LLMTrace.system_prompt, LLMTrace.prompt,
                       LLMTrace.response, LLMTrace.params_json,
                       LLMTrace.error)
                .where(LLMTrace.id == int(trace_id), _visible_clause())
            ).first()
        if row is None:
            return None
        return {
            "system_prompt": row.system_prompt,
            "prompt": row.prompt,
            "response": row.response,
            "params": row.params_json,
            "error": row.error,
        }
    except Exception as e:
        logger.debug(f"llm trace body query failed: {e}")
        return None


def list_run_events(run_id: str, after_id: int = 0, limit: int = 1000) -> list[dict]:
    """A run's activity_log events including payloads, oldest first.

    Visibility mirrors get_activity_runs: admins read every user's rows,
    everyone else only their own bucket.
    """
    try:
        from sqlalchemy import select

        from db.models import ActivityLog
        from db.session import get_session
        from services import progress_service as prog

        conds = [ActivityLog.run_id == run_id,
                 ActivityLog.id > int(after_id or 0)]
        if not prog.viewer_is_admin():
            conds.append(ActivityLog.user_id == prog._user_id())

        with get_session() as session:
            rows = session.execute(
                select(ActivityLog).where(*conds)
                .order_by(ActivityLog.id.asc())
                .limit(limit)
            ).scalars().all()
        return [{
            "id": r.id, "ts": r.created_at, "stage": r.stage,
            "message": r.message, "payload": r.payload,
        } for r in rows]
    except Exception as e:
        logger.debug(f"trace event query failed: {e}")
        return []


def list_run_predictions(run_id: str) -> list[dict]:
    """model_predictions joined to a run by the new run_id column."""
    try:
        from sqlalchemy import select

        from db.models import ModelPrediction
        from db.session import get_session

        with get_session() as session:
            rows = session.execute(
                select(ModelPrediction.symbol, ModelPrediction.model_name,
                       ModelPrediction.decision, ModelPrediction.confidence,
                       ModelPrediction.up_probability,
                       ModelPrediction.duration_ms, ModelPrediction.news_status,
                       ModelPrediction.prediction_date,
                       ModelPrediction.target_date)
                .where(ModelPrediction.run_id == run_id)
                .order_by(ModelPrediction.symbol, ModelPrediction.model_name)
            ).all()
        return [{
            "symbol": r.symbol, "model_name": r.model_name,
            "decision": r.decision, "confidence": r.confidence,
            "up_probability": r.up_probability, "duration_ms": r.duration_ms,
            "news_status": r.news_status,
            "prediction_date": str(r.prediction_date),
            "target_date": str(r.target_date),
        } for r in rows]
    except Exception as e:
        logger.debug(f"trace prediction query failed: {e}")
        return []


def latest_run_marker() -> int:
    """Id of the newest run boundary visible to this user — the poll's cheap
    "did the run list change?" probe (start_run writes a stage='run' event
    the moment a run opens, so a new run moves this immediately)."""
    try:
        from sqlalchemy import func, select

        from db.models import ActivityLog
        from db.session import get_session
        from services import progress_service as prog

        conds = [ActivityLog.stage == "run"]
        if not prog.viewer_is_admin():
            conds.append(ActivityLog.user_id == prog._user_id())
        with get_session() as session:
            return int(session.execute(
                select(func.max(ActivityLog.id)).where(*conds)
            ).scalar() or 0)
    except Exception as e:
        logger.debug(f"run marker query failed: {e}")
        return 0


def run_watermarks(run_id: str) -> dict:
    """Newest row ids for a run — the poll's cheap "anything new?" probe."""
    try:
        from sqlalchemy import func, select

        from db.models import ActivityLog, LLMTrace
        from db.session import get_session

        with get_session() as session:
            ev_max = session.execute(
                select(func.max(ActivityLog.id))
                .where(ActivityLog.run_id == run_id)
            ).scalar() or 0
            tr_max = session.execute(
                select(func.max(LLMTrace.id))
                .where(LLMTrace.run_id == run_id)
            ).scalar() or 0
        return {"events": int(ev_max), "traces": int(tr_max)}
    except Exception as e:
        logger.debug(f"trace watermark query failed: {e}")
        return {"events": 0, "traces": 0}
