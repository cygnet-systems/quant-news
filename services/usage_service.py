"""LLM usage and cost telemetry.

Every call through :meth:`LLMService.generate` lands here as one row: exact
token counts from the provider response, the rates applied, the derived cost,
and: via :func:`track`: what the call was actually for.

This is a one-way sink. Nothing recorded here is ever read back into a prompt;
it exists to answer "what did today's run cost, and which stage spent it".

Attribution uses a ContextVar rather than threading a parameter through two
dozen call sites: the stage that starts the work declares itself once, and any
LLM call underneath it, including ones inside a model class several layers
down: inherits that label.

Context crosses ``asyncio.to_thread`` and new tasks for free. It does NOT
cross a bare ``ThreadPoolExecutor.submit``: worker threads start empty, so
the prediction pipeline submits through ``copy_context().run`` to carry the
label into the models it fans out. A stage that forgets this records as
"unknown" rather than losing the row.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageContext:
    """What the LLM call underneath this context is for.

    ``section`` narrows ``stage`` to the report section served (e.g.
    "research:BE", "ai_report:overall", "recommendations"). It rides the
    same ContextVar; only the trace table stores it. Llm_usage is keyed by
    stage alone and stays unchanged.
    """

    stage: str = "unknown"
    symbol: Optional[str] = None
    trade_date: Optional[str] = None
    section: Optional[str] = None


_context: ContextVar[UsageContext] = ContextVar(
    "llm_usage_context", default=UsageContext()
)


@contextmanager
def track(stage: str, symbol: str | None = None, trade_date: str | None = None,
          section: str | None = None):
    """Label every LLM call made inside this block."""
    token = _context.set(UsageContext(stage=stage, symbol=symbol,
                                      trade_date=trade_date, section=section))
    try:
        yield
    finally:
        _context.reset(token)


def current() -> UsageContext:
    return _context.get()


def compute_cost(model: str | None, input_tokens: int,
                 output_tokens: int) -> tuple[float | None, float | None, float | None]:
    """Return ``(cost_usd, input_rate, output_rate)``; cost is None if unpriced."""
    from config import get_llm_rates

    in_rate, out_rate = get_llm_rates(model)
    if in_rate is None or out_rate is None:
        return None, in_rate, out_rate
    cost = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
    return round(cost, 6), in_rate, out_rate


class SpendCeilingReached(RuntimeError):
    """Raised instead of buying another model call once a run is over budget.

    A RuntimeError rather than a new base: every caller of ``generate`` already
    treats an exception as "this block is unavailable" and degrades, which is
    exactly the behaviour wanted here. The run finishes and says what it could
    not afford, rather than dying or silently spending on.
    """


# Spend per run, accumulated in this process as rows are written. Deliberately
# not a query: the check runs before every model call, and a database round
# trip per call would tax the healthy path to police the pathological one.
# A run's model stage is one forked subprocess, so per-process is per-run in
# practice; a restart resets it, which fails OPEN (a resumed run gets a fresh
# budget) because a ceiling that strands a half-finished run is worse than
# one that occasionally allows a second.
_SPENT: dict[str, float] = {}
_SPENT_LOCK = threading.Lock()


def spent_on_run(run_id: str | None) -> float:
    """Dollars this process has recorded against ``run_id`` (tokens + tools)."""
    if not run_id:
        return 0.0
    with _SPENT_LOCK:
        return _SPENT.get(run_id, 0.0)


def _accrue(run_id: str | None, amount: float) -> None:
    if not run_id or not amount:
        return
    with _SPENT_LOCK:
        _SPENT[run_id] = _SPENT.get(run_id, 0.0) + amount


def reset_run_spend(run_id: str | None = None) -> None:
    """Forget accumulated spend, for one run or all. Tests and run start."""
    with _SPENT_LOCK:
        if run_id is None:
            _SPENT.clear()
        else:
            _SPENT.pop(run_id, None)


def check_spend_ceiling(stage: str = "") -> None:
    """Raise ``SpendCeilingReached`` when the current run is over budget.

    Called before a model call, never after: the point is not to spend the
    money. Never raises anything else -- a telemetry fault must not be able
    to stop a run that is within budget.
    """
    try:
        from config import RUN_SPEND_CEILING_USD
        from services import progress_service as prog

        ceiling = float(RUN_SPEND_CEILING_USD or 0)
        if ceiling <= 0:
            return
        run_id = prog.current_run_id()
        spent = spent_on_run(run_id)
        if spent < ceiling:
            return
    except SpendCeilingReached:
        raise
    except Exception as e:
        logger.debug(f"spend ceiling check skipped: {e}")
        return
    logger.error(
        "run %s has spent $%.2f, at or over the $%.2f ceiling: refusing "
        "further model calls%s. Raise RUN_SPEND_CEILING_USD to allow more.",
        run_id, spent, ceiling, f" for {stage}" if stage else "")
    raise SpendCeilingReached(
        f"run spend ${spent:.2f} reached the ${ceiling:.2f} ceiling")


def compute_tool_cost(provider: str | None,
                      searches: int) -> float | None:
    """What ``searches`` server-side web searches cost, or None if unpriced.

    Separate from token cost because it is billed separately: a provider
    charges per search regardless of how many tokens the surrounding call
    used. Zero searches is a real zero, not an unknown.
    """
    if not searches:
        return 0.0
    from config import get_web_search_rate

    rate = get_web_search_rate(provider)
    if rate is None:
        return None
    return round(searches * rate / 1000.0, 6)


def record(
    *,
    model: str | None,
    provider: str | None,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int | None = None,
    ok: bool = True,
    error: str | None = None,
    searches: int = 0,
    cached_input_tokens: int = 0,
) -> int | None:
    """Write one usage row. Never raises: telemetry must not break a run.

    Returns the new row's id (so the trace row for the same physical call
    can link to its cost record), or None when the write was skipped.
    """
    try:
        from db.models import LLMUsage
        from db.session import get_session
        from services import progress_service as prog

        ctx = current()
        cost, in_rate, out_rate = compute_cost(model, input_tokens, output_tokens)
        searches = int(searches or 0)
        tool_cost = compute_tool_cost(provider, searches)

        try:
            owner_uid = _current_uid()
        except Exception:
            owner_uid = None

        # Same run grouping the Activity Log uses, so a run's events and its
        # spend join on one id.
        run_id = prog.current_run_id()

        with get_session() as session:
            row = LLMUsage(
                run_id=run_id,
                stage=ctx.stage,
                symbol=ctx.symbol,
                trade_date=ctx.trade_date,
                provider=provider,
                model=model,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                input_rate_per_mtok=in_rate,
                output_rate_per_mtok=out_rate,
                cost_usd=cost,
                searches=searches,
                tool_cost_usd=tool_cost,
                cached_input_tokens=int(cached_input_tokens or 0),
                duration_ms=duration_ms,
                ok=ok,
                error=(error or None) and str(error)[:500],
                owner_uid=owner_uid,
            )
            session.add(row)
            session.flush()
            _accrue(run_id, (cost or 0.0) + (tool_cost or 0.0))
            return row.id
    except Exception as e:
        logger.debug(f"usage telemetry write failed: {e}")
        return None


def _current_uid() -> str | None:
    from services.auth_service import current_uid
    return current_uid()


def summarize(days: int = 7) -> list[dict]:
    """Spend grouped by day and stage, newest first."""
    from sqlalchemy import text as _text

    from db.session import get_session

    with get_session() as session:
        rows = session.execute(_text("""
            select date(created_at) as day, stage, model,
                   count(*) as calls,
                   sum(input_tokens) as in_tok,
                   sum(output_tokens) as out_tok,
                   sum(cost_usd) as cost,
                   sum(searches) as searches,
                   sum(cached_input_tokens) as cached_in,
                   sum(tool_cost_usd) as tool_cost,
                   -- What was actually spent: tokens plus the searches the
                   -- old ledger could not see.
                   sum(coalesce(cost_usd, 0) + coalesce(tool_cost_usd, 0))
                       as total_cost,
                   count(*) filter (where cost_usd is null) as unpriced
            from llm_usage
            where created_at >= now() - (:days || ' days')::interval
            group by 1, 2, 3
            order by 1 desc, cost desc nulls last
        """), {"days": days}).mappings().all()
    return [dict(r) for r in rows]
