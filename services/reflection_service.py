"""Deferred, alpha-grounded reflection loop.

Ported in spirit from TradingAgents' memory-log + reflector (the standout idea in
the framework), reimplemented natively so it depends only on our own price data.

The problem it solves: you cannot reflect on an outcome you do not have yet. So:

  1. record_pending() — when the agent makes a call, append a *pending* entry.
  2. resolve_pending() — on a later run, for any pending entry whose hold window
     has elapsed and whose realized price is available, compute the raw return and
     the alpha vs. SPY, ask the LLM for a 2-4 sentence lesson, and mark it resolved.
  3. get_past_context() — inject the most recent same-ticker decisions (full) plus
     a few recent cross-ticker lessons (reflection-only) into the next prompt.

The learning signal is realized alpha, not self-grading, and the loop is naturally
lookahead-safe: at as-of date T it only resolves decisions whose outcome date is
already <= T. Storage is an append-only JSONL file with atomic rewrites — human
readable, greppable, crash-safe, and swappable for a table later.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_HOLD_DAYS = 5
_N_SAME = 5   # recent same-ticker decisions to inject (full)
_N_CROSS = 3  # recent cross-ticker lessons to inject (reflection only)


def _store_path() -> str:
    path = os.environ.get("REFLECTION_LOG_PATH")
    if path:
        return path
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "reflections.jsonl")


def _load_all() -> list[dict]:
    path = _store_path()
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _atomic_write_all(entries: list[dict]) -> None:
    path = _store_path()
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _append(entry: dict) -> None:
    with open(_store_path(), "a") as f:
        f.write(json.dumps(entry) + "\n")


def record_pending(
    symbol: str,
    as_of: str,
    decision: str,
    confidence: float,
    thesis: str = "",
    hold_days: int = DEFAULT_HOLD_DAYS,
) -> None:
    """Append a pending decision. Idempotent per (symbol, as_of)."""
    symbol = symbol.upper()
    for e in _load_all():
        if e.get("symbol") == symbol and e.get("as_of") == as_of:
            return  # already recorded (pending or resolved)
    _append({
        "symbol": symbol,
        "as_of": as_of,
        "decision": (decision or "HOLD").upper(),
        "confidence": round(float(confidence or 0.0), 3),
        "thesis": (thesis or "")[:800],
        "hold_days": hold_days,
        "status": "pending",
        "raw_return": None,
        "alpha_return": None,
        "reflection": None,
        "resolved_date": None,
    })


def realized_alpha(
    symbol: str, as_of: str, hold_days: int = DEFAULT_HOLD_DAYS,
    benchmark: str = "SPY",
) -> tuple[Optional[float], Optional[float], Optional[int]]:
    """Return (raw_return, alpha_vs_benchmark, actual_hold_days) or (None, None, None).

    Uses our own OHLCV. Buys at the close of as_of and holds `hold_days` trading
    sessions (the first available bar at/after as_of + hold_days). Returns None
    when the future bar is not yet available (entry stays pending, retries later).
    """
    try:
        from services.stock_data import fetch_stock_data
        import pandas as pd

        def _ret(sym: str):
            df = fetch_stock_data(sym, period="1y")
            df = df[pd.to_datetime(df.index) >= pd.to_datetime(as_of) - timedelta(days=10)]
            at = df[pd.to_datetime(df.index) <= pd.to_datetime(as_of)]
            if at.empty:
                return None, None
            entry = float(at["Close"].iloc[-1])
            target_dt = pd.to_datetime(as_of) + timedelta(days=hold_days)
            after = df[pd.to_datetime(df.index) >= target_dt]
            if after.empty:
                return None, None  # future bar not available yet
            exit_px = float(after["Close"].iloc[0])
            days = (pd.to_datetime(after.index[0]) - pd.to_datetime(at.index[-1])).days
            return (exit_px - entry) / entry, days

        raw, days = _ret(symbol)
        if raw is None:
            return None, None, None
        bench_raw, _ = _ret(benchmark)
        alpha = raw - bench_raw if bench_raw is not None else None
        return raw, alpha, days
    except Exception as e:
        logger.debug(f"realized_alpha failed for {symbol}@{as_of}: {e}")
        return None, None, None


REFLECTION_SYSTEM = (
    "You are a trading post-mortem analyst. You are given a past DECISION "
    "(BUY, SELL, or HOLD), the realized forward return, and the alpha vs SPY. "
    "Correctness rule: a BUY is correct iff the return was POSITIVE; a SELL is "
    "correct iff the return was NEGATIVE; a HOLD took no position. Do NOT infer "
    "the direction from the sign of the return — use the stated decision. Write a "
    "SHORT lesson (2-4 sentences): (1) was the call correct given the rule above — "
    "cite the alpha figure; (2) which part of the thesis held or failed; (3) one "
    "concrete, reusable lesson for next time. No preamble."
)


def _default_reflect(decision: str, thesis: str, raw: float, alpha: float) -> str:
    """LLM reflection via our own llm_service (used when no llm is injected).

    `decision` is stated explicitly so the reflector does not misread a wrong
    SELL that happened to precede a price rise as a "correct long."
    """
    try:
        from services.llm_service import get_llm
        correct = (decision == "BUY" and raw > 0) or (decision == "SELL" and raw < 0)
        verdict = "CORRECT" if correct else "WRONG"
        prompt = (
            f"DECISION: {decision}\n"
            f"Realized forward return: {raw:+.1%} (so this {decision} was {verdict})\n"
            f"Alpha vs SPY: {alpha:+.1%}\n\n"
            f"Thesis at the time:\n{(thesis or '')[:2000]}"
        )
        out = get_llm().generate(
            prompt, system_prompt=REFLECTION_SYSTEM, max_tokens=250, temperature=0.3
        )
        return (out or "").strip()
    except Exception as e:
        logger.debug(f"reflection LLM failed: {e}")
        return ""


def resolve_pending(
    symbol: str,
    as_of: str,
    reflect_fn: Optional[Callable[[str, str, float, float], str]] = None,
    benchmark: str = "SPY",
) -> int:
    """Resolve pending entries for `symbol` whose outcome is known by `as_of`.

    Returns the number resolved. `reflect_fn(decision, thesis, raw, alpha) -> str`
    is injectable for testing; defaults to an LLM call.
    """
    symbol = symbol.upper()
    reflect_fn = reflect_fn or _default_reflect
    entries = _load_all()
    changed = 0
    now_dt = datetime.strptime(as_of, "%Y-%m-%d")
    for e in entries:
        if e.get("symbol") != symbol or e.get("status") != "pending":
            continue
        entry_dt = datetime.strptime(e["as_of"], "%Y-%m-%d")
        hold = e.get("hold_days", DEFAULT_HOLD_DAYS)
        if entry_dt + timedelta(days=hold) > now_dt:
            continue  # outcome not yet due as of `as_of`
        raw, alpha, days = realized_alpha(symbol, e["as_of"], hold, benchmark)
        if raw is None or alpha is None:
            continue  # price not available yet — retry on a later run
        e["raw_return"] = round(raw, 4)
        e["alpha_return"] = round(alpha, 4)
        e["reflection"] = reflect_fn(e["decision"], e.get("thesis", ""), raw, alpha)
        e["resolved_date"] = as_of
        e["status"] = "resolved"
        changed += 1
    if changed:
        _atomic_write_all(entries)
    return changed


def empirical_edge(
    symbol: Optional[str] = None, min_n: int = 8,
) -> tuple[Optional[float], int]:
    """Realized directional hit-rate over resolved *active* (BUY/SELL) decisions.

    This is the grounded alternative to an LLM's self-reported confidence: the
    model earns weight by being right over time, not by claiming certainty. A
    BUY is correct iff the realized return was positive; a SELL iff negative.
    Returns (hit_rate, n). hit_rate is None until at least `min_n` resolved
    active decisions exist (honest "no track record yet" -> caller uses neutral).
    """
    active = []
    for e in _load_all():
        if e.get("status") != "resolved" or e.get("raw_return") is None:
            continue
        if symbol and e.get("symbol") != symbol.upper():
            continue
        dec = e.get("decision")
        if dec not in ("BUY", "SELL"):
            continue
        active.append(e)
    n = len(active)
    if n < min_n:
        return None, n
    correct = sum(
        1 for e in active
        if (e["decision"] == "BUY" and e["raw_return"] > 0)
        or (e["decision"] == "SELL" and e["raw_return"] < 0)
    )
    return correct / n, n


def get_past_context(symbol: str, n_same: int = _N_SAME, n_cross: int = _N_CROSS) -> str:
    """Build the lessons block injected into the next prompt.

    Most-recent same-ticker resolved decisions (with reflection) + a few recent
    cross-ticker lessons (reflection only). Empty string if nothing resolved yet.
    """
    symbol = symbol.upper()
    resolved = [e for e in _load_all() if e.get("status") == "resolved" and e.get("reflection")]
    resolved.sort(key=lambda e: e.get("resolved_date") or "", reverse=True)

    same, cross = [], []
    for e in resolved:
        if e["symbol"] == symbol and len(same) < n_same:
            same.append(e)
        elif e["symbol"] != symbol and len(cross) < n_cross:
            cross.append(e)

    if not same and not cross:
        return ""

    parts = ["[PAST DECISIONS & OUTCOMES — learn from these, do not repeat mistakes]"]
    if same:
        parts.append(f"Prior {symbol} calls (most recent first):")
        for e in same:
            parts.append(
                f"- {e['as_of']}: {e['decision']} (conf {e['confidence']:.0%}) -> "
                f"raw {e['raw_return']:+.1%}, alpha {e['alpha_return']:+.1%}. "
                f"Lesson: {e['reflection']}"
            )
    if cross:
        parts.append("Recent cross-ticker lessons:")
        for e in cross:
            parts.append(f"- {e['symbol']} {e['as_of']}: {e['reflection']}")
    return "\n".join(parts)
