"""Read-side shaping for the launch screen and the performance page.

Everything here is derived from model_predictions. It holds no rendering, so
the same aggregate backs the Home scorecard and the Performance tables and the
two cannot drift apart.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import diskcache

from services.cache_service import get_cache

logger = logging.getLogger(__name__)

# Separate from the background-callback cache so a stale dashboard entry can
# never collide with Dash's own job bookkeeping.
_MEMO_DIR = Path("cache/dashboard")
_MEMO_TTL_S = 900
_memo: diskcache.Cache | None = None


def _cache_handle() -> "diskcache.Cache | None":
    global _memo
    if _memo is None:
        try:
            _MEMO_DIR.mkdir(parents=True, exist_ok=True)
            _memo = diskcache.Cache(str(_MEMO_DIR))
        except Exception as e:
            logger.warning("Dashboard memo cache unavailable: %s", e)
            return None
    return _memo


def _memo_key(name: str) -> str | None:
    """Key on who is asking, the newest data, and the pipeline generation.

    PIPELINE_EPOCH is already what gates prediction reuse upstream, so a data
    fix that changes what the models were fed also invalidates these reads.
    Returns None when there is nothing to key on, which disables caching
    rather than risking a wrong hit.
    """
    try:
        from services.analysis_runner import PIPELINE_EPOCH
    except Exception:
        PIPELINE_EPOCH = "unknown"
    try:
        from services.cache_service import _current_uid
        uid = _current_uid() or "anon"
    except Exception:
        uid = "anon"
    latest = get_cache().get_latest_prediction_date()
    if latest is None:
        return None
    return f"{name}|{uid}|{latest}|{PIPELINE_EPOCH}"


def _memoized(name: str, build):
    """Read-through cache around a pure query, best-effort.

    Any cache failure falls through to a live read: a dashboard that is slow
    is better than one that errors.
    """
    c = _cache_handle()
    if c is None:
        return build()
    key = _memo_key(name)
    if key is None:
        return build()
    try:
        hit = c.get(key)
        if hit is not None:
            return hit
    except Exception:
        return build()
    value = build()
    try:
        c.set(key, value, expire=_MEMO_TTL_S)
    except Exception:
        pass
    return value


def invalidate_memo() -> None:
    """Drop cached reads after predictions or evaluations change.

    Called from the callbacks that observe those two events, so a completed
    run shows on Home immediately instead of at the end of the TTL.
    """
    c = _cache_handle()
    if c is None:
        return
    try:
        c.clear()
    except Exception as e:
        logger.debug("Dashboard memo clear skipped: %s", e)

# Luna's per-symbol verdict is persisted as a prediction row and scored like a
# model. It IS counted in the scorecards and the P&L series: although it reads
# the other models' output, it makes its own call over reports + signals, a
# distinct prediction, exactly like the ensemble row, which was always counted.
# The name is still special-cased where the LAYOUT differs: the Home cohort
# board gives it its own slot instead of a column in the models grid.
SYNTHESIS_MODEL = "recommendation_synthesis"


def resolution_state(pred: dict) -> str:
    """One of 'resolved', 'held' or 'pending' for a single prediction.

    'held' survives only for legacy rows: the evaluator now scores a HOLD
    against the no-trade band (was_correct is set), so a modern HOLD is
    'resolved' like everything else. Rows scored before that rule change have
    was_correct None with a resolved price, and those still read as 'held'.
    """
    if pred.get("was_correct") is not None:
        return "resolved"
    if pred.get("pnl_dollars") is not None or pred.get("actual_close") is not None:
        return "held"
    return "pending"


def _cost_bps(prev_close) -> float:
    """Round-trip friction estimate in basis points, by price bucket.

    Price is the proxy we have on every prediction row; it correlates well
    enough with spread for an honesty adjustment (this is a haircut, not a
    microstructure model): sub-$5 names trade wide, $100+ names trade tight.
    """
    try:
        price = float(prev_close)
    except (TypeError, ValueError):
        return 20.0
    if price < 5:
        return 40.0
    if price < 20:
        return 25.0
    if price < 100:
        return 15.0
    return 8.0


def aggregate_predictions(preds: list[dict], group_key: str) -> list[dict]:
    """Group evaluated predictions by model or symbol into summary stats.

    Hit rate counts BUY/SELL only. HOLD takes no position, scores "correct" by
    default against the no-trade band, and would otherwise inflate the rate
    into meaninglessness -- so holds are reported as their own count instead.

    Beyond the raw aggregates, each group carries the three honesty stats the
    profit question actually turns on:
      * hit_se: binomial standard error of the hit rate; a 60% on 11 trades
        is a coin flip wearing a costume, and the SE is what says so.
      * net_pnl: pnl minus a per-trade friction haircut (see _cost_bps),
        because gross P&L on small caps flatters every edge.
      * concentration: the largest single symbol's share of gross |P&L|;
        an "edge" that is one ticker is a position, not a strategy.
    """
    groups: dict[str, dict] = {}
    for p in preds:
        g = groups.setdefault(
            p.get(group_key, "?"),
            {"scored": 0, "hits": 0, "trades": 0, "trade_hits": 0,
             "holds": 0, "pnl": 0.0, "conf": [], "cost": 0.0,
             "pnl_by_symbol": {}},
        )
        is_trade = (p.get("decision") or "HOLD").upper() != "HOLD"
        state = resolution_state(p)
        if state == "resolved":
            g["scored"] += 1
            g["hits"] += 1 if p["was_correct"] else 0
            if is_trade:
                g["trades"] += 1
                g["trade_hits"] += 1 if p["was_correct"] else 0
            else:
                # A scored HOLD still took no position. Count it as held so
                # the "N held" chip survives the HOLD-scoring rule change.
                g["holds"] += 1
        elif state == "held" and not is_trade:
            g["holds"] += 1
        if p.get("pnl_dollars") is not None:
            g["pnl"] += p["pnl_dollars"]
            sym = p.get("symbol", "?")
            g["pnl_by_symbol"][sym] = (g["pnl_by_symbol"].get(sym, 0.0)
                                       + p["pnl_dollars"])
            if is_trade:
                # $1,000 notional per trade x round-trip bps.
                g["cost"] += 1000.0 * _cost_bps(p.get("previous_close")) / 10000.0
        if p.get("confidence") is not None:
            g["conf"].append(p["confidence"])

    out = []
    for name in sorted(groups):
        g = groups[name]
        trades = g["trades"]
        hit_rate = (g["trade_hits"] / trades) if trades else None
        hit_se = ((hit_rate * (1 - hit_rate) / trades) ** 0.5
                  if hit_rate is not None and trades >= 2 else None)
        gross_abs = sum(abs(v) for v in g["pnl_by_symbol"].values())
        concentration = (max(abs(v) for v in g["pnl_by_symbol"].values())
                         / gross_abs if gross_abs > 0 else None)
        out.append({
            "name": name,
            "scored": g["scored"],
            "hits": g["hits"],
            "trades": trades,
            "trade_hits": g["trade_hits"],
            "holds": g["holds"],
            "pnl": g["pnl"],
            "net_pnl": g["pnl"] - g["cost"],
            "est_costs": g["cost"],
            "hit_rate": hit_rate,
            "hit_se": hit_se,
            "significant": (hit_rate is not None and hit_se is not None
                            and hit_se > 0 and abs(hit_rate - 0.5) > 2 * hit_se),
            "concentration": concentration,
            "pnl_per_trade": (g["pnl"] / trades) if trades else None,
            "avg_confidence": (sum(g["conf"]) / len(g["conf"])) if g["conf"] else None,
        })
    return out


def get_rolling_performance(days: int = 30, group_key: str = "model_name",
                            symbols: list[str] | None = None) -> list[dict]:
    """Trailing-window scorecard, newest `days` of prediction cutoffs."""
    if symbols is None:
        return _memoized(f"rolling|{days}|{group_key}",
                         lambda: _rolling_uncached(days, group_key, None))
    return _rolling_uncached(days, group_key, symbols)


def _rolling_uncached(days, group_key, symbols):
    end = datetime.now().date()
    start = end - timedelta(days=days)
    preds = get_cache().get_predictions_between(start, end, symbols=symbols)
    return aggregate_predictions(preds, group_key)


def get_latest_cohort() -> dict:
    """The latest cutoff's cohort (memoized)."""
    return get_cohort(None)


def get_cohort(prediction_date: str | None = None) -> dict:
    """Memoized wrapper: see _cohort_uncached. None means the latest cutoff."""
    key = f"cohort:{prediction_date or 'latest'}"
    return _memoized(key, lambda: _cohort_uncached(prediction_date))


def get_available_cutoffs(limit: int = 90) -> list[str]:
    """Distinct prediction cutoffs, newest first, as ISO strings."""
    return _memoized(f"cutoffs:{limit}",
                     lambda: [str(d) for d in
                              get_cache().get_prediction_dates(limit=limit)])


def _cohort_uncached(prediction_date: str | None = None) -> dict:
    """One prediction cutoff, shaped one row per symbol.

    Predictions carry no run id, so the cutoff date is the only grouping the
    schema guarantees. For the launch screen that is also the more truthful
    unit: it answers "what was the call on each name as of this cutoff",
    which survives a run being re-executed or topped up symbol by symbol.
    ``prediction_date=None`` means the most recent cutoff.
    """
    cache = get_cache()
    latest = prediction_date or cache.get_latest_prediction_date()
    if latest is None:
        return {"prediction_date": None, "symbols": [], "model_names": [],
                "counts": {"resolved": 0, "held": 0, "pending": 0},
                "pnl": 0.0, "target_date": None}

    preds = cache.get_predictions_between(latest, latest)

    by_symbol: dict[str, dict] = {}
    model_names: set[str] = set()
    counts = {"resolved": 0, "held": 0, "pending": 0}
    pnl = 0.0
    target_date = None

    for p in preds:
        state = resolution_state(p)
        counts[state] += 1
        if p.get("pnl_dollars") is not None:
            pnl += p["pnl_dollars"]
        target_date = target_date or p.get("target_date")

        row = by_symbol.setdefault(p["symbol"], {
            "symbol": p["symbol"],
            "previous_close": p.get("previous_close"),
            "target_date": p.get("target_date"),
            "models": {},
            "synthesis": None,
        })
        entry = {**p, "state": state}
        if p["model_name"] == SYNTHESIS_MODEL:
            row["synthesis"] = entry
        else:
            model_names.add(p["model_name"])
            row["models"][p["model_name"]] = entry
        if row["previous_close"] is None:
            row["previous_close"] = p.get("previous_close")

    return {
        "prediction_date": str(latest),
        "target_date": target_date,
        "symbols": [by_symbol[s] for s in sorted(by_symbol)],
        "model_names": sorted(model_names),
        "counts": counts,
        "pnl": pnl,
    }


def get_open_predictions(limit: int = 200) -> dict:
    """In-flight calls, grouped by the session they resolve on."""
    return _memoized(f"open|{limit}", lambda: _open_uncached(limit))


def _open_uncached(limit: int) -> dict:
    preds = get_cache().get_open_predictions(limit=limit)
    by_date: dict[str, list] = {}
    for p in preds:
        by_date.setdefault(p["target_date"], []).append(p)
    return {
        "total": len(preds),
        "dates": [{"target_date": d, "predictions": by_date[d],
                   "symbols": sorted({p["symbol"] for p in by_date[d]})}
                  for d in sorted(by_date)],
    }


def get_last_run() -> dict | None:
    """Envelope for the most recent pipeline run, or None if never run.

    The activity log is the only place a run exists as a first-class thing --
    predictions themselves carry no run id -- so title, duration and error
    count come from there while the content comes from the cohort.

    Events emitted outside any run land in an 'adhoc' bucket whose span covers
    every such event ever recorded, so it sorts to the top and reports a
    multi-day duration. It is not a run and must not be shown as one.
    """
    try:
        from services import progress_service
        runs = progress_service.get_activity_runs(limit_runs=25)
    except Exception as e:
        logger.warning("Could not read activity runs: %s", e)
        return None

    real = [r for r in runs if r.get("run_id") and r["run_id"] != "adhoc"]
    if not real:
        return None
    return max(real, key=lambda r: r.get("started") or datetime.min)
