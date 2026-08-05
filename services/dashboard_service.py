"""Read-side shaping for the launch screen and the performance page.

Everything here is derived from model_predictions. It holds no rendering, so
the same aggregate backs the Home scorecard and the Performance tables and the
two cannot drift apart.
"""

import logging
from datetime import date, datetime, timedelta

from services.cache_service import get_cache

logger = logging.getLogger(__name__)

# Luna's per-symbol verdict is persisted as a prediction row so it gets scored
# like a model, but it is a synthesis over the others rather than a peer.
# Counting it as "a model that ran" double-counts the same evidence.
SYNTHESIS_MODEL = "recommendation_synthesis"


def resolution_state(pred: dict) -> str:
    """One of 'resolved', 'held' or 'pending' for a single prediction.

    was_correct alone is not enough. The evaluator leaves it None for a HOLD
    (a HOLD cannot be right or wrong in the directional sense) while still
    setting pnl_dollars to 0.0, so keying purely on was_correct reports an
    evaluated HOLD as if it were still awaiting its close.
    """
    if pred.get("was_correct") is not None:
        return "resolved"
    if pred.get("pnl_dollars") is not None or pred.get("actual_close") is not None:
        return "held"
    return "pending"


def aggregate_predictions(preds: list[dict], group_key: str) -> list[dict]:
    """Group evaluated predictions by model or symbol into summary stats.

    Hit rate counts BUY/SELL only. HOLD takes no position, scores "correct" by
    default against the no-trade band, and would otherwise inflate the rate
    into meaninglessness -- so holds are reported as their own count instead.
    """
    groups: dict[str, dict] = {}
    for p in preds:
        g = groups.setdefault(
            p.get(group_key, "?"),
            {"scored": 0, "hits": 0, "trades": 0, "trade_hits": 0,
             "holds": 0, "pnl": 0.0, "conf": []},
        )
        is_trade = (p.get("decision") or "HOLD").upper() != "HOLD"
        state = resolution_state(p)
        if state == "resolved":
            g["scored"] += 1
            g["hits"] += 1 if p["was_correct"] else 0
            if is_trade:
                g["trades"] += 1
                g["trade_hits"] += 1 if p["was_correct"] else 0
        elif state == "held" and not is_trade:
            g["holds"] += 1
        if p.get("pnl_dollars") is not None:
            g["pnl"] += p["pnl_dollars"]
        if p.get("confidence") is not None:
            g["conf"].append(p["confidence"])

    out = []
    for name in sorted(groups):
        g = groups[name]
        trades = g["trades"]
        out.append({
            "name": name,
            "scored": g["scored"],
            "hits": g["hits"],
            "trades": trades,
            "trade_hits": g["trade_hits"],
            "holds": g["holds"],
            "pnl": g["pnl"],
            "hit_rate": (g["trade_hits"] / trades) if trades else None,
            "pnl_per_trade": (g["pnl"] / trades) if trades else None,
            "avg_confidence": (sum(g["conf"]) / len(g["conf"])) if g["conf"] else None,
        })
    return out


def get_rolling_performance(days: int = 30, group_key: str = "model_name",
                            symbols: list[str] | None = None) -> list[dict]:
    """Trailing-window scorecard, newest `days` of prediction cutoffs."""
    end = datetime.now().date()
    start = end - timedelta(days=days)
    preds = get_cache().get_predictions_between(start, end, symbols=symbols)
    return aggregate_predictions(
        [p for p in preds if p["model_name"] != SYNTHESIS_MODEL], group_key)


def get_pnl_series(days: int = 30, symbols: list[str] | None = None) -> list[dict]:
    """Cumulative P&L by prediction date, for the Home sparkline."""
    end = datetime.now().date()
    start = end - timedelta(days=days)
    preds = get_cache().get_predictions_between(start, end, symbols=symbols)

    daily: dict[str, float] = {}
    for p in preds:
        if p["model_name"] == SYNTHESIS_MODEL or p.get("pnl_dollars") is None:
            continue
        daily[p["prediction_date"]] = daily.get(p["prediction_date"], 0.0) + p["pnl_dollars"]

    running = 0.0
    series = []
    for d in sorted(daily):
        running += daily[d]
        series.append({"date": d, "daily": daily[d], "cumulative": running})
    return series


def get_latest_cohort() -> dict:
    """The most recent prediction cutoff, shaped one row per symbol.

    Predictions carry no run id, so the cutoff date is the only grouping the
    schema guarantees. For the launch screen that is also the more truthful
    unit: it answers "what is the current call on each name", which survives a
    run being re-executed or topped up symbol by symbol.
    """
    cache = get_cache()
    latest = cache.get_latest_prediction_date()
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
    """
    try:
        from services import progress_service
        runs = progress_service.get_activity_runs(limit_runs=1)
    except Exception as e:
        logger.warning("Could not read activity runs: %s", e)
        return None
    return runs[0] if runs else None
