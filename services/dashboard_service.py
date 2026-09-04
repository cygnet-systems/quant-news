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


def _memoized(name: str, build, cacheable=None):
    """Read-through cache around a pure query, best-effort.

    Any cache failure falls through to a live read: a dashboard that is slow
    is better than one that errors. ``cacheable(value)`` may veto storing a
    result that is still changing (a run in flight) so the next read sees
    the live rows instead of the snapshot.
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
    if cacheable is not None and not cacheable(value):
        return value
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

# Run statuses whose artifacts are settled enough to cache. A cancelled run
# is not one of them (see get_run_view).
MEMOIZED_RUN_STATUSES = ("done", "failed")


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


def collapse_rerun_duplicates(preds: list[dict]) -> tuple[list[dict], int]:
    """One row per (symbol, model, cutoff): the newest scored write wins.

    An ad-hoc run of a name the daily job already called writes a SECOND row
    for that call: manual predictions live in their own id space so the
    rerun cannot destroy the scheduled row or its evaluation (see
    cache_service._prediction_id). Each of those rows is then evaluated on
    its own, so aggregating them all reports three reruns of one TSLA call
    as three trades and three times its P&L. The scoreboard is what this
    project judges its models by, so it counts the call once.

    Returns the kept rows in their original order plus how many were
    dropped, because a silent collapse is its own kind of lie: the page
    says how many rows it folded.
    """
    def rank(p):
        # A scored row outranks an unscored one before recency does: a rerun
        # started this morning is not yet evaluated, and letting it win would
        # delete a resolved outcome from the scoreboard rather than
        # deduplicate it. Then newest, by created_at (an ISO timestamp, so
        # string order is time order), with the id breaking a same-tick tie.
        scored = (p.get("was_correct") is not None
                  or p.get("pnl_dollars") is not None)
        return (scored, p.get("created_at") or "", p.get("id") or "")

    def call_key(p):
        return (p.get("symbol"), p.get("model_name"), p.get("prediction_date"))

    newest: dict[tuple, dict] = {}
    for p in preds:
        held = newest.get(call_key(p))
        if held is None or rank(p) >= rank(held):
            newest[call_key(p)] = p
    kept = [p for p in preds if newest[call_key(p)] is p]
    return kept, len(preds) - len(kept)


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


def _symbols_key(symbols) -> str:
    """Memo-key fragment for a symbol filter: 'any' for no filter, else the
    sorted upper-cased names so two spellings of one watchlist share a hit."""
    if symbols is None:
        return "any"
    return ",".join(sorted({str(s).upper() for s in symbols})) or "none"


def get_rolling_performance(days: int = 30, group_key: str = "model_name",
                            symbols: list[str] | None = None,
                            kind: str | None = None) -> list[dict]:
    """Trailing-window scorecard, newest `days` of prediction cutoffs.

    ``kind='scheduled'`` is what the Home strip asks for, so an ad-hoc
    experiment never moves the number the user tracks day to day.
    """
    if symbols is None:
        return _memoized(f"rolling|{days}|{group_key}|{kind or 'any'}",
                         lambda: _rolling_uncached(days, group_key, None, kind))
    return _rolling_uncached(days, group_key, symbols, kind)


def _rolling_uncached(days, group_key, symbols, kind=None):
    end = datetime.now().date()
    start = end - timedelta(days=days)
    preds = get_cache().get_predictions_between(start, end, symbols=symbols,
                                                kind=kind)
    return aggregate_predictions(preds, group_key)


def get_latest_cohort(kind: str | None = "scheduled",
                      symbols: list[str] | None = None) -> dict:
    """The latest cutoff's cohort (memoized): the Scheduled tab's source.

    With the default kind the cutoff is the newest one a scheduled run
    wrote, so an ad-hoc run landing later in the day never displaces the
    board; pass the watchlist as ``symbols`` to keep it to those names.
    """
    return get_cohort(None, kind=kind, symbols=symbols)


def get_cohort(prediction_date: str | None = None, kind: str | None = None,
               symbols: list[str] | None = None) -> dict:
    """Memoized wrapper: see _cohort_uncached. None means the latest cutoff
    (of ``kind`` when one is given)."""
    key = (f"cohort:{prediction_date or 'latest'}|{kind or 'any'}"
           f"|{_symbols_key(symbols)}")
    return _memoized(key, lambda: _cohort_uncached(prediction_date, kind, symbols))


def get_available_cutoffs(limit: int = 90,
                          kind: str | None = "scheduled") -> list[str]:
    """Distinct prediction cutoffs, newest first, as ISO strings.

    Defaults to the scheduled runs' cutoffs: the Home dropdown pairs with
    the Scheduled tab, and a manual run's cutoff belongs to This session.
    """
    return _memoized(f"cutoffs:{limit}|{kind or 'any'}",
                     lambda: [str(d) for d in
                              get_cache().get_prediction_dates(limit=limit,
                                                               kind=kind)])


def _new_row(symbol: str, previous_close=None, target_date=None) -> dict:
    return {
        "symbol": symbol,
        "previous_close": previous_close,
        "target_date": target_date,
        "models": {},
        "synthesis": None,
    }


def shape_symbol_rows(preds: list[dict]) -> dict:
    """Fold prediction rows into one row per symbol.

    The one shaping both the Home board and a run page use, so a model chip
    or an outcome cell reads the same on either. Each prediction gains its
    resolution ``state``; the synthesis verdict takes the row's own slot
    instead of a model column. Returns ``by_symbol`` (insertion order of
    first sight), the sorted ``model_names`` seen, the resolution ``counts``,
    the summed ``pnl`` and the first ``target_date``.
    """
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

        row = by_symbol.setdefault(p["symbol"], _new_row(
            p["symbol"], p.get("previous_close"), p.get("target_date")))
        entry = {**p, "state": state}
        if p["model_name"] == SYNTHESIS_MODEL:
            row["synthesis"] = entry
        else:
            model_names.add(p["model_name"])
            row["models"][p["model_name"]] = entry
        if row["previous_close"] is None:
            row["previous_close"] = p.get("previous_close")

    return {
        "by_symbol": by_symbol,
        "model_names": sorted(model_names),
        "counts": counts,
        "pnl": pnl,
        "target_date": target_date,
    }


def _cohort_uncached(prediction_date: str | None = None,
                     kind: str | None = None,
                     symbols: list[str] | None = None) -> dict:
    """One prediction cutoff, shaped one row per symbol.

    The cutoff date is the grouping the launch screen is built on: it
    answers "what was the call on each name as of this cutoff", which
    survives a run being re-executed or topped up symbol by symbol.
    ``prediction_date=None`` means the most recent cutoff. ``kind`` keeps
    the board to what scheduled (or manual) runs wrote, the newest cutoff
    included, so a manual run on a watchlist name never enters the
    Scheduled tab; ``symbols`` narrows the rows to those names.
    """
    cache = get_cache()
    latest = prediction_date or cache.get_latest_prediction_date(kind=kind)
    if latest is None:
        return {"prediction_date": None, "symbols": [], "model_names": [],
                "counts": {"resolved": 0, "held": 0, "pending": 0},
                "pnl": 0.0, "target_date": None}

    shaped = shape_symbol_rows(cache.get_predictions_between(
        latest, latest, symbols=symbols, kind=kind))
    by_symbol = shaped["by_symbol"]
    return {
        "prediction_date": str(latest),
        "target_date": shaped["target_date"],
        "symbols": [by_symbol[s] for s in sorted(by_symbol)],
        "model_names": shaped["model_names"],
        "counts": shaped["counts"],
        "pnl": shaped["pnl"],
    }


def _report_headline(report: dict) -> dict:
    return {
        "id": report.get("id"),
        "decision": report.get("decision"),
        "confidence": report.get("confidence"),
        "trade_date": report.get("trade_date"),
        "model_name": report.get("model_name"),
    }


def _run_rows(run: dict, shaped: dict, target_date, extra=()) -> list[dict]:
    """The run's rows in its own symbol order (the dialog's chips or the
    watchlist), then any symbol in ``shaped`` or ``extra`` the row does not
    name. A configured symbol nothing was written for still gets a row, so
    a cancelled or in-flight run lists what it was asked for."""
    by_symbol = shaped["by_symbol"]
    ordered = list(run.get("symbols") or [])
    for sym in list(by_symbol) + list(extra):
        if sym not in ordered:
            ordered.append(sym)
    return [by_symbol.get(sym) or _new_row(sym, target_date=target_date)
            for sym in ordered]


def _run_artifacts_uncached(run: dict) -> dict:
    """Everything a run wrote, one indexed query per table.

    Rows follow the run's own symbol order (the dialog's chips or the
    watchlist), then any symbol an artifact names that the row does not,
    so a cancelled or still-running run lists every symbol it was asked
    for with empty cells rather than only the ones that finished.
    """
    cache = get_cache()
    run_id = run["run_id"]
    shaped = shape_symbol_rows(cache.get_predictions_for_run(run_id))

    reports_by_symbol: dict[str, dict] = {}
    for r in cache.get_trading_agent_reports_for_run(run_id):
        # Newest first from the query, so the first hit per symbol wins.
        reports_by_symbol.setdefault(r["symbol"], _report_headline(r))

    target_date = shaped["target_date"] or run.get("target_date")
    rows = _run_rows(run, shaped, target_date, extra=sorted(reports_by_symbol))
    for row in rows:
        row["report"] = reports_by_symbol.get(row["symbol"])

    rec = cache.get_recommendation_for_run(run_id)
    recommendation = None
    if rec is not None:
        recommendation = {
            "result_json": rec.get("result_json") or {},
            "model_used": rec.get("model_used"),
            "duration_ms": rec.get("duration_ms"),
            "created_at": rec.get("created_at"),
        }

    return {
        "symbols": rows,
        "model_names": shaped["model_names"],
        "counts": shaped["counts"],
        "pnl": shaped["pnl"],
        "target_date": target_date,
        "recommendation": recommendation,
    }


def get_run_view(run_id: str) -> dict | None:
    """The run page's data: the run row plus everything it wrote.

    None when there is no such run. The artifact reads are memoized only
    once the run has finished or failed; a run in flight is re-read on
    every visit so rows fill in as stages land. Cancelled is deliberately
    not memoized: cancelling closes the row but the in-process research
    stage is not killable, so it can still write a report afterwards, and
    a memo taken at cancel time would hide it. The row itself is always
    read live (it is one PK lookup) so the header never shows a stale
    status.
    """
    from services import run_service

    run = run_service.get_run(run_id) if run_id else None
    if run is None:
        return None
    if run["status"] in MEMOIZED_RUN_STATUSES:
        artifacts = _memoized(f"run:{run_id}",
                              lambda: _run_artifacts_uncached(run))
    else:
        artifacts = _run_artifacts_uncached(run)
    return {"run": run, **artifacts}


def current_session_date() -> str:
    """The prediction cutoff a run started right now would carry.

    The same resolution the Run dialog does (resolve_target_and_cutoff with
    no pick), so "this session" means the session the next ad-hoc run would
    predict from, not whatever the newest manual run happens to be. It
    rolls forward at the close, with the session it names.
    """
    from utils.trading_calendar import resolve_target_and_cutoff

    return resolve_target_and_cutoff(None)[1].isoformat()


def get_session_runs(prediction_date: str | None = None,
                     limit: int = 20) -> list[dict]:
    """This session's ad-hoc runs, newest first, each with its own rows.

    A "session" is one prediction cutoff: the current one (see
    current_session_date) or ``prediction_date``. Every manual run at that
    cutoff is listed, queued and cancelled ones included, so the tab is the
    answer to "what did I run today" rather than "what finished", and a day
    with no ad-hoc run is empty rather than showing last month's. Each entry
    is ``{run, symbols, model_names, counts, pnl, target_date}`` with the rows
    shaped exactly like a cohort's (see shape_symbol_rows), in the run's
    symbol order. Memoized like the cohorts, except that a list holding a
    run still in flight is read live so its rows fill in.

    The key carries the runs table's own generation because the shared memo
    key cannot see a run row: without it a run started after the last Home
    render would be missing from this tab for the whole time it is in
    flight, and a report-only run (which writes no prediction) for the
    whole TTL after it finished.
    """
    from services import run_service

    session_date = str(prediction_date or current_session_date())[:10]
    key = (f"session_runs:{session_date}|{limit}"
           f"|{run_service.runs_generation('manual')}")
    return _memoized(key, lambda: _session_runs_uncached(session_date, limit),
                     cacheable=lambda runs: not any(r["run"]["active"]
                                                    for r in runs))


def _session_runs_uncached(session_date, limit) -> list[dict]:
    from services import run_service

    # Newest first by start; the row limit is generous because one day's
    # ad-hoc runs are a handful and the cutoff filter below does the rest.
    runs = run_service.list_runs(limit=max(limit * 10, 100), kind="manual")
    cache = get_cache()
    out = []
    for run in runs:
        if run.get("prediction_date") != session_date:
            continue
        shaped = shape_symbol_rows(cache.get_predictions_for_run(run["run_id"]))
        target_date = shaped["target_date"] or run.get("target_date")
        out.append({
            "run": run,
            "symbols": _run_rows(run, shaped, target_date),
            "model_names": shaped["model_names"],
            "counts": shaped["counts"],
            "pnl": shaped["pnl"],
            "target_date": target_date,
        })
        if len(out) >= limit:
            break
    return out


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
