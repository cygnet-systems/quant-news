"""Per-model confidence calibration from evaluated prediction history.

Raw model confidences are demonstrably anti-calibrated on this platform
(claimed 71% → 42.5% actual across the live era). Everything that consumes
a confidence: UI badges, ensemble weights, the scoreboard, should go
through :func:`calibrate` instead of trusting the raw number.

The mapping is an isotonic regression fit per model on (raw_confidence,
was_correct) pairs from evaluated ACTIVE calls (HOLDs carry no direction to
score). Isotonic is the right shape here: it is monotone, so a model whose
confidence is directionally meaningful keeps its ordering, and a model whose
confidence is pure noise flattens toward its base rate, which is itself the
honest answer.

Fits are cached in-process for CALIBRATION_TTL_S and refit lazily; the data
changes at most once per evaluation run, so staleness is bounded and cheap.

Every read here is scheduled-only. An ad-hoc rerun of a call the daily job
already made stores its own prediction row (cache_service._prediction_id),
so counting all rows would weight a re-run name once per rerun and, worse,
would quote an inflated sample back into a report through
:func:`track_record_sentence`. The track record is the daily job.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# A fit below this many evaluated active calls is an anecdote, not a mapping.
MIN_SAMPLES = 30
CALIBRATION_TTL_S = 15 * 60


@dataclass
class _ModelFit:
    n: int = 0
    # Step function: sorted list of (raw_threshold, calibrated_value).
    steps: list[tuple[float, float]] = field(default_factory=list)
    base_rate: Optional[float] = None


_lock = threading.Lock()
_fits: dict[str, _ModelFit] = {}
_fitted_at: float = 0.0


def _scheduled_calls(*columns):
    """Select over the scheduled daily job's scored ACTIVE calls.

    Core rather than text() so the LEFT JOIN and the "no run row means
    scheduled" COALESCE (cache_service._by_run_kind is the same rule) are
    written once, and so the date bounds bind as dates on any dialect.
    """
    from db.models import AnalysisRun, ModelPrediction
    from sqlalchemy import func, select

    return (select(*columns)
            .select_from(ModelPrediction)
            .outerjoin(AnalysisRun,
                       AnalysisRun.run_id == ModelPrediction.run_id)
            .where(func.coalesce(AnalysisRun.kind, "scheduled") == "scheduled",
                   ModelPrediction.was_correct.isnot(None),
                   ModelPrediction.decision != "HOLD"))


def _hit_avg():
    """Share of scored calls that were directionally right, as SQL."""
    from db.models import ModelPrediction
    from sqlalchemy import case, func

    return func.avg(case((ModelPrediction.was_correct.is_(True), 1.0),
                         else_=0.0))


def _as_of_date(as_of: Optional[str]) -> "date":
    """Upper bound of a window: the as_of session, else today.

    The process clock rather than the database's CURRENT_DATE, which is what
    every other window in the codebase already uses (dashboard_service's
    rolling window, get_open_predictions); app and database run in the same
    timezone, and binding a real date keeps the query dialect-neutral.
    """
    return date.fromisoformat(str(as_of)[:10]) if as_of else date.today()


def _pav(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pool-adjacent-violators on (raw_conf, outcome) pairs, ascending by raw.

    Returns the isotonic step function as (raw, calibrated) knots. Implemented
    directly: scikit-learn is not a dependency of this project.
    """
    pairs = sorted(pairs, key=lambda p: p[0])
    # Each block: [sum_y, count, min_x]
    blocks: list[list[float]] = []
    for x, y in pairs:
        blocks.append([y, 1, x])
        while len(blocks) > 1 and (blocks[-2][0] / blocks[-2][1]
                                   > blocks[-1][0] / blocks[-1][1]):
            y2, n2, x2 = blocks.pop()
            blocks[-1][0] += y2
            blocks[-1][1] += n2
    return [(b[2], b[0] / b[1]) for b in blocks]


def _fit_all(as_of: Optional[str] = None) -> dict[str, _ModelFit]:
    """Isotonic fits per model. ``as_of`` bounds the outcomes used (by
    target_date, the session whose close resolved the call) so a backtest
    does not calibrate on sessions it has not reached."""
    from db.models import ModelPrediction
    from db.session import get_session

    q = _scheduled_calls(ModelPrediction.model_name, ModelPrediction.confidence,
                         ModelPrediction.was_correct).where(
                             ModelPrediction.confidence.isnot(None))
    if as_of:
        q = q.where(ModelPrediction.target_date <= _as_of_date(as_of))
    fits: dict[str, _ModelFit] = {}
    with get_session() as session:
        rows = session.execute(q).all()

    by_model: dict[str, list[tuple[float, float]]] = {}
    for model, conf, correct in rows:
        by_model.setdefault(model, []).append(
            (float(conf), 1.0 if correct else 0.0))

    for model, pairs in by_model.items():
        fit = _ModelFit(n=len(pairs))
        if len(pairs) >= MIN_SAMPLES:
            fit.steps = _pav(pairs)
            fit.base_rate = sum(y for _, y in pairs) / len(pairs)
        fits[model] = fit
    return fits


# Historical fits are immutable, so one per as_of is kept for the process.
_fits_as_of: dict[str, dict[str, _ModelFit]] = {}


def _fits_for(as_of: Optional[str]) -> dict[str, _ModelFit]:
    if not as_of:
        _ensure_fresh()
        return _fits
    key = str(as_of)[:10]
    if key not in _fits_as_of:
        with _lock:
            if key not in _fits_as_of:
                try:
                    _fits_as_of[key] = _fit_all(key)
                except Exception as e:
                    logger.warning(f"Calibration fit as of {key} failed: {e}")
                    _fits_as_of[key] = {}
    return _fits_as_of[key]


def _ensure_fresh() -> None:
    global _fits, _fitted_at
    if time.monotonic() - _fitted_at < CALIBRATION_TTL_S and _fits:
        return
    with _lock:
        if time.monotonic() - _fitted_at < CALIBRATION_TTL_S and _fits:
            return
        try:
            _fits = _fit_all()
            _fitted_at = time.monotonic()
            sized = {m: f.n for m, f in _fits.items()}
            logger.info(f"Calibration refit: {sized}")
        except Exception as e:
            # Keep serving the previous fit; a DB hiccup must not take
            # calibration down with it.
            logger.warning(f"Calibration refit failed: {e}")
            _fitted_at = time.monotonic()


def calibrate(model_name: str, raw_confidence: Optional[float],
              as_of: Optional[str] = None) -> Optional[float]:
    """Map a raw confidence to its historical hit rate for this model.

    Returns None when there is not enough evaluated history to say anything
    (< MIN_SAMPLES active evaluated calls). Callers should then show no
    number rather than an unearned one. ``as_of`` (backtests) restricts the
    history to outcomes resolved by that date.
    """
    if raw_confidence is None:
        return None
    fit = _fits_for(as_of).get(model_name)
    if fit is None or not fit.steps:
        return None
    value = fit.steps[0][1]
    for threshold, calibrated in fit.steps:
        if raw_confidence >= threshold:
            value = calibrated
        else:
            break
    return round(value, 3)


_hit_cache: dict[tuple[str, int], tuple[float, Optional[float]]] = {}


def rolling_hit_rate(model_name: str, days: int = 30,
                     as_of: Optional[str] = None) -> Optional[float]:
    """Decay-free rolling hit rate on active calls over the last `days`.

    Used for performance-decay ensemble weighting. None below MIN_SAMPLES/3.
    a weekly-scale window needs a lower floor than the full calibration fit,
    but still refuses single-digit sample sizes. TTL-cached: a 20-symbol
    batch would otherwise repeat the same query per member per symbol.
    """
    from db.models import ModelPrediction
    from db.session import get_session
    from sqlalchemy import func

    key = (model_name, days, str(as_of)[:10] if as_of else None)
    cached = _hit_cache.get(key)
    if cached and time.monotonic() - cached[0] < CALIBRATION_TTL_S:
        return cached[1]

    # Window ends at as_of (outcomes resolved by then) instead of today, so
    # a backtest's ensemble weights cannot see the future.
    anchor = _as_of_date(as_of)
    q = (_scheduled_calls(func.count(), _hit_avg())
         .where(ModelPrediction.model_name == model_name,
                ModelPrediction.target_date <= anchor,
                ModelPrediction.prediction_date >= anchor - timedelta(days=days)))
    with get_session() as session:
        row = session.execute(q).one()
    n, rate = (row[0] or 0), row[1]
    result = (None if n < max(10, MIN_SAMPLES // 3) or rate is None
              else round(float(rate), 3))
    _hit_cache[key] = (time.monotonic(), result)
    return result


# A report may not quote a hit rate off fewer evaluated calls than this. The
# floor is deliberately lower than MIN_SAMPLES (which gates a whole isotonic
# fit) but high enough that the number is not one week of luck.
MIN_STATEABLE_SAMPLES = 20


def evaluated_hit_rate(
    model_name: Optional[str] = None,
    days: int = 90,
    as_of: Optional[str] = None,
) -> dict:
    """Evaluated directional hit rate on active (non-HOLD) calls.

    ``model_name`` None means every model. The platform-wide number. ``as_of``
    bounds the window's upper edge so a historical/backtest run cannot quote a
    rate measured on sessions it has not reached yet; None means "through the
    latest evaluated row".

    Returns ``{"n", "hit_rate", "days", "through", "model"}`` with ``hit_rate``
    None whenever ``n`` is below :data:`MIN_STATEABLE_SAMPLES`, callers must
    then say so rather than print an unearned number.
    """
    from db.models import ModelPrediction
    from db.session import get_session
    from sqlalchemy import func

    # The upper bound is on target_date, not prediction_date: an outcome is not
    # knowable until the session it predicted has closed, so a run dated `as_of`
    # may only count calls that had already resolved by then.
    upper = _as_of_date(as_of)
    q = (_scheduled_calls(func.count(), _hit_avg(),
                          func.max(ModelPrediction.target_date))
         .where(ModelPrediction.prediction_date >= upper - timedelta(days=days),
                ModelPrediction.target_date <= upper))
    if model_name:
        q = q.where(ModelPrediction.model_name == model_name)

    with get_session() as session:
        n, rate, through = session.execute(q).one()

    n = int(n or 0)
    return {
        "model": model_name or "all models",
        "days": days,
        "n": n,
        "hit_rate": (round(float(rate), 3)
                     if n >= MIN_STATEABLE_SAMPLES and rate is not None else None),
        "through": str(through) if through else None,
    }


def track_record_sentence(
    model_name: Optional[str] = None,
    days: int = 90,
    as_of: Optional[str] = None,
) -> str:
    """One literal sentence a report may quote verbatim about its own accuracy.

    This is the only place the wording lives, so a reader-facing surface and a
    prompt cannot drift apart, and the LLM never has to phrase (or invent) it.
    """
    try:
        stats = evaluated_hit_rate(model_name, days=days, as_of=as_of)
    except Exception as e:  # a DB hiccup must not take a report down
        logger.warning(f"track record lookup failed: {e}")
        return ("Unavailable for this run, the evaluated history could not "
                "be read, so no hit rate is stated.")

    label = model_name or "all models on this platform"
    if stats["hit_rate"] is None:
        return (f"Not enough evaluated history to state a hit rate "
                f"({stats['n']} scored non-HOLD call"
                f"{'' if stats['n'] == 1 else 's'} for {label} in the last "
                f"{days} days; {MIN_STATEABLE_SAMPLES} is the minimum). Treat "
                f"the conviction below as an unproven estimate.")
    return (f"{stats['hit_rate']:.0%} of scored non-HOLD calls for "
            f"{label} were directionally correct over the last {days} days "
            f"(n={stats['n']}, through {stats['through']}). Coin-flip is 50%.")


def invalidate() -> None:
    """Force a refit on next use. Call after an evaluation run."""
    global _fitted_at
    _fitted_at = 0.0
    _hit_cache.clear()
