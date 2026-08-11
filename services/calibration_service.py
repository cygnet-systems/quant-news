"""Per-model confidence calibration from evaluated prediction history.

Raw model confidences are demonstrably anti-calibrated on this platform
(claimed 71% → 42.5% actual across the live era). Everything that consumes
a confidence — UI badges, ensemble weights, the scoreboard — should go
through :func:`calibrate` instead of trusting the raw number.

The mapping is an isotonic regression fit per model on (raw_confidence,
was_correct) pairs from evaluated ACTIVE calls (HOLDs carry no direction to
score). Isotonic is the right shape here: it is monotone, so a model whose
confidence is directionally meaningful keeps its ordering, and a model whose
confidence is pure noise flattens toward its base rate — which is itself the
honest answer.

Fits are cached in-process for CALIBRATION_TTL_S and refit lazily; the data
changes at most once per evaluation run, so staleness is bounded and cheap.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
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


def _pav(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pool-adjacent-violators on (raw_conf, outcome) pairs, ascending by raw.

    Returns the isotonic step function as (raw, calibrated) knots. Implemented
    directly — scikit-learn is not a dependency of this project.
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


def _fit_all() -> dict[str, _ModelFit]:
    from db.session import get_session
    from sqlalchemy import text

    fits: dict[str, _ModelFit] = {}
    with get_session() as session:
        rows = session.execute(text("""
            SELECT model_name, confidence, was_correct
            FROM model_predictions
            WHERE was_correct IS NOT NULL
              AND decision != 'HOLD'
              AND confidence IS NOT NULL
        """)).fetchall()

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


def calibrate(model_name: str, raw_confidence: Optional[float]) -> Optional[float]:
    """Map a raw confidence to its historical hit rate for this model.

    Returns None when there is not enough evaluated history to say anything
    (< MIN_SAMPLES active evaluated calls) — callers should then show no
    number rather than an unearned one.
    """
    if raw_confidence is None:
        return None
    _ensure_fresh()
    fit = _fits.get(model_name)
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


def rolling_hit_rate(model_name: str, days: int = 30) -> Optional[float]:
    """Decay-free rolling hit rate on active calls over the last `days`.

    Used for performance-decay ensemble weighting. None below MIN_SAMPLES/3 —
    a weekly-scale window needs a lower floor than the full calibration fit,
    but still refuses single-digit sample sizes. TTL-cached: a 20-symbol
    batch would otherwise repeat the same query per member per symbol.
    """
    from db.session import get_session
    from sqlalchemy import text

    key = (model_name, days)
    cached = _hit_cache.get(key)
    if cached and time.monotonic() - cached[0] < CALIBRATION_TTL_S:
        return cached[1]

    with get_session() as session:
        row = session.execute(text("""
            SELECT count(*),
                   avg(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END)
            FROM model_predictions
            WHERE model_name = :model
              AND was_correct IS NOT NULL
              AND decision != 'HOLD'
              AND prediction_date >= (CURRENT_DATE - make_interval(days => :days))
        """), {"model": model_name, "days": days}).fetchone()
    n, rate = (row[0] or 0), row[1]
    result = (None if n < max(10, MIN_SAMPLES // 3) or rate is None
              else round(float(rate), 3))
    _hit_cache[(model_name, days)] = (time.monotonic(), result)
    return result


def calibration_table() -> dict[str, dict]:
    """Snapshot for the scoreboard: per model, sample size, base rate, and
    the fitted mapping at a few reference raw confidences."""
    _ensure_fresh()
    out = {}
    for model, fit in _fits.items():
        entry: dict = {"n": fit.n, "base_rate": fit.base_rate}
        if fit.steps:
            entry["mapping"] = {
                str(ref): calibrate(model, ref)
                for ref in (0.3, 0.5, 0.7, 0.9)
            }
        out[model] = entry
    return out


def invalidate() -> None:
    """Force a refit on next use — call after an evaluation run."""
    global _fitted_at
    _fitted_at = 0.0
    _hit_cache.clear()
