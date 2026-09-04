"""Score every combination method against history, without re-running models.

The run dialog offers four ways of turning member votes into one decision, but
only calls actually made get scored, so three of the four have no track record
and the choice has never been evidence-based.

Nothing needs re-predicting to fix that. Every member's decision, confidence
and up-probability is already persisted per (symbol, prediction_date), and so
is the outcome. Replaying the combiner over stored inputs yields exactly what
each method WOULD have decided, and the same scoring rules the live evaluator
uses answer whether it would have been right.

Two properties make this honest rather than a backtest fantasy:

  - No lookahead. Inputs are the member rows as stored, outcomes are the
    already-recorded actual_close. Nothing is recomputed from prices the
    models could not have seen.
  - Nothing is written. Counterfactual verdicts are not predictions anyone
    made, and persisting them would corrupt the scoreboard. This reports.

Members and weights are held constant across methods so the method is the only
variable. Weights come from config defaults rather than whatever was configured
on the day, which is a deliberate simplification: per-run weights were never
persisted in recoverable form (`weights_used` stores weight x confidence for
confidence_weighted, which cannot be inverted).

Usage:
    python scripts/replay_ensemble_methods.py
    python scripts/replay_ensemble_methods.py --min-agree 2 --since 2026-06-01
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODEL                                    # noqa: E402
from db.session import get_session                          # noqa: E402
from models.base import compute_pnl                         # noqa: E402
from models.ensemble_model import METHODS, EnsembleModel    # noqa: E402
from sqlalchemy import text                                 # noqa: E402

SYNTHETIC = {"ensemble", "recommendation_synthesis"}


def load_cohorts(since: str | None) -> dict:
    """Member rows plus outcome, grouped by (symbol, prediction_date).

    Scheduled rows only, like every other scoreboard: an ad-hoc rerun writes
    its own row for a call the daily job already made, and here the later
    row would silently replace that member's decision in the cohort.
    """
    from services.cache_service import SCHEDULED_ONLY_SQL
    sql = f"""
        SELECT symbol, prediction_date, target_date, model_name, decision,
               confidence, up_probability, previous_close, actual_close
          FROM model_predictions
         WHERE model_name NOT IN ('ensemble', 'recommendation_synthesis')
           AND actual_close IS NOT NULL
           AND previous_close IS NOT NULL
           AND {SCHEDULED_ONLY_SQL}
    """
    if since:
        sql += " AND prediction_date >= :since"
    params = {"since": since} if since else {}

    cohorts: dict = defaultdict(lambda: {"members": {}, "outcome": None})
    with get_session() as s:
        for r in s.execute(text(sql), params).all():
            key = (r.symbol, str(r.prediction_date))
            cohorts[key]["members"][r.model_name] = {
                "decision": r.decision,
                "confidence": r.confidence if r.confidence is not None else 0.5,
                "up_probability": r.up_probability,
            }
            cohorts[key]["outcome"] = {
                "previous_close": float(r.previous_close),
                "actual_close": float(r.actual_close),
                "target_date": str(r.target_date),
            }
    return cohorts


def hold_bands(cohorts: dict) -> dict:
    """Per-(symbol, target_date) no-trade band, matching the live evaluator.

    A HOLD is correct when the move stayed inside the symbol's own typical
    daily range, and that band must come only from bars before the target
    session. Reusing CacheService._hold_band keeps this identical to how the
    stored rows were scored, rather than a second opinion about it.
    """
    from services.cache_service import get_cache
    cache = get_cache()
    bands, memo = {}, {}
    with get_session() as s:
        for (symbol, _pdate), c in cohorts.items():
            tgt = c["outcome"]["target_date"]
            key = (symbol, tgt)
            if key not in bands:
                bands[key] = cache._hold_band(s, symbol, memo, before_date=tgt)
    return bands


def score(decision: str, prev: float, actual: float, band: float) -> tuple:
    """(was_correct, pnl) under the live evaluator's rules."""
    move = (actual - prev) / prev if prev else 0.0
    if decision == "HOLD":
        return abs(move) <= band, 0.0
    went_up = actual > prev
    correct = went_up if decision == "BUY" else not went_up
    return correct, compute_pnl(decision, prev, actual)


def replay(cohorts: dict, bands: dict, min_agree: int) -> dict:
    model = EnsembleModel()
    weights = dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)
    df = pd.DataFrame({"Close": [100.0] * 60})   # unused by the combiner
    stats = {m: defaultdict(float) for m in METHODS}
    for m in METHODS:
        stats[m]["decisions"] = defaultdict(int)

    for (symbol, _pdate), c in cohorts.items():
        members, out = c["members"], c["outcome"]
        if len(members) < 2:
            continue                       # nothing to combine
        band = bands[(symbol, out["target_date"])]
        for method in METHODS:
            r = model.predict(symbol, df, other_results=members,
                              ensemble_config={
                                  "enabled_models": list(members),
                                  "weights": weights,
                                  "method": method,
                                  "min_agree": min_agree})
            correct, pnl = score(r.decision, out["previous_close"],
                                 out["actual_close"], band)
            st = stats[method]
            st["n"] += 1
            st["decisions"][r.decision] += 1
            st["pnl"] += pnl
            st["conf_sum"] += r.up_probability if r.decision == "BUY" else (
                1 - r.up_probability if r.decision == "SELL" else 0.5)
            if r.decision in ("BUY", "SELL"):
                st["directional"] += 1
                st["dir_correct"] += int(correct)
                st["dir_pnl"] += pnl
            else:
                st["holds"] += 1
                st["hold_correct"] += int(correct)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-agree", type=int, default=MODEL.ENSEMBLE_MIN_AGREE)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD lower bound")
    args = ap.parse_args()

    cohorts = load_cohorts(args.since)
    usable = {k: v for k, v in cohorts.items() if len(v["members"]) >= 2}
    if not usable:
        print("No scored cohorts with 2+ members. Nothing to replay.")
        return 1

    dates = sorted({d for _, d in usable})
    print(f"Replaying {len(usable)} (symbol, date) cohorts "
          f"across {len(dates)} sessions: {dates[0]} to {dates[-1]}")
    print(f"min_agree={args.min_agree}, weights=config defaults, "
          f"members held constant per cohort\n")

    stats = replay(usable, hold_bands(usable), args.min_agree)

    hdr = (f"{'method':22s} {'trades':>7s} {'hit':>7s} {'z':>6s} "
           f"{'P&L':>10s} {'$/trade':>8s} {'holds':>6s} {'hold ok':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for method in METHODS:
        st = stats[method]
        d = int(st["directional"])
        hit = st["dir_correct"] / d if d else float("nan")
        z = (hit - 0.5) / math.sqrt(0.25 / d) if d else float("nan")
        per = st["dir_pnl"] / d if d else float("nan")
        h = int(st["holds"])
        hok = st["hold_correct"] / h if h else float("nan")
        print(f"{method:22s} {d:7d} {hit:6.1%} {z:+6.2f} "
              f"{st['dir_pnl']:10.2f} {per:8.2f} {h:6d} {hok:7.1%}")

    print("\nmix of calls")
    for method in METHODS:
        dec = stats[method]["decisions"]
        total = sum(dec.values())
        parts = "  ".join(f"{k} {dec[k]:3d} ({dec[k]/total:4.0%})"
                          for k in ("BUY", "SELL", "HOLD"))
        print(f"  {method:22s} {parts}")

    best = max(METHODS, key=lambda m: stats[m]["dir_pnl"])
    print(f"\nBest P&L: {best} (${stats[best]['dir_pnl']:.2f}). "
          f"A |z| under about 2 means the hit rate is not distinguishable "
          f"from a coin flip at this sample size.")
    print("Counterfactual: nothing here was written to the database.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
