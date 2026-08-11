"""Alpha Lab — standing hypothesis tests over the live prediction history.

Three candidate edges were proposed and hand-tested on 2026-08-11; none was
significant on the data available (12 prediction dates). Rather than shipping
them as trading features on faith, they live here as experiments that re-run
on a schedule against the GROWING dataset and report when — if ever — one
crosses significance. The verdict framework is fixed in advance so the
goalposts cannot move with the noise:

  * cross_sectional — daily Spearman IC of the composite model score vs
    next-session return, and the top-k/bottom-k long-short spread.
    PASS requires |t| >= 2 on the spread with >= 40 trading days.
  * event_drift — signed 5- and 10-session drift after a >= move_pct one-day
    move. PASS requires |t| >= 2 with >= 300 events per horizon.
  * calibration_gate — walk-forward per-model isotonic calibration; trade
    only calibrated p >= gate. PASS requires the gated subset to beat the
    ungated hit rate by >= 3pp with >= 100 gated trades.

Every run stores its full result so the trajectory of each hypothesis is
auditable, not just its latest verdict.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DIR = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}

# Pre-registered pass criteria — see module docstring.
MIN_DAYS_XS = 40
MIN_EVENTS = 300
MIN_GATED = 100
T_CRIT = 2.0
GATE_EDGE_PP = 3.0


def _t_stat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / len(x) ** 0.5))


def _load_frames():
    from db.session import get_session
    from sqlalchemy import text

    with get_session() as s:
        preds = pd.read_sql(text("""
            SELECT prediction_date::text, symbol, model_name, decision,
                   confidence, up_probability, previous_close, actual_close,
                   was_correct, pnl_dollars
            FROM model_predictions
            WHERE actual_close IS NOT NULL AND previous_close IS NOT NULL
        """), s.get_bind())
        prices = pd.read_sql(text("""
            SELECT symbol, date, close FROM stock_prices
            WHERE symbol IN (SELECT DISTINCT symbol FROM model_predictions)
            ORDER BY symbol, date
        """), s.get_bind())
    return preds, prices


def test_cross_sectional(preds: pd.DataFrame, top_frac: float = 6.0) -> dict:
    """Daily rank IC + long/short spread of the composite score."""
    from scipy import stats as sps

    p = preds.copy()
    p["ret"] = p.actual_close / p.previous_close - 1
    p["score"] = np.where(
        p.up_probability.notna(), p.up_probability,
        0.5 + p.decision.map(_DIR) * p.confidence.fillna(0.5) / 2)
    by_ds = (p.groupby(["prediction_date", "symbol"])
             .agg(score=("score", "mean"), ret=("ret", "first")).reset_index())

    ics, spreads = [], []
    for _, g in by_ds.groupby("prediction_date"):
        if len(g) < 8 or g.score.nunique() < 3:
            continue
        ics.append(sps.spearmanr(g.score, g.ret).statistic)
        g = g.sort_values("score")
        k = max(2, int(len(g) // top_frac))
        spreads.append(g.tail(k).ret.mean() - g.head(k).ret.mean())

    t = _t_stat(np.array(spreads))
    n = len(spreads)
    return {
        "hypothesis": "cross_sectional",
        "n_days": n,
        "mean_ic": round(float(np.nanmean(ics)), 4) if ics else None,
        "mean_spread_pct": round(float(np.mean(spreads)) * 100, 3) if spreads else None,
        "t_spread": round(t, 2),
        "significant": bool(abs(t) >= T_CRIT and n >= MIN_DAYS_XS),
        "power": f"{n}/{MIN_DAYS_XS} days",
    }


def test_event_drift(prices: pd.DataFrame, move_pct: float = 5.0) -> dict:
    """Signed 5/10-session drift after a one-day move >= move_pct."""
    px = prices.copy()
    px["date"] = pd.to_datetime(px.date)
    px = px.sort_values(["symbol", "date"])
    px["r1"] = px.groupby("symbol").close.pct_change(fill_method=None)

    d5s, d10s = [], []
    for _, g in px.groupby("symbol"):
        c = g.close.to_numpy()
        r = g.r1.to_numpy()
        for i in np.where(np.abs(r) >= move_pct / 100)[0]:
            if i + 10 >= len(c) or i < 1:
                continue
            sign = np.sign(r[i])
            d5s.append((c[i + 5] / c[i] - 1) * sign)
            d10s.append((c[i + 10] / c[i] - 1) * sign)

    t5, t10 = _t_stat(np.array(d5s)), _t_stat(np.array(d10s))
    n = len(d5s)
    return {
        "hypothesis": "event_drift",
        "move_pct": move_pct,
        "n_events": n,
        "drift_5d_pct": round(float(np.mean(d5s)) * 100, 3) if d5s else None,
        "drift_10d_pct": round(float(np.mean(d10s)) * 100, 3) if d10s else None,
        "t_5d": round(t5, 2),
        "t_10d": round(t10, 2),
        "significant": bool((abs(t5) >= T_CRIT or abs(t10) >= T_CRIT)
                            and n >= MIN_EVENTS),
        "power": f"{n}/{MIN_EVENTS} events",
    }


def test_calibration_gate(preds: pd.DataFrame, gate: float = 0.55) -> dict:
    """Walk-forward isotonic gate: does trading only calibrated-p >= gate
    beat trading everything?"""
    from services.calibration_service import _pav

    def _apply(steps, x):
        v = steps[0][1]
        for thr, val in steps:
            if x >= thr:
                v = val
            else:
                break
        return v

    act = preds[(preds.decision != "HOLD") & preds.was_correct.notna()
                & preds.confidence.notna()].sort_values("prediction_date")
    gated, ungated = [], []
    for _, g in act.groupby("model_name"):
        if len(g) < 40:
            continue
        cut = int(len(g) * 0.6)
        train, test = g.iloc[:cut], g.iloc[cut:]
        steps = _pav(list(zip(train.confidence.astype(float),
                              train.was_correct.astype(float))))
        if not steps:
            continue
        calp = test.confidence.astype(float).map(lambda x: _apply(steps, x))
        ungated.append(test)
        gated.append(test[calp >= gate])

    if not ungated:
        return {"hypothesis": "calibration_gate", "n_gated": 0,
                "significant": False, "power": f"0/{MIN_GATED} gated trades"}
    u = pd.concat(ungated)
    g = pd.concat(gated) if gated else u.iloc[0:0]
    u_hit = float(u.was_correct.mean())
    g_hit = float(g.was_correct.mean()) if len(g) else None
    edge_pp = (g_hit - u_hit) * 100 if g_hit is not None else None
    return {
        "hypothesis": "calibration_gate",
        "gate": gate,
        "n_ungated": len(u),
        "n_gated": len(g),
        "hit_ungated": round(u_hit, 3),
        "hit_gated": round(g_hit, 3) if g_hit is not None else None,
        "edge_pp": round(edge_pp, 1) if edge_pp is not None else None,
        "significant": bool(edge_pp is not None and edge_pp >= GATE_EDGE_PP
                            and len(g) >= MIN_GATED),
        "power": f"{len(g)}/{MIN_GATED} gated trades",
    }


def run_all(move_pct: float = 5.0, gate: float = 0.55,
            top_frac: float = 6.0) -> dict:
    """Run every standing hypothesis test and persist the snapshot."""
    from services import progress_service as prog

    preds, prices = _load_frames()
    results = {
        "as_of": datetime.now().isoformat(),
        "n_evaluated_predictions": len(preds),
        "tests": [
            test_cross_sectional(preds, top_frac=top_frac),
            test_event_drift(prices, move_pct=move_pct),
            test_calibration_gate(preds, gate=gate),
        ],
    }
    passing = [t["hypothesis"] for t in results["tests"] if t["significant"]]
    results["passing"] = passing

    for t in results["tests"]:
        prog.emit("action",
                  f"Alpha Lab {t['hypothesis']}: "
                  f"{'SIGNIFICANT' if t['significant'] else 'not significant'} "
                  f"({t['power']})")
    if passing:
        prog.emit("action",
                  f"Alpha Lab: {', '.join(passing)} crossed the pre-registered "
                  f"bar — review before acting; this is a flag, not a trade")

    # Persist the snapshot so the trajectory is auditable.
    try:
        import json
        from services import persistence_service as ps
        ps.store_report(
            symbol=None, trade_date=datetime.now().strftime("%Y-%m-%d"),
            report_type="alpha_lab",
            input_data_hash=ps.compute_data_hash(
                {"n": len(preds), "params": [move_pct, gate, top_frac]}),
            content=json.dumps(results, default=str, indent=2),
            file_format="json",
        )
    except Exception as e:
        logger.warning(f"alpha lab snapshot not archived: {e}")

    return results
