"""Bad Apples quality screen, ported from CygnetResearchTerminal.

Score-based detector that flags potentially poor-quality stocks against
performance, fundamental, valuation, and (best-effort) qualitative checks.
Every check returns a uniform 4-tuple (status, value, threshold, note) with
status in {"pass", "fail", "n/a"}.

Point-in-time adaptations for this pipeline's as_of contract:
  - price history (stock and benchmark) is truncated to as_of
  - quarterly statements keep only fiscal periods ending on/before as_of
    (same convention as the fundamentals block in models/single_agent.py;
    report-date lag is not modeled. Yfinance doesn't expose it)
  - insider transactions and earnings history are filtered to as_of
  - snapshot-only ``info`` fields (multiples, analyst targets, leverage
    ratios) cannot be rewound; those checks are labeled "snapshot" in their
    note so a backtest reader knows they reflect today's values

The 2026-08-07 experiment showed fail-count is NOT a next-day direction
signal (high-fail names rallied on that bounce day). It is a quality/risk
screen. It feeds the research and synthesis prompts as context, not a gate.
"""

import logging
import math
import re
import threading

import pandas as pd

logger = logging.getLogger(__name__)

# Tech uses QQQ; the rest use SPDR sector ETFs.
_SECTOR_BENCH = {
    "Technology": "QQQ",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
_DEFAULT_BENCH = "SPY"
_RF_ANNUAL = 0.045   # risk-free proxy for Sharpe
_COC_PROXY = 0.10    # cost-of-capital proxy for the ROIC check

_CACHE: dict[tuple[str, str], dict] = {}
_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Performance checks
# ---------------------------------------------------------------------------

def _ret_over(prices, days):
    if prices is None or len(prices) < 2:
        return None
    idx = max(0, len(prices) - 1 - days)
    return float(prices.iloc[-1] / prices.iloc[idx] - 1.0)


def check_underperformance(stock_prices, bench_prices):
    """Lags benchmark on multiple horizons."""
    if stock_prices is None or bench_prices is None:
        return ("n/a", None, None, "No price history")
    periods = [("6mo", 126), ("12mo", 252), ("36mo", 756)]
    rows, lag_count = [], 0
    for label, d in periods:
        s = _ret_over(stock_prices, d)
        b = _ret_over(bench_prices, d)
        if s is None or b is None:
            continue
        rows.append(f"{label}: {(s - b) * 100:+.1f}pp")
        if (s - b) < -0.05:
            lag_count += 1
    if not rows:
        return ("n/a", None, None, "Insufficient history")
    return ("fail" if lag_count >= 2 else "pass",
            f"{lag_count} of {len(rows)} periods lag",
            "≤1 of 3 lagging by >5pp", "; ".join(rows))


def check_sharpe(stock_prices):
    if stock_prices is None or len(stock_prices) < 60:
        return ("n/a", None, None, "Insufficient history")
    rets = stock_prices.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return ("n/a", None, None, "Flat returns")
    sharpe = (rets.mean() - _RF_ANNUAL / 252) / rets.std() * math.sqrt(252)
    return ("fail" if sharpe < 0 else "pass", f"{sharpe:.2f}", "≥ 0",
            f"σ={rets.std() * math.sqrt(252) * 100:.1f}% annualised")


def check_alpha(stock_prices, bench_prices):
    if stock_prices is None or bench_prices is None:
        return ("n/a", None, None, "Insufficient history")
    df = pd.concat([stock_prices.pct_change(), bench_prices.pct_change()],
                   axis=1).dropna()
    df.columns = ["s", "b"]
    if len(df) < 30 or df["b"].std() == 0:
        return ("n/a", None, None, "Insufficient overlap")
    beta = df.cov().iloc[0, 1] / df["b"].var()
    alpha_ann = (df["s"].mean() - beta * df["b"].mean()) * 252
    return ("fail" if alpha_ann < 0 else "pass",
            f"{alpha_ann * 100:+.1f}%/yr", "≥ 0", f"β={beta:.2f}")


def check_drawdown(stock_prices, bench_prices):
    """Still deep underwater while the benchmark has recovered."""
    if stock_prices is None or len(stock_prices) < 60:
        return ("n/a", None, None, "Insufficient history")
    peak = stock_prices.cummax()
    cur_dd = float(stock_prices.iloc[-1] / peak.iloc[-1] - 1.0)
    max_dd = float((stock_prices / peak - 1.0).min())
    bench_dd = None
    if bench_prices is not None and len(bench_prices) >= 60:
        bench_dd = float(bench_prices.iloc[-1] / bench_prices.cummax().iloc[-1] - 1.0)
    fail = cur_dd < -0.20 and (bench_dd is None or bench_dd > -0.05)
    note = f"max DD {max_dd * 100:.1f}%"
    if bench_dd is not None:
        note += f"; bench DD {bench_dd * 100:.1f}%"
    return ("fail" if fail else "pass", f"{cur_dd * 100:.1f}% from peak",
            "> -20% OR bench similarly down", note)


def check_vol(stock_prices, bench_prices):
    if stock_prices is None or bench_prices is None:
        return ("n/a", None, None, "Insufficient history")
    s = stock_prices.pct_change().std() * math.sqrt(252)
    b = bench_prices.pct_change().std() * math.sqrt(252)
    if b == 0:
        return ("n/a", None, None, "Bench vol zero")
    ratio = s / b
    return ("fail" if ratio > 1.75 else "pass", f"{ratio:.2f}x bench σ",
            "≤ 1.75x", f"stock σ={s * 100:.1f}%, bench σ={b * 100:.1f}%")


# ---------------------------------------------------------------------------
# Fundamental checks
# ---------------------------------------------------------------------------

def _ascending(series):
    return series.dropna().sort_index()


def check_revenue_trend(q_inc):
    if q_inc is None or q_inc.empty or "Total Revenue" not in q_inc.index:
        return ("n/a", None, None, "No revenue data")
    rev = _ascending(q_inc.loc["Total Revenue"])
    if len(rev) < 4:
        return ("n/a", None, None, "<4 quarters")
    declines = int((rev.diff().dropna().tail(3) < 0).sum())
    yoy = float(rev.iloc[-1] / rev.iloc[-4] - 1.0)
    return ("fail" if declines >= 2 or yoy < -0.05 else "pass",
            f"{declines} of last 3 QoQ down; YoY {yoy * 100:+.1f}%",
            "≤1 decline AND YoY > -5%", "")


def check_earnings_trend(q_inc):
    if q_inc is None or q_inc.empty or "Net Income" not in q_inc.index:
        return ("n/a", None, None, "No net income")
    earn = _ascending(q_inc.loc["Net Income"])
    if len(earn) < 4:
        return ("n/a", None, None, "<4 quarters")
    declines = int((earn.diff().dropna().tail(3) < 0).sum())
    return ("fail" if declines >= 2 else "pass",
            f"{declines} of last 3 QoQ down", "≤1 decline",
            f"Latest: ${earn.iloc[-1] / 1e6:,.0f}M")


def check_margins(q_inc):
    if q_inc is None or q_inc.empty or "Total Revenue" not in q_inc.index:
        return ("n/a", None, None, "No revenue")
    rev = _ascending(q_inc.loc["Total Revenue"])
    notes, total, fails = [], 0, 0
    for label, field in [("Gross", "Gross Profit"),
                         ("Op", "Operating Income"),
                         ("Net", "Net Income")]:
        if field not in q_inc.index:
            continue
        x = _ascending(q_inc.loc[field])
        common = rev.index.intersection(x.index)
        if len(common) < 3:
            continue
        m = (x.loc[common] / rev.loc[common]).sort_index()
        first, last = float(m.iloc[0]), float(m.iloc[-1])
        notes.append(f"{label}: {first * 100:.1f}%→{last * 100:.1f}%")
        total += 1
        if last < first - 0.02:    # 2pp compression
            fails += 1
    if total == 0:
        return ("n/a", None, None, "No margin data")
    return ("fail" if fails >= 2 else "pass",
            f"{fails} of {total} margins compressing", "≤1 compressing",
            "; ".join(notes))


def check_fcf(q_cf):
    if q_cf is None or q_cf.empty:
        return ("n/a", None, None, "No cash flow data")
    op_cf = next((q_cf.loc[f] for f in
                  ("Operating Cash Flow", "Total Cash From Operating Activities")
                  if f in q_cf.index), None)
    capex = next((q_cf.loc[f] for f in
                  ("Capital Expenditure", "Capital Expenditures")
                  if f in q_cf.index), None)
    if op_cf is None or capex is None:
        return ("n/a", None, None, "OCF/CAPEX missing")
    # CAPEX is reported negative, so OCF + CAPEX = FCF.
    fcf = _ascending(op_cf + capex)
    if fcf.empty:
        return ("n/a", None, None, "No FCF data")
    last = float(fcf.iloc[-1])
    prior = float(fcf.iloc[-2]) if len(fcf) >= 2 else None
    return ("fail" if last < 0 else "pass",
            f"Latest qtr FCF: ${last / 1e6:,.0f}M", "≥ 0",
            f"prior qtr: ${prior / 1e6:,.0f}M" if prior is not None else "")


def check_leverage(info):
    de = info.get("debtToEquity")     # yfinance reports as %
    cr = info.get("currentRatio")
    notes, flags = [], 0
    if de is not None:
        notes.append(f"D/E: {de:.0f}%")
        if de > 200:
            flags += 1
    if cr is not None:
        notes.append(f"Curr Ratio: {cr:.2f}")
        if cr < 1.0:
            flags += 1
    if not notes:
        return ("n/a", None, None, "No leverage metrics")
    return ("fail" if flags >= 1 else "pass", f"{flags} leverage flag(s)",
            "0 flags", "snapshot; " + "; ".join(notes))


def check_roic_proxy(info):
    """ROIC proxy = ROE (yfinance lacks true ROIC) vs 10% cost of capital."""
    roe = info.get("returnOnEquity")
    roa = info.get("returnOnAssets")
    metric = roe if roe is not None else roa
    if metric is None:
        return ("n/a", None, None, "No return metric")
    note = "snapshot" + (f"; ROA: {roa * 100:.1f}%" if roa is not None else "")
    return ("fail" if metric < _COC_PROXY else "pass",
            f"ROE: {metric * 100:.1f}%", f"≥ {_COC_PROXY * 100:.0f}%", note)


def check_earnings_misses(tk, as_of):
    if tk is None:
        return ("n/a", None, None, "Ticker unavailable")
    eh = None
    fetch_error = None
    for attr in ("earnings_history", "get_earnings_dates"):
        try:
            val = getattr(tk, attr)
            eh = val(limit=12) if callable(val) else val
            if eh is not None and not eh.empty:
                fetch_error = None
                break
        except Exception as e:
            # The note lands in the research prompt, so it must not claim the
            # company has no earnings history when the truth is that the
            # lookup failed. Both are "n/a"; only one is a fact.
            fetch_error = f"{type(e).__name__}: {e}"
            eh = None
    if eh is None or eh.empty:
        if fetch_error:
            logger.warning("earnings history lookup failed: %s", fetch_error)
            return ("n/a", None, None,
                    f"Earnings history unavailable ({fetch_error[:60]})")
        return ("n/a", None, None, "No earnings history")
    try:
        idx = pd.to_datetime(eh.index, errors="coerce")
        # errors="coerce" does not raise; it yields NaT. The old code took
        # `not idx.notna().any()` as licence to skip the filter and score the
        # whole frame, which is lookahead reported as a clean result. No
        # usable dates means the point-in-time contract cannot be honoured.
        if not idx.notna().any():
            logger.warning("earnings history has no parseable dates; "
                           "refusing rather than scoring unfiltered")
            return ("n/a", None, None, "Earnings dates unparseable; cannot "
                                       "guarantee point-in-time")
        eh = eh[idx.tz_localize(None) <= pd.Timestamp(as_of)]
    except Exception as e:
        logger.warning("earnings history as_of filter failed: %s", e)
        return ("n/a", None, None, "Earnings dates unparseable; cannot "
                                   "guarantee point-in-time")
    if eh.empty:
        return ("n/a", None, None, "No earnings history before as_of")
    cols = {c.lower(): c for c in eh.columns}
    actual_col = next((c for k, c in cols.items()
                       if "actual" in k and "eps" in k), None)
    est_col = next((c for k, c in cols.items()
                    if "estimate" in k and "eps" in k), None)
    if actual_col is None or est_col is None:
        for k in ("EPS Actual", "Reported EPS"):
            if k in eh.columns:
                actual_col = k
                break
        for k in ("EPS Estimate", "Expected EPS"):
            if k in eh.columns:
                est_col = k
                break
    if actual_col is None or est_col is None:
        return ("n/a", None, None, "Actual/estimate cols missing")
    df = eh[[actual_col, est_col]].dropna().tail(8)
    if df.empty:
        return ("n/a", None, None, "No paired actual/estimate")
    misses = int((df[actual_col] < df[est_col]).sum())
    return ("fail" if misses >= 2 else "pass",
            f"{misses} miss(es) in last {len(df)}", "≤1 miss", "")


def check_accounting(q_inc, q_bs):
    """Rising DSO and inventory growth outpacing revenue growth."""
    if (q_inc is None or q_inc.empty or q_bs is None or q_bs.empty
            or "Total Revenue" not in q_inc.index):
        return ("n/a", None, None, "Missing data")
    rev = _ascending(q_inc.loc["Total Revenue"])
    notes, flags = [], 0
    rec_field = next((c for c in
                      ("Accounts Receivable", "Net Receivables", "Receivables")
                      if c in q_bs.index), None)
    if rec_field:
        rec = _ascending(q_bs.loc[rec_field])
        common = rec.index.intersection(rev.index)
        if len(common) >= 3:
            dso = (rec.loc[common] / rev.loc[common] * 90).sort_index()
            first, last = float(dso.iloc[0]), float(dso.iloc[-1])
            notes.append(f"DSO {first:.0f}d→{last:.0f}d")
            if last > first * 1.20:
                flags += 1
    inv_field = next((c for c in ("Inventory", "Inventories")
                      if c in q_bs.index), None)
    if inv_field:
        inv = _ascending(q_bs.loc[inv_field])
        common = inv.index.intersection(rev.index)
        if len(common) >= 3:
            inv_g = float(inv.loc[common].iloc[-1] / inv.loc[common].iloc[0] - 1)
            rev_g = float(rev.loc[common].iloc[-1] / rev.loc[common].iloc[0] - 1)
            notes.append(f"Inv {inv_g * 100:+.0f}% vs Rev {rev_g * 100:+.0f}%")
            if inv_g - rev_g > 0.15:
                flags += 1
    if not notes:
        return ("n/a", None, None, "No DSO/inventory data")
    return ("fail" if flags >= 1 else "pass", f"{flags} accounting flag(s)",
            "0 flags", "; ".join(notes))


# ---------------------------------------------------------------------------
# Valuation checks
# ---------------------------------------------------------------------------

def check_multiples(info):
    pe = info.get("trailingPE")
    ev_ebitda = info.get("enterpriseToEbitda")
    ps = info.get("priceToSalesTrailing12Months")
    notes, total, expensive = [], 0, 0
    if pe is not None:
        notes.append(f"P/E {pe:.1f}")
        total += 1
        if pe > 35:
            expensive += 1
    if ev_ebitda is not None:
        notes.append(f"EV/EBITDA {ev_ebitda:.1f}")
        total += 1
        if ev_ebitda > 20:
            expensive += 1
    if ps is not None:
        notes.append(f"P/S {ps:.1f}")
        total += 1
        if ps > 8:
            expensive += 1
    if total == 0:
        return ("n/a", None, None, "No multiples")
    return ("fail" if expensive >= 2 else "pass",
            f"{expensive} of {total} multiples expensive", "≤1 expensive",
            "snapshot; " + "; ".join(notes))


def check_analyst_target(info, current_price):
    target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
    if not target or not current_price:
        return ("n/a", None, None, "No target")
    upside = (target / current_price - 1.0)
    rec = info.get("recommendationMean")
    note = "snapshot" + (f"; Rec mean: {rec:.1f}/5" if rec is not None else "")
    return ("fail" if upside < -0.05 else "pass",
            f"Target ${target:.2f} ({upside * 100:+.1f}%)",
            "Upside ≥ -5%", note)


def check_peg(info):
    peg = info.get("pegRatio") or info.get("trailingPegRatio")
    if peg is None:
        return ("n/a", None, None, "No PEG")
    return ("fail" if peg > 3.0 or peg < 0 else "pass",
            f"PEG {peg:.2f}", "0 < PEG ≤ 3", "snapshot")


def check_short_interest(info):
    """Crowded shorts: high float share or many days to cover."""
    pct_float = info.get("shortPercentOfFloat")
    days_cover = info.get("shortRatio")
    if pct_float is None and days_cover is None:
        return ("n/a", None, None, "No short-interest data")
    flags, notes = 0, []
    if pct_float is not None:
        notes.append(f"{pct_float * 100:.1f}% of float short")
        if pct_float > 0.10:
            flags += 1
    if days_cover is not None:
        notes.append(f"{days_cover:.1f} days to cover")
        if days_cover > 8:
            flags += 1
    return ("fail" if flags else "pass", f"{flags} short-interest flag(s)",
            "≤10% of float AND ≤8 days to cover", "snapshot; " + "; ".join(notes))


# How far back to look for a prior warmed .info snapshot to diff against.
_REVISION_LOOKBACK_DAYS = 10


def check_analyst_revisions(symbol, as_of, info):
    """Revision momentum: diff the analyst target/rec against a prior warmed
    .info snapshot from the Terminal cache. n/a until warm history exists."""
    target_now = info.get("targetMeanPrice")
    rec_now = info.get("recommendationMean")
    if target_now is None and rec_now is None:
        return ("n/a", None, None, "No analyst data")
    try:
        from datetime import date, timedelta
        from services import terminal_cache
        prior, prior_day = None, None
        end = date.fromisoformat(str(as_of)[:10])
        for back in range(3, _REVISION_LOOKBACK_DAYS + 1):
            d = end - timedelta(days=back)
            if d.weekday() >= 5:
                continue
            prior = terminal_cache.get_info(symbol, d.isoformat())
            if prior:
                prior_day = d.isoformat()
                break
    except Exception:
        prior = None
    if not prior:
        return ("n/a", None, None,
                f"No warmed snapshot in prior {_REVISION_LOOKBACK_DAYS}d to diff")
    target_then = prior.get("targetMeanPrice")
    rec_then = prior.get("recommendationMean")
    notes, flags = [f"vs {prior_day}"], 0
    if target_now is not None and target_then:
        chg = target_now / target_then - 1.0
        notes.append(f"target ${target_then:.2f}→${target_now:.2f} "
                     f"({chg * 100:+.1f}%)")
        if chg < -0.05:
            flags += 1
    if rec_now is not None and rec_then is not None:
        notes.append(f"rec mean {rec_then:.2f}→{rec_now:.2f}")
        if rec_now - rec_then > 0.3:   # 1=strong buy .. 5=sell; rising = worse
            flags += 1
    if len(notes) == 1:
        return ("n/a", None, None, "Prior snapshot lacks analyst fields")
    return ("fail" if flags else "pass", f"{flags} negative revision(s)",
            "no target cut >5%, no rec downgrade >0.3", "; ".join(notes))


# ---------------------------------------------------------------------------
# Qualitative (best-effort; yfinance has limited coverage here)
# ---------------------------------------------------------------------------

def check_insider_selling(tk, as_of):
    if tk is None:
        return ("n/a", None, None, "Ticker unavailable")
    try:
        ins = tk.insider_transactions
    except Exception as e:
        logger.warning("insider transactions lookup failed: %s", e)
        return ("n/a", None, None,
                f"Insider data unavailable ({type(e).__name__})")
    if ins is None or ins.empty:
        return ("n/a", None, None, "No insider data")
    cols = {c.lower(): c for c in ins.columns}
    date_col = next((c for k, c in cols.items() if "date" in k), None)
    val_col = next((c for k, c in cols.items() if "value" in k), None)
    type_col = next((c for k, c in cols.items()
                     if "transaction" in k or "txn" in k or "type" in k), None)
    if date_col is None or val_col is None:
        return ("n/a", None, None, "Insider cols missing")
    df = ins.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # Unparseable dates become NaT, and every NaT comparison is False, so
        # the rows drop out and the check reports "no transactions in 90d".
        # That is a finding the data does not support: we cannot tell whether
        # there were none or whether we simply could not read the dates.
        if df[date_col].notna().sum() == 0 and len(df):
            logger.warning("insider transactions have no parseable dates; "
                           "refusing rather than reporting no activity")
            return ("n/a", None, None, "Insider dates unparseable; cannot "
                                       "guarantee point-in-time")
        end = pd.Timestamp(as_of)
        df = df[(df[date_col] >= end - pd.Timedelta(days=90))
                & (df[date_col] <= end)]
    except Exception as e:
        # Swallowing this left df holding every transaction yfinance returned,
        # including any dated after as_of, and the check then scored against
        # them. That is both lookahead and the wrong window, reported as a
        # clean "pass".
        logger.warning("insider as_of filter failed: %s", e)
        return ("n/a", None, None, "Insider dates unparseable; cannot "
                                   "guarantee point-in-time")
    if df.empty:
        return ("pass", "No txns in last 90d", "Net sells ≤ $1M", "")
    if type_col is None:
        return ("n/a", None, None, "Type col missing")
    types = df[type_col].astype(str)
    sells = df[types.str.contains("Sale|Sell|Dispos", case=False, na=False)]
    buys = df[types.str.contains("Buy|Purchase|Acqui", case=False, na=False)]
    sv = float(sells[val_col].fillna(0).sum()) if not sells.empty else 0.0
    bv = float(buys[val_col].fillna(0).sum()) if not buys.empty else 0.0
    net = sv - bv
    return ("fail" if net > 1_000_000 else "pass",
            f"Net sells: ${net / 1e6:,.1f}M (90d)", "Net sells ≤ $1M",
            f"{len(sells)} sell / {len(buys)} buy")


# ---------------------------------------------------------------------------
# News red flags. PIT window scan
# ---------------------------------------------------------------------------

# Keyword-matched categories over the point-in-time news window. Hits are
# surfaced WITH their headline so the research model can judge context, for
# a cybersecurity name like PANW, "breach"/"hack" mentions are usually about
# the product space, not the company.
_RED_FLAG_PATTERNS = [
    ("leadership", re.compile(
        r"\b(?:ceo|cfo|coo|cto|chief \w+ officer|president|chairman|founder)\b"
        r".{0,60}?\b(?:resign\w*|steps? down|stepp\w+ down|depart\w*|exits?|"
        r"ousted|fired|retir\w+|replac\w+|succeed\w*|transition)\b", re.I)),
    ("layoffs", re.compile(
        r"\b(?:layoffs?|lays? off|job cuts?|workforce reduction|"
        r"cuts? \d[\d,]* jobs|headcount reduction|restructuring charge)\b", re.I)),
    ("investigation", re.compile(
        r"\b(?:sec|doj|ftc|antitrust)\b.{0,50}?\b(?:investigat\w+|probe|"
        r"subpoena|charges?|settl\w+)\b|\bclass action\b|\bsecurities fraud\b",
        re.I)),
    ("accounting", re.compile(
        r"\b(?:restat\w+ (?:earnings|results|financials)|material weakness|"
        r"delayed (?:10-[kq]|filing)|auditor (?:resign\w*|dismiss\w*)|"
        r"goodwill impairment)\b", re.I)),
    ("guidance", re.compile(
        r"\b(?:cuts?|lowers?|slash\w+|withdraw\w*|suspend\w*)\b.{0,30}?"
        r"\b(?:guidance|outlook|forecast)\b|\bprofit warning\b", re.I)),
    ("short_seller", re.compile(
        r"\bshort[- ]sell\w+ report\b|\b(?:hindenburg|muddy waters|"
        r"citron|kerrisdale)\b", re.I)),
    ("dilution", re.compile(
        r"\b(?:secondary offering|follow-on offering|share sale program|"
        r"convertible (?:notes?|debt) offering)\b", re.I)),
]

RED_FLAG_LOOKBACK_DAYS = 14
_SERIOUS_FLAGS = {"leadership", "investigation", "accounting", "guidance",
                  "short_seller"}


def scan_news_red_flags(symbol: str, as_of: str,
                        lookback_days: int = RED_FLAG_LOOKBACK_DAYS) -> list[dict] | None:
    """Scan the point-in-time news window for red-flag headlines.

    Returns a list of {category, headline, date, url} (empty when clean),
    or None when the news source is unavailable, "no news access" must not
    render as "no red flags".
    """
    try:
        from services.news_window import fetch_point_in_time_news
        articles = fetch_point_in_time_news(symbol, str(as_of)[:10],
                                            lookback_days=lookback_days)
    except Exception as e:
        logger.warning(f"{symbol}: red-flag news scan unavailable: {e}")
        return None

    hits: list[dict] = []
    seen: set[tuple] = set()
    for a in articles or []:
        text = f"{getattr(a, 'title', '') or ''}. {getattr(a, 'summary', '') or ''}"
        for category, pat in _RED_FLAG_PATTERNS:
            if not pat.search(text):
                continue
            headline = (getattr(a, "title", "") or "")[:160]
            key = (category, headline)
            if key in seen:
                continue
            seen.add(key)
            pub = getattr(a, "published_at", None)
            hits.append({
                "category": category,
                "headline": headline,
                "date": str(pub)[:10] if pub else None,
                "url": getattr(a, "url", None),
            })
    hits.sort(key=lambda h: (h["category"] not in _SERIOUS_FLAGS,
                             h["category"], h["date"] or ""))
    return hits


def check_news_red_flags(red_flags: list[dict] | None):
    if red_flags is None:
        return ("n/a", None, None, "News source unavailable")
    serious = [h for h in red_flags if h["category"] in _SERIOUS_FLAGS]
    cats = sorted({h["category"] for h in red_flags})
    if not red_flags:
        return ("pass", "No red-flag headlines",
                "0 serious hits", f"{RED_FLAG_LOOKBACK_DAYS}d window")
    return ("fail" if serious else "pass",
            f"{len(red_flags)} hit(s): {', '.join(cats)}",
            "0 serious hits",
            "; ".join(h["headline"][:80] for h in serious[:3]))


# ---------------------------------------------------------------------------
# Per-symbol orchestrator
# ---------------------------------------------------------------------------

def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return ("n/a", None, None, f"err: {type(e).__name__}")


_CHECK_SPEC = [
    ("Performance", "Underperformance vs benchmark",
        lambda ctx: check_underperformance(ctx["stock"], ctx["bench"])),
    ("Performance", "Sharpe ratio (1y)",
        lambda ctx: check_sharpe(ctx["stock"])),
    ("Performance", "Alpha vs benchmark",
        lambda ctx: check_alpha(ctx["stock"], ctx["bench"])),
    ("Performance", "Persistent drawdown",
        lambda ctx: check_drawdown(ctx["stock"], ctx["bench"])),
    ("Performance", "Volatility vs benchmark",
        lambda ctx: check_vol(ctx["stock"], ctx["bench"])),

    ("Fundamental", "Revenue trend (qtrly)",
        lambda ctx: check_revenue_trend(ctx["q_inc"])),
    ("Fundamental", "Net income trend (qtrly)",
        lambda ctx: check_earnings_trend(ctx["q_inc"])),
    ("Fundamental", "Margin trajectory",
        lambda ctx: check_margins(ctx["q_inc"])),
    ("Fundamental", "Free cash flow",
        lambda ctx: check_fcf(ctx["q_cf"])),
    ("Fundamental", "Leverage (D/E, current ratio)",
        lambda ctx: check_leverage(ctx["info"])),
    ("Fundamental", "ROIC proxy vs cost of capital",
        lambda ctx: check_roic_proxy(ctx["info"])),
    ("Fundamental", "Earnings vs estimates",
        lambda ctx: check_earnings_misses(ctx["tk"], ctx["as_of"])),
    ("Fundamental", "Accounting flags (DSO, inventory)",
        lambda ctx: check_accounting(ctx["q_inc"], ctx["q_bs"])),

    ("Valuation", "Expensive multiples",
        lambda ctx: check_multiples(ctx["info"])),
    ("Valuation", "Analyst price target",
        lambda ctx: check_analyst_target(ctx["info"], ctx["price"])),
    ("Valuation", "PEG ratio",
        lambda ctx: check_peg(ctx["info"])),
    ("Valuation", "Analyst revision momentum",
        lambda ctx: check_analyst_revisions(ctx["symbol"], ctx["as_of"],
                                            ctx["info"])),

    ("Qualitative", "Short interest",
        lambda ctx: check_short_interest(ctx["info"])),
    ("Qualitative", "Insider net selling (90d)",
        lambda ctx: check_insider_selling(ctx["tk"], ctx["as_of"])),
    ("Qualitative", "News red flags (leadership/legal/guidance)",
        lambda ctx: check_news_red_flags(ctx["red_flags"])),
]


def _pit_prices(tk, as_of):
    """3y of adjusted closes truncated to as_of. None when unavailable."""
    try:
        hist = tk.history(period="3y", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        closes = hist["Close"].dropna()
        idx = closes.index.tz_localize(None) if closes.index.tz else closes.index
        return closes[idx <= pd.Timestamp(as_of)]
    except Exception as e:
        # Callers turn None into "No price history", which reads as a fact
        # about the company. Log so the run can be told apart from one where
        # the company really has no history.
        logger.warning("price history unavailable for as_of %s: %s", as_of, e)
        return None


def _pit_quarters(frame, as_of):
    """Keep only fiscal periods ending on/before as_of."""
    if frame is None or frame.empty:
        return frame
    try:
        cols = pd.to_datetime(frame.columns, errors="coerce")
        keep = [c for c, d in zip(frame.columns, cols)
                if pd.notna(d) and d <= pd.Timestamp(as_of)]
        return frame[keep] if keep else frame.iloc[:, :0]
    except Exception:
        return frame


def analyze_symbol(symbol: str, as_of: str) -> dict:
    """Run the full scorecard for one symbol as of a date. Cached per
    (symbol, as_of); never raises: a failed fetch yields n/a checks."""
    from services.stock_data import get_ticker

    sym = symbol.strip().upper()
    as_of = str(as_of)[:10]
    key = (sym, as_of)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    out = {"symbol": sym, "as_of": as_of, "company": "", "sector": "",
           "benchmark": "", "checks": [], "scores": {}}
    try:
        tk = get_ticker(sym)
    except Exception:
        tk = None

    # The Terminal's nightly warmer caches the full .info blob per market
    # session: for that session the snapshot fields are ALSO point-in-time
    # correct, and the yfinance call is skipped entirely.
    info = None
    try:
        from services import terminal_cache
        info = terminal_cache.get_info(sym, as_of)
    except Exception:
        pass
    if not info:
        try:
            info = (tk.info or {}) if tk else {}
        except Exception:
            info = {}

    out["company"] = info.get("shortName") or info.get("longName") or ""
    sector = info.get("sector") or ""
    out["sector"] = sector
    bench = _SECTOR_BENCH.get(sector, _DEFAULT_BENCH)
    out["benchmark"] = bench

    stock_prices = _pit_prices(tk, as_of) if tk else None
    try:
        bench_prices = _pit_prices(get_ticker(bench), as_of)
    except Exception:
        bench_prices = None

    price = None
    if stock_prices is not None and len(stock_prices):
        price = float(stock_prices.iloc[-1])

    def _frame(attr):
        try:
            return _pit_quarters(getattr(tk, attr), as_of) if tk else None
        except Exception as e:
            # Same reasoning as _pit_closes: downstream this becomes
            # "No revenue data" in a prompt.
            logger.warning("statement %r unavailable: %s", attr, e)
            return None

    red_flags = scan_news_red_flags(sym, as_of)
    out["red_flags"] = red_flags or []
    out["headcount"] = info.get("fullTimeEmployees")

    ctx = {"tk": tk, "info": info, "as_of": as_of, "price": price,
           "symbol": sym,
           "stock": stock_prices, "bench": bench_prices,
           "red_flags": red_flags,
           "q_inc": _frame("quarterly_financials"),
           "q_cf": _frame("quarterly_cashflow"),
           "q_bs": _frame("quarterly_balance_sheet")}

    for cat, label, fn in _CHECK_SPEC:
        status, val, thresh, note = _safe(fn, ctx)
        out["checks"].append({"category": cat, "check": label,
                              "status": status, "value": val,
                              "threshold": thresh, "note": note})

    scores: dict = {}
    for c in out["checks"]:
        s = scores.setdefault(c["category"], {"fail": 0, "pass": 0, "n/a": 0})
        s[c["status"]] = s.get(c["status"], 0) + 1
    out["scores"] = scores
    out["total_fails"] = sum(s["fail"] for s in scores.values())
    out["total_checks"] = len(out["checks"])
    out["flag"] = ("bad_apple" if out["total_fails"] >= 8
                   else "caution" if out["total_fails"] >= 5 else "clean")

    with _CACHE_LOCK:
        _CACHE[key] = out
    return out


def summarize(result: dict) -> dict:
    """Compact dict for storage in the analysis payload (renderers/synthesis)."""
    return {
        "as_of": result["as_of"],
        "flag": result["flag"],
        "total_fails": result["total_fails"],
        "total_checks": result["total_checks"],
        "by_category": {cat: s["fail"] for cat, s in result["scores"].items()},
        "failed_checks": [
            {"category": c["category"], "check": c["check"],
             "value": c["value"], "note": c["note"]}
            for c in result["checks"] if c["status"] == "fail"
        ],
        "red_flags": (result.get("red_flags") or [])[:8],
    }


def format_bad_apples_block(symbol: str, result: dict) -> str:
    """Prompt block. Empty string when every check came back n/a."""
    checks = result.get("checks") or []
    if not checks or all(c["status"] == "n/a" for c in checks):
        return ""
    lines = [f"[{symbol}: Bad Apples quality screen as of {result['as_of']} "
             f"(vs {result.get('benchmark') or 'benchmark'})]"]
    if result["total_fails"]:
        by_cat = ", ".join(f"{cat} {s['fail']}"
                           for cat, s in result["scores"].items() if s["fail"])
        lines.append(f"Verdict: {result['flag'].upper()}: "
                     f"{result['total_fails']} of {result['total_checks']} "
                     f"checks failed ({by_cat})")
    else:
        lines.append(f"Verdict: CLEAN: 0 of {result['total_checks']} "
                     f"checks failed")
    for c in checks:
        if c["status"] != "fail":
            continue
        detail = f": {c['note']}" if c["note"] else ""
        lines.append(f"FAIL {c['category']}: {c['check']} = {c['value']}{detail}")
    red_flags = result.get("red_flags") or []
    if red_flags:
        lines.append(f"News red-flag mentions ({RED_FLAG_LOOKBACK_DAYS}d window, "
                     f"keyword-matched: judge each against its headline):")
        for h in red_flags[:6]:
            when = f" ({h['date']})" if h.get("date") else ""
            lines.append(f"  [{h['category']}] {h['headline']}{when}")
    if result.get("headcount"):
        lines.append(f"Headcount (snapshot): {result['headcount']:,} full-time employees")
    lines.append(
        "This is a QUALITY/RISK screen, not a timing signal. A high fail "
        "count argues for smaller size, tighter risk, and skepticism toward "
        "bullish theses; it does not predict next-day direction."
    )
    return "\n".join(lines)
