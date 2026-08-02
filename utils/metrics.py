"""Computed trading metrics for prompt grounding.

Every number an LLM cites should be computed here, not asserted from
narrative. All functions take an OHLCV DataFrame already truncated to the
as-of date, so backtests carry no lookahead bias.

Expected DataFrame: columns Open/High/Low/Close/Volume (case-insensitive),
DatetimeIndex or a Date column, ascending order.
"""

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def wilder_rsi_series(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI(14) with Wilder smoothing — exact match to ta.RSIIndicator.

    The hand-rolled copies this replaces used a simple rolling mean
    (Cutler's RSI) while claiming to be Wilder's; the two disagree by
    several points right where the oversold/neutral interpretation
    boundary sits. Uses ewm(alpha=1/n) like the ta library the ML
    features are built on, so the LLM and the models see the same number.
    All-loss==0 windows return 100 (ta's convention), not NaN.
    """
    diff = close.diff(1)
    up = diff.where(diff > 0, 0.0)
    down = -diff.where(diff < 0, 0.0)
    emaup = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    emadn = down.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = emaup / emadn
    return pd.Series(
        np.where(emadn == 0, 100.0, 100.0 - 100.0 / (1.0 + rs)),
        index=close.index,
    )


def wilder_atr_series(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
) -> pd.Series:
    """ATR(14) matching ta.AverageTrueRange bar-for-bar.

    ta seeds bar n-1 with the SMA of the first n true ranges and then
    applies the Wilder recursion — a pure ewm(alpha=1/n) does NOT converge
    to it even after 120 bars, so the seeding is replicated exactly.
    One deliberate difference: warm-up bars are NaN here, where ta emits
    literal 0.0 (which can leak through .dropna() as fake volatility).
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)  # skipna: first bar = high - low, as in ta
    atr = np.full(len(tr), np.nan)
    if len(tr) >= n:
        atr[n - 1] = tr.iloc[:n].mean()
        for i in range(n, len(tr)):
            atr[i] = (atr[i - 1] * (n - 1) + tr.iloc[i]) / n
    return pd.Series(atr, index=close.index)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and index; return ascending-date copy."""
    out = df.copy()
    if "Date" in out.columns:
        out = out.set_index("Date")
    out.columns = [str(c).capitalize() for c in out.columns]
    out = out.sort_index()
    return out


def compute_trading_metrics(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Compute the canonical short-horizon metrics from OHLCV.

    Returns a dict of floats (or None where insufficient data):
      atr_14, atr_pct, realized_vol_20d (annualized %), support_20d,
      resistance_20d, close, rr_ratio (reward/risk), volume_last,
      volume_avg_20d, volume_ratio, drawdown_from_high_pct (period),
      period_high, period_high_date
    """
    if df is None or not len(df):
        return {}
    try:
        d = _norm(df)
        # Per-metric gating below: each metric computes when ITS minimum
        # history exists. The old single len<15 gate meant one short frame
        # deleted the entire "validated" block the prompt tells the LLM to
        # prefer — support/resistance need far fewer bars than ATR.
        if len(d) < 2:
            return {}

        close = d["Close"]
        high = d["High"]
        low = d["Low"]

        last_close = float(close.iloc[-1])
        m: dict = {
            "close": round(last_close, 2),
            "as_of": str(d.index[-1])[:10],
        }

        # ATR(14) — true Wilder (ta-equivalent seeding + recursion)
        atr14 = wilder_atr_series(high, low, close).iloc[-1]
        m["atr_14"] = round(float(atr14), 2) if pd.notna(atr14) else None
        m["atr_pct"] = (
            round(float(atr14) / last_close * 100, 2)
            if pd.notna(atr14) and last_close > 0 else None
        )

        # 20d realized volatility, annualized
        rets = close.pct_change().dropna()
        rv = None
        if len(rets) >= lookback:
            rv = float(rets.tail(lookback).std() * math.sqrt(252) * 100)
        m["realized_vol_20d"] = round(rv, 1) if rv is not None else None

        # Swing support/resistance from last `lookback` sessions (excluding
        # today's bar would be stricter, but the last bar is a completed
        # session here). Meaningful from a handful of bars.
        if len(d) >= 5:
            window = d.tail(lookback)
            support = float(window["Low"].min())
            resistance = float(window["High"].max())
            m["support_20d"] = round(support, 2)
            m["resistance_20d"] = round(resistance, 2)

            # Reward:risk from named levels
            risk = last_close - support
            reward = resistance - last_close
            m["rr_ratio"] = round(reward / risk, 2) if risk > 0 else None

            # Distances in ATR multiples — precomputed so the LLM interprets
            # instead of doing this arithmetic (wrongly) itself.
            if m["atr_14"]:
                m["dist_to_support_atr"] = round(risk / m["atr_14"], 1)
                m["dist_to_resistance_atr"] = round(reward / m["atr_14"], 1)

        # Volume trend
        if "Volume" in d.columns and d["Volume"].notna().any():
            vol_avg = float(d["Volume"].tail(lookback).mean())
            m["volume_last"] = float(d["Volume"].iloc[-1])
            m["volume_avg_20d"] = vol_avg
            m["volume_ratio"] = (
                round(m["volume_last"] / vol_avg, 2) if vol_avg else None
            )

        # Drawdown vs the trailing-252-bar close high — a defined window, so
        # the >20% hard rule no longer changes with the caller's chart period.
        basis = close.tail(252)
        period_high = float(basis.max())
        m["period_high"] = round(period_high, 2)
        m["period_high_date"] = str(basis.idxmax())[:10]
        m["period_high_bars"] = int(len(basis))
        m["drawdown_from_high_pct"] = (
            round((last_close / period_high - 1.0) * 100, 1)
            if period_high > 0 else None
        )

        return m
    except Exception as e:
        logger.warning(f"Metric computation failed: {e}")
        return {}


def format_metrics_block(symbol: str, m: dict) -> str:
    """Format computed metrics as a prompt block. Empty string if no metrics."""
    if not m or m.get("close") is None:
        return ""
    # Every key is optional now (per-metric gating upstream) — .get()
    # throughout so a partial dict renders what it has instead of KeyError.
    lines = [f"[{symbol} — computed from validated OHLCV through {m.get('as_of', '?')}]"]
    lines.append(f"Last close: ${m['close']}")
    if m.get("atr_14") is not None and m.get("atr_pct") is not None:
        lines.append(f"ATR(14): ${m['atr_14']} ({m['atr_pct']}% of price) — expected 1-day move")
    if m.get("realized_vol_20d") is not None:
        lines.append(f"Realized volatility (20d, annualized): {m['realized_vol_20d']}%")
    if m.get("support_20d") is not None and m.get("resistance_20d") is not None:
        lines.append(
            f"20-day range: support ${m['support_20d']} / resistance ${m['resistance_20d']}"
        )
        if m.get("rr_ratio") is not None:
            risk = round(m["close"] - m["support_20d"], 2)
            reward = round(m["resistance_20d"] - m["close"], 2)
            lines.append(
                f"Reward:Risk from close: +${reward} to resistance vs -${risk} to support "
                f"= {m['rr_ratio']}:1"
            )
        if (m.get("dist_to_support_atr") is not None
                and m.get("dist_to_resistance_atr") is not None):
            lines.append(
                f"Distance in ATR multiples: {m['dist_to_support_atr']} ATRs above "
                f"support, {m['dist_to_resistance_atr']} ATRs below resistance"
            )
    if m.get("volume_ratio") is not None:
        lines.append(
            f"Volume: last session {m['volume_last']:,.0f} vs 20d avg {m['volume_avg_20d']:,.0f} "
            f"({m['volume_ratio']}x average)"
        )
    if m.get("drawdown_from_high_pct") is not None:
        bars = m.get("period_high_bars") or 0
        label = "52-week close high" if bars >= 252 else f"{bars}-bar close high"
        lines.append(
            f"Drawdown from {label} (${m.get('period_high')} on {m.get('period_high_date')}): "
            f"{m['drawdown_from_high_pct']}%"
        )
    return "\n".join(lines)


def compute_peer_relative_strength(
    symbol: str,
    peers: list[str],
    as_of: str | None = None,
    period_days: int = 30,
    sector_etf: str | None = None,
) -> str:
    """Compare 1-month AND 1-week returns of symbol vs peers (and sector ETF).

    Fetches peer OHLCV (truncated to as_of for backtests) and returns a
    formatted prompt block. The two horizons separate "chronic laggard"
    from "hit this week"; the sector ETF row anchors the comparison beyond
    the direct peer set. Empty string on failure or no peers.
    """
    if not peers:
        return ""
    try:
        from services.stock_data import fetch_stock_data

        def _rets(sym: str):
            """(1-month, 1-week) percentage returns, either may be None."""
            df = fetch_stock_data(sym, period="3mo")
            if df is None or df.empty:
                return None, None
            d = _norm(df)
            if as_of:
                d = d[d.index <= str(as_of)]
            if len(d) < 2:
                return None, None

            def _window_ret(n_sessions: int):
                w = d.tail(n_sessions)
                if len(w) < 2:
                    return None
                return (float(w["Close"].iloc[-1]) / float(w["Close"].iloc[0]) - 1) * 100

            return _window_ret(period_days), _window_ret(6)  # 6 bars = 5 sessions

        def _fmt_pair(mo, wk):
            mo_s = f"{mo:+.1f}% (1mo)" if mo is not None else "n/a (1mo)"
            wk_s = f"{wk:+.1f}% (1wk)" if wk is not None else "n/a (1wk)"
            return f"{mo_s} / {wk_s}"

        own_mo, own_wk = _rets(symbol)
        if own_mo is None:
            return ""
        lines = [f"[Peer relative strength — returns"
                 + (f" through {as_of}" if as_of else "") + "]"]
        lines.append(f"{symbol}: {_fmt_pair(own_mo, own_wk)}")
        peer_rets = []
        for p in peers:
            mo, wk = _rets(p)
            if mo is not None:
                peer_rets.append(mo)
                lines.append(f"{p}: {_fmt_pair(mo, wk)}")
        if not peer_rets:
            return ""
        avg = sum(peer_rets) / len(peer_rets)
        lines.append(f"Peer average: {avg:+.1f}% (1mo) | {symbol} vs peers: {own_mo - avg:+.1f}pp")
        if sector_etf:
            etf_mo, etf_wk = _rets(sector_etf)
            if etf_mo is not None:
                lines.append(
                    f"Sector ETF {sector_etf}: {_fmt_pair(etf_mo, etf_wk)} | "
                    f"{symbol} vs sector: {own_mo - etf_mo:+.1f}pp (1mo)"
                )
        if own_mo < avg - 5:
            lines.append("NOTE: material underperformance vs peers — likely company-specific cause.")
        elif abs(own_mo - avg) <= 5:
            lines.append("NOTE: moves roughly with peers — sector-wide repricing more likely "
                         "than company-specific mispricing. 'Cheap' may be correct.")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Peer RS computation failed for {symbol}: {e}")
        return ""
