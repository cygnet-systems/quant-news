"""Verify the shared Wilder helpers match the ta library bar-for-bar.

The hand-rolled RSI/ATR in the LLM prompt path were replaced with
utils.metrics.wilder_rsi_series / wilder_atr_series, which claim exact
equivalence with ta.RSIIndicator / ta.AverageTrueRange (the implementations
the ML features are built on). This script proves it, including on SHORT
frames (15/30/60 bars), where ATR's SMA-seeded recursion diverges hard from
naive ewm implementations and where walk-forward backtests actually operate.

Run: python scripts/verify_indicators.py
Exit code 0 = all equivalences hold.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import ta

from services.stock_data import fetch_stock_data
from utils.metrics import (
    compute_trading_metrics,
    format_metrics_block,
    wilder_atr_series,
    wilder_rsi_series,
)

TOL = 1e-9
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


for symbol in ("AMBA", "SPY"):
    df = fetch_stock_data(symbol, period="2y")
    close, high, low = df["Close"], df["High"], df["Low"]
    print(f"\n{symbol} ({len(df)} bars)")

    # Full-length equivalence (compare where both are past warm-up)
    rsi_ours = wilder_rsi_series(close)
    rsi_ta = ta.momentum.RSIIndicator(close, window=14).rsi()
    both = rsi_ours.notna() & rsi_ta.notna()
    check("RSI == ta.RSIIndicator (full frame)",
          bool(np.allclose(rsi_ours[both], rsi_ta[both], atol=TOL)),
          f"max|diff|={float((rsi_ours[both] - rsi_ta[both]).abs().max()):.2e}")

    atr_ours = wilder_atr_series(high, low, close)
    atr_ta = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    # ta emits literal 0.0 for warm-up bars (its known wart); ours emits NaN
    # there on purpose. Compare from the seed bar onward.
    ours_tail, ta_tail = atr_ours.iloc[13:], atr_ta.iloc[13:]
    check("ATR == ta.AverageTrueRange (from seed bar)",
          bool(np.allclose(ours_tail, ta_tail, atol=TOL)),
          f"max|diff|={float((ours_tail - ta_tail).abs().max()):.2e}")

    # Short frames: where the seeding matters and backtests actually run
    for n in (15, 30, 60):
        sub = df.tail(n)
        a_ours = wilder_atr_series(sub["High"], sub["Low"], sub["Close"]).iloc[-1]
        a_ta = ta.volatility.AverageTrueRange(
            sub["High"], sub["Low"], sub["Close"], window=14
        ).average_true_range().iloc[-1]
        check(f"ATR on {n}-bar frame", abs(float(a_ours) - float(a_ta)) < TOL,
              f"ours={a_ours:.4f} ta={a_ta:.4f}")

        r_ours = wilder_rsi_series(sub["Close"]).iloc[-1]
        r_ta = ta.momentum.RSIIndicator(sub["Close"], window=14).rsi().iloc[-1]
        check(f"RSI on {n}-bar frame", abs(float(r_ours) - float(r_ta)) < TOL,
              f"ours={r_ours:.2f} ta={r_ta:.2f}")

# RSI degenerate case: all-up window must be 100 (ta convention), not NaN
import pandas as pd

up_only = pd.Series(np.linspace(100, 130, 40))
check("RSI all-gains window == 100",
      float(wilder_rsi_series(up_only).iloc[-1]) == 100.0)

# format_metrics_block must not KeyError on any partial dict
df = fetch_stock_data("SPY", period="2y")
for n in (2, 5, 10, 14, 20, 60):
    m = compute_trading_metrics(df.tail(n))
    try:
        format_metrics_block("TEST", m)
        check(f"format_metrics_block on {n}-bar metrics ({len(m)} keys)", True)
    except KeyError as e:
        check(f"format_metrics_block on {n}-bar metrics", False, f"KeyError {e}")

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'}")
sys.exit(1 if failures else 0)
