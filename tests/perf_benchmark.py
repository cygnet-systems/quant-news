"""Performance benchmark for quant-news platform.

Measures latency and memory for key operations:
  1. Stock data fetch (yfinance) - 1, 4, 8 symbols
  2. News fetch (Alpha Vantage) - 1, 4 symbols
  3. Technical indicator computation
  4. LLM structured analysis (Anthropic)
  5. Model predictions (Kronos, XGBoost)
  6. Cold start vs warm (cache hit)
  7. Memory usage (RSS delta)

Each test runs 3 iterations; reports min/avg/max plus pass/warn/fail.
"""

import gc
import os
import sys
import time
import traceback

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Disable HuggingFace online lookups if weights are cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import psutil

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ITERATIONS = 3

# Symbols used in various tests
SYMBOL_1 = ["AAPL"]
SYMBOL_4 = ["AAPL", "MSFT", "GOOGL", "NVDA"]
SYMBOL_8 = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "JPM"]


def _rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _run_timed(fn, iterations=ITERATIONS):
    """Run *fn* multiple times, return (min, avg, max) in seconds."""
    times = []
    result = None
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    mn = min(times)
    mx = max(times)
    avg = sum(times) / len(times)
    return mn, avg, mx, result


def _status(avg_s, pass_s, warn_s):
    """Return PASS / WARN / FAIL label."""
    if avg_s <= pass_s:
        return "PASS"
    elif avg_s <= warn_s:
        return "WARN"
    else:
        return "FAIL"


def _fmt(seconds):
    """Format seconds for display."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

results = []  # (name, min, avg, max, status)
notes = []    # free-form observations


def record(name, mn, avg, mx, pass_s, warn_s):
    status = _status(avg, pass_s, warn_s)
    results.append((name, mn, avg, mx, status))
    print(f"  {name:.<50s} {_fmt(mn):>8s} / {_fmt(avg):>8s} / {_fmt(mx):>8s}  [{status}]")


# ---- 1. Stock data fetch ---------------------------------------------------

def bench_stock_fetch():
    from services.stock_data import fetch_stock_data, clear_ticker_cache

    print("\n=== 1. Stock Data Fetch (yfinance, 1y OHLCV) ===")
    print(f"  {'Test':.<50s} {'Min':>8s} / {'Avg':>8s} / {'Max':>8s}  Status")
    print("  " + "-" * 80)

    # 1 symbol
    clear_ticker_cache()
    mn, avg, mx, _ = _run_timed(lambda: fetch_stock_data("AAPL", period="1y"))
    record("1 symbol (AAPL)", mn, avg, mx, pass_s=2, warn_s=5)

    # 4 symbols sequential
    clear_ticker_cache()
    def fetch_4():
        out = {}
        for s in SYMBOL_4:
            out[s] = fetch_stock_data(s, period="1y")
        return out
    mn, avg, mx, _ = _run_timed(fetch_4)
    record("4 symbols sequential", mn, avg, mx, pass_s=8, warn_s=20)

    # 8 symbols sequential
    clear_ticker_cache()
    def fetch_8():
        out = {}
        for s in SYMBOL_8:
            out[s] = fetch_stock_data(s, period="1y")
        return out
    mn, avg, mx, _ = _run_timed(fetch_8)
    record("8 symbols sequential", mn, avg, mx, pass_s=16, warn_s=40)

    notes.append(
        "Stock fetch is sequential. ThreadPoolExecutor would cut 4-sym wall time ~4x "
        "(yfinance uses HTTP I/O, no GIL contention)."
    )


# ---- 2. News fetch (Alpha Vantage) -----------------------------------------

def bench_news_fetch():
    from config import API
    from services.news_service import fetch_alpha_vantage_news

    print("\n=== 2. News Fetch (Alpha Vantage) ===")

    if not API.ALPHA_VANTAGE_API_KEY:
        print("  SKIPPED - ALPHA_VANTAGE_API_KEY not set")
        results.append(("News: 1 symbol (AV)", 0, 0, 0, "SKIP"))
        results.append(("News: 4 symbols (AV)", 0, 0, 0, "SKIP"))
        return

    print(f"  {'Test':.<50s} {'Min':>8s} / {'Avg':>8s} / {'Max':>8s}  Status")
    print("  " + "-" * 80)

    # 1 symbol
    mn, avg, mx, _ = _run_timed(lambda: fetch_alpha_vantage_news("AAPL"))
    record("News: 1 symbol (AAPL)", mn, avg, mx, pass_s=2, warn_s=5)

    # 4 symbols sequential
    def fetch_news_4():
        out = {}
        for s in SYMBOL_4:
            out[s] = fetch_alpha_vantage_news(s)
        return out
    mn, avg, mx, _ = _run_timed(fetch_news_4)
    record("News: 4 symbols sequential", mn, avg, mx, pass_s=8, warn_s=20)


# ---- 3. Technical indicators -----------------------------------------------

def bench_indicators():
    from services.stock_data import fetch_stock_data
    from services.analytics import calculate_all_indicators, add_indicators_to_df

    print("\n=== 3. Technical Indicator Computation (1y data) ===")
    print(f"  {'Test':.<50s} {'Min':>8s} / {'Avg':>8s} / {'Max':>8s}  Status")
    print("  " + "-" * 80)

    # Pre-fetch data once (don't measure network time here)
    df = fetch_stock_data("AAPL", period="1y")

    # calculate_all_indicators
    mn, avg, mx, _ = _run_timed(lambda: calculate_all_indicators(df))
    record("calculate_all_indicators", mn, avg, mx, pass_s=0.5, warn_s=1.0)

    # add_indicators_to_df (includes copy + assignment)
    mn, avg, mx, _ = _run_timed(lambda: add_indicators_to_df(df))
    record("add_indicators_to_df", mn, avg, mx, pass_s=0.5, warn_s=1.0)


# ---- 4. LLM structured analysis --------------------------------------------

def bench_llm():
    from config import API
    from services.llm_service import LLMService

    print("\n=== 4. LLM Structured Analysis ===")

    svc = LLMService()
    info = svc.get_provider_info()

    # Skip if only LM Studio is available (user request) or no provider
    if not svc.is_available():
        print("  SKIPPED - No LLM provider available")
        results.append(("LLM structured analysis", 0, 0, 0, "SKIP"))
        return

    if svc.provider == "lm_studio":
        print("  SKIPPED - LM Studio only (user requested skip)")
        results.append(("LLM structured analysis", 0, 0, 0, "SKIP"))
        return

    print(f"  Provider: {svc.provider}")
    print(f"  {'Test':.<50s} {'Min':>8s} / {'Avg':>8s} / {'Max':>8s}  Status")
    print("  " + "-" * 80)

    # Build fake articles for the call
    fake_articles = [
        {"title": "Apple beats Q3 earnings estimates", "summary": "Revenue up 8% YoY.", "sentiment": "bullish"},
        {"title": "iPhone sales slow in China", "summary": "Market share down 2%.", "sentiment": "bearish"},
        {"title": "Apple unveils new AI features", "summary": "Apple Intelligence launches.", "sentiment": "bullish"},
        {"title": "Analyst raises AAPL price target to $250", "summary": "Morgan Stanley upgrade.", "sentiment": "bullish"},
        {"title": "Apple supply chain diversifies", "summary": "India production ramps up.", "sentiment": "neutral"},
    ]

    mn, avg, mx, _ = _run_timed(
        lambda: svc.summarize_news_structured(fake_articles, ["AAPL"])
    )
    record("LLM structured analysis (Anthropic)", mn, avg, mx, pass_s=5, warn_s=10)


# ---- 5. Model predictions ---------------------------------------------------

def bench_models():
    print("\n=== 5. Model Prediction Latency ===")
    print(f"  {'Test':.<50s} {'Min':>8s} / {'Avg':>8s} / {'Max':>8s}  Status")
    print("  " + "-" * 80)

    from services.stock_data import fetch_stock_data

    # Pre-fetch data
    ohlcv = fetch_stock_data("AAPL", period="1y")

    # --- Kronos ---
    try:
        from models.kronos_model import KronosModel, KRONOS_AVAILABLE
        if not KRONOS_AVAILABLE:
            raise ImportError("Kronos deps not available")
        km = KronosModel()
        mn, avg, mx, pred = _run_timed(lambda: km.predict("AAPL", ohlcv))
        if pred and pred.error:
            print(f"  Kronos returned error: {pred.error}")
            results.append(("Kronos predict (AAPL)", mn, avg, mx, "ERR"))
        else:
            record("Kronos predict (AAPL)", mn, avg, mx, pass_s=10, warn_s=20)
            if pred:
                notes.append(f"Kronos result: {pred.decision} conf={pred.confidence:.2f}")
    except Exception as e:
        print(f"  Kronos SKIPPED: {e}")
        results.append(("Kronos predict (AAPL)", 0, 0, 0, "SKIP"))

    # --- XGBoost ---
    try:
        from config import API
        if not API.ALPHA_VANTAGE_API_KEY:
            print("  XGBoost SKIPPED - needs ALPHA_VANTAGE_API_KEY for features")
            results.append(("XGBoost predict (AAPL)", 0, 0, 0, "SKIP"))
        else:
            from models.xgboost_model import XGBoostModel
            spy_df = fetch_stock_data("SPY", period="1y")
            xm = XGBoostModel()

            # XGBoost needs AV news for features; pass empty to measure pure model time
            mn, avg, mx, pred = _run_timed(
                lambda: xm.predict(
                    "AAPL", ohlcv,
                    spy_df=spy_df,
                    sector_df=spy_df,  # sector fallback
                    av_news=[],
                    global_news=[],
                    historical_av_news={},
                    historical_global_news={},
                )
            )
            if pred and pred.error:
                print(f"  XGBoost returned error: {pred.error}")
                results.append(("XGBoost predict (AAPL)", mn, avg, mx, "ERR"))
            else:
                record("XGBoost predict (AAPL)", mn, avg, mx, pass_s=10, warn_s=20)
                if pred:
                    notes.append(f"XGBoost result: {pred.decision} conf={pred.confidence:.2f}")
    except Exception as e:
        print(f"  XGBoost SKIPPED: {e}")
        traceback.print_exc()
        results.append(("XGBoost predict (AAPL)", 0, 0, 0, "SKIP"))


# ---- 6. Cold start vs warm (cache) -----------------------------------------

def bench_cache():
    from services.stock_data import fetch_stock_data, clear_ticker_cache

    print("\n=== 6. Cold Start vs Warm (yfinance cache) ===")
    print(f"  {'Test':.<50s} {'Min':>8s} / {'Avg':>8s} / {'Max':>8s}  Status")
    print("  " + "-" * 80)

    # Cold: clear cache, single fetch
    clear_ticker_cache()
    mn, avg, mx, _ = _run_timed(lambda: fetch_stock_data("TSLA", period="1y"), iterations=1)
    record("Cold fetch (TSLA, 1y)", mn, avg, mx, pass_s=2, warn_s=5)

    # Warm: ticker object cached (yfinance still hits network for history though)
    mn, avg, mx, _ = _run_timed(lambda: fetch_stock_data("TSLA", period="1y"), iterations=1)
    record("Warm fetch (TSLA, 1y, cached ticker)", mn, avg, mx, pass_s=2, warn_s=5)


# ---- 7. Memory usage -------------------------------------------------------

def bench_memory():
    print("\n=== 7. Memory Usage (RSS) ===")

    rss_start = _rss_mb()
    print(f"  Baseline RSS: {rss_start:.1f} MB")

    # Load models and measure delta
    from services.stock_data import fetch_stock_data

    # Fetch data for several symbols
    data = {}
    for s in SYMBOL_4:
        data[s] = fetch_stock_data(s, period="1y")
    rss_after_data = _rss_mb()
    data_delta = rss_after_data - rss_start
    print(f"  After fetching 4 symbols (1y): {rss_after_data:.1f} MB (delta: +{data_delta:.1f} MB)")

    # Load Kronos model
    try:
        from models.kronos_model import _get_predictor, KRONOS_AVAILABLE
        if KRONOS_AVAILABLE:
            _get_predictor()
            rss_after_kronos = _rss_mb()
            kronos_delta = rss_after_kronos - rss_after_data
            print(f"  After loading Kronos: {rss_after_kronos:.1f} MB (delta: +{kronos_delta:.1f} MB)")
        else:
            rss_after_kronos = rss_after_data
            print("  Kronos not available - skipped")
    except Exception as e:
        rss_after_kronos = rss_after_data
        print(f"  Kronos load failed: {e}")

    # Compute indicators on all 4
    from services.analytics import add_indicators_to_df
    enriched = {}
    for s, df in data.items():
        enriched[s] = add_indicators_to_df(df)
    rss_after_indicators = _rss_mb()
    ind_delta = rss_after_indicators - rss_after_kronos
    print(f"  After computing indicators (4 sym): {rss_after_indicators:.1f} MB (delta: +{ind_delta:.1f} MB)")

    rss_final = _rss_mb()
    total_delta = rss_final - rss_start
    print(f"  Final RSS: {rss_final:.1f} MB (total delta: +{total_delta:.1f} MB)")

    status = _status(rss_final, 500, 1024)
    results.append(("Total RSS", 0, rss_final, 0, status))
    print(f"  Status: [{status}] (threshold: <500MB pass, <1GB warn)")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary():
    print("\n")
    print("=" * 90)
    print("  PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"  {'Test':<45s} {'Min':>8s}  {'Avg':>8s}  {'Max':>8s}  {'Status':>6s}")
    print("  " + "-" * 82)

    pass_count = 0
    warn_count = 0
    fail_count = 0
    skip_count = 0
    err_count = 0

    for name, mn, avg, mx, status in results:
        if status == "SKIP":
            skip_count += 1
            print(f"  {name:<45s} {'--':>8s}  {'--':>8s}  {'--':>8s}  {'SKIP':>6s}")
        elif status == "ERR":
            err_count += 1
            print(f"  {name:<45s} {_fmt(mn):>8s}  {_fmt(avg):>8s}  {_fmt(mx):>8s}  {'ERR':>6s}")
        elif name == "Total RSS":
            # avg field holds the RSS value in MB
            rss_val = avg
            stat = status
            if stat == "PASS":
                pass_count += 1
            elif stat == "WARN":
                warn_count += 1
            else:
                fail_count += 1
            print(f"  {name:<45s} {'--':>8s}  {rss_val:>5.0f}MB  {'--':>8s}  {stat:>6s}")
        else:
            if status == "PASS":
                pass_count += 1
            elif status == "WARN":
                warn_count += 1
            else:
                fail_count += 1
            print(f"  {name:<45s} {_fmt(mn):>8s}  {_fmt(avg):>8s}  {_fmt(mx):>8s}  {status:>6s}")

    print("  " + "-" * 82)
    total = pass_count + warn_count + fail_count
    print(f"  Totals: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL, {skip_count} SKIP, {err_count} ERR  (of {total + skip_count + err_count} tests)")

    if notes:
        print("\n  Notes:")
        for n in notes:
            print(f"    - {n}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("  Quant-News Platform Performance Benchmark")
    print(f"  Python: {sys.executable}")
    print(f"  PID: {os.getpid()}, RSS at start: {_rss_mb():.1f} MB")
    print(f"  Iterations per test: {ITERATIONS}")
    print("=" * 90)

    bench_stock_fetch()
    bench_news_fetch()
    bench_indicators()
    bench_llm()
    bench_models()
    bench_cache()
    bench_memory()
    print_summary()


if __name__ == "__main__":
    main()
