"""Platform Evaluation Test Suite.

Runs realistic end-to-end tests across all platform capabilities:
- News data fetching (volume, freshness, relevance, source diversity)
- LLM analysis quality (sentiment, recommendations, coherence)
- Prediction models (Kronos, XGBoost, LLM Agent)
- Stock data & technical indicators
- Overall usefulness assessment
"""

import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Helpers ──────────────────────────────────────────────────────────────

PASS = "✅"
WARN = "⚠️"
FAIL = "❌"
INFO = "ℹ️"

results = []


def log(icon, category, message, detail=""):
    entry = {"icon": icon, "category": category, "message": message, "detail": detail}
    results.append(entry)
    detail_str = f"  → {detail}" if detail else ""
    print(f"  {icon} [{category}] {message}{detail_str}")


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── Test Symbols ─────────────────────────────────────────────────────────

# Mix of mega-cap (high news volume), mid-cap, and volatile stocks
TEST_SYMBOLS = ["AAPL", "NVDA", "TSLA", "JPM"]


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: NEWS DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════

def test_news_fetching():
    section("TEST 1: NEWS DATA FETCHING")
    from services.news_service import (
        fetch_news,
        fetch_alpha_vantage_news,
        fetch_yfinance_news,
        get_sentiment_summary,
    )

    for symbol in TEST_SYMBOLS:
        print(f"\n  --- {symbol} ---")

        # 1a. Alpha Vantage news
        t0 = time.time()
        av_articles = fetch_alpha_vantage_news(symbol, max_articles=50)
        av_time = time.time() - t0

        if av_articles:
            log(PASS, "AV-fetch", f"{symbol}: {len(av_articles)} articles in {av_time:.1f}s")

            # Freshness: how old is the newest article?
            newest = max(a.published_at for a in av_articles)
            age_hours = (datetime.now() - newest).total_seconds() / 3600
            if age_hours < 24:
                log(PASS, "AV-fresh", f"Newest article: {age_hours:.1f}h ago")
            elif age_hours < 72:
                log(WARN, "AV-fresh", f"Newest article: {age_hours:.1f}h ago (slightly stale)")
            else:
                log(FAIL, "AV-fresh", f"Newest article: {age_hours:.1f}h ago (STALE)")

            # Source diversity
            sources = set(a.source for a in av_articles)
            if len(sources) >= 3:
                log(PASS, "AV-sources", f"{len(sources)} unique sources", ", ".join(list(sources)[:5]))
            else:
                log(WARN, "AV-sources", f"Only {len(sources)} sources — low diversity")

            # Sentiment distribution
            sentiments = Counter(a.sentiment for a in av_articles)
            log(INFO, "AV-sentiment", f"Distribution: {dict(sentiments)}")

            # Relevance scores
            relevances = [a.ticker_relevance_score for a in av_articles if a.ticker_relevance_score]
            if relevances:
                avg_rel = sum(relevances) / len(relevances)
                log(
                    PASS if avg_rel > 0.6 else WARN,
                    "AV-relevance",
                    f"Avg relevance: {avg_rel:.2f} (min={min(relevances):.2f}, max={max(relevances):.2f})",
                )

            # Summaries available?
            has_summary = sum(1 for a in av_articles if a.summary)
            log(
                PASS if has_summary > len(av_articles) * 0.5 else WARN,
                "AV-summaries",
                f"{has_summary}/{len(av_articles)} articles have summaries",
            )

            # Impact extraction
            has_impact = sum(1 for a in av_articles if a.impact)
            log(INFO, "AV-impact", f"{has_impact}/{len(av_articles)} articles have extracted impact tags")

        else:
            log(FAIL, "AV-fetch", f"{symbol}: No articles returned", "Check ALPHA_VANTAGE_API_KEY")

        # 1b. yfinance fallback
        t0 = time.time()
        yf_articles = fetch_yfinance_news(symbol, max_articles=10)
        yf_time = time.time() - t0

        if yf_articles:
            log(PASS, "YF-fetch", f"{symbol}: {len(yf_articles)} yfinance articles in {yf_time:.1f}s")
        else:
            log(WARN, "YF-fetch", f"{symbol}: No yfinance articles (fallback unavailable)")

        # 1c. Sentiment summary accuracy
        all_articles = av_articles or yf_articles
        if all_articles:
            summary = get_sentiment_summary(all_articles)
            total = summary["total"]
            if total > 0:
                log(
                    PASS,
                    "Sentiment-summary",
                    f"Overall: {summary['overall']} (score={summary['score']:.3f}), "
                    f"B={summary['bullish']} N={summary['neutral']} Be={summary['bearish']}",
                )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: LLM ANALYSIS QUALITY
# ═══════════════════════════════════════════════════════════════════════════

def test_llm_analysis():
    section("TEST 2: LLM ANALYSIS QUALITY")
    from services.llm_service import get_llm
    from services.news_service import fetch_news

    llm = get_llm()

    # Check provider
    info = llm.get_provider_info()
    if llm.is_available():
        log(PASS, "LLM-provider", f"Provider: {info['provider']}")
    else:
        log(FAIL, "LLM-provider", "No LLM provider available — skipping LLM tests")
        return

    for symbol in TEST_SYMBOLS[:2]:  # Test 2 symbols to save API costs
        print(f"\n  --- {symbol} LLM Analysis ---")

        articles = fetch_news(symbol, max_articles=10)
        if not articles:
            log(WARN, "LLM-input", f"{symbol}: No articles to analyze")
            continue

        article_dicts = [
            {"title": a.title, "summary": a.summary or "", "sentiment": a.sentiment or "neutral"}
            for a in articles
        ]

        # 2a. Structured analysis
        t0 = time.time()
        structured = llm.summarize_news_structured(article_dicts, [symbol])
        llm_time = time.time() - t0

        if structured:
            log(PASS, "LLM-structured", f"{symbol}: Got structured analysis in {llm_time:.1f}s")

            # Validate recommendation
            rec = structured.get("recommendation", "")
            valid_recs = ["BULLISH", "CAUTIOUS_BULLISH", "NEUTRAL", "CAUTIOUS_BEARISH", "BEARISH"]
            if rec in valid_recs:
                log(PASS, "LLM-rec", f"Recommendation: {rec}")
            else:
                log(FAIL, "LLM-rec", f"Invalid recommendation: {rec}")

            # Validate confidence
            conf = structured.get("confidence", -1)
            if 0 <= conf <= 1:
                log(PASS, "LLM-conf", f"Confidence: {conf:.2f}")
            else:
                log(FAIL, "LLM-conf", f"Invalid confidence: {conf}")

            # Key developments quality
            kd = structured.get("key_developments", "")
            if len(kd) > 30:
                log(PASS, "LLM-kd", f"Key developments: {len(kd)} chars", kd[:120] + "...")
            else:
                log(WARN, "LLM-kd", f"Key developments too short: {len(kd)} chars")

            # Sentiment explanation
            se = structured.get("sentiment_explanation", "")
            if len(se) > 15:
                log(PASS, "LLM-sent-expl", f"Sentiment explanation: {len(se)} chars")
            else:
                log(WARN, "LLM-sent-expl", f"Explanation too short: {len(se)} chars")

            # Cross-check: does LLM recommendation align with raw sentiment?
            bullish_count = sum(1 for a in articles if a.sentiment == "bullish")
            bearish_count = sum(1 for a in articles if a.sentiment == "bearish")
            raw_lean = "bullish" if bullish_count > bearish_count else ("bearish" if bearish_count > bullish_count else "neutral")
            rec_lean = "bullish" if "BULLISH" in rec else ("bearish" if "BEARISH" in rec else "neutral")

            if raw_lean == rec_lean:
                log(PASS, "LLM-alignment", f"LLM rec ({rec}) aligns with raw sentiment ({raw_lean})")
            else:
                log(WARN, "LLM-alignment",
                    f"LLM rec ({rec}) differs from raw sentiment ({raw_lean}: {bullish_count}B/{bearish_count}Be)",
                    "LLM may be applying deeper reasoning — not necessarily wrong")
        else:
            log(FAIL, "LLM-structured", f"{symbol}: Structured analysis returned None")

        # 2b. Free-form summary
        t0 = time.time()
        summary = llm.summarize_news(article_dicts, symbol)
        sum_time = time.time() - t0

        if summary:
            log(PASS, "LLM-summary", f"{symbol}: Free-form summary in {sum_time:.1f}s ({len(summary)} chars)")

            # Check for markdown structure
            has_headers = "###" in summary or "**" in summary
            if has_headers:
                log(PASS, "LLM-format", "Summary uses proper markdown formatting")
            else:
                log(WARN, "LLM-format", "Summary lacks markdown structure")

            # Check for hallucination red flags (generic filler)
            filler_phrases = ["it is important to note", "investors should consider", "as always, do your own research"]
            fillers_found = [p for p in filler_phrases if p.lower() in summary.lower()]
            if fillers_found:
                log(WARN, "LLM-filler", f"Contains generic filler: {fillers_found}")
            else:
                log(PASS, "LLM-filler", "No generic filler detected")
        else:
            log(FAIL, "LLM-summary", f"{symbol}: Free-form summary returned None")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: PREDICTION MODELS
# ═══════════════════════════════════════════════════════════════════════════

def test_prediction_models():
    section("TEST 3: PREDICTION MODELS")
    from services.stock_data import fetch_stock_data

    # Import individual models to test them in isolation
    models_to_test = []

    try:
        from models.kronos_model import KronosModel
        models_to_test.append(("kronos_mini", KronosModel()))
        log(PASS, "Model-load", "Kronos loaded")
    except Exception as e:
        log(FAIL, "Model-load", f"Kronos failed to load: {e}")

    try:
        from models.xgboost_model import XGBoostModel
        models_to_test.append(("xgboost_shap", XGBoostModel()))
        log(PASS, "Model-load", "XGBoost loaded")
    except Exception as e:
        log(FAIL, "Model-load", f"XGBoost failed to load: {e}")

    # Fetch data for test
    test_symbol = "AAPL"
    try:
        ohlcv = fetch_stock_data(test_symbol, period="1y")
        spy_df = fetch_stock_data("SPY", period="1y")
        log(PASS, "Data-fetch", f"{test_symbol}: {len(ohlcv)} bars, SPY: {len(spy_df)} bars")
    except Exception as e:
        log(FAIL, "Data-fetch", f"Failed to fetch stock data: {e}")
        return

    # Fetch news for models that need it
    from services.news_service import fetch_news
    news_articles = fetch_news(test_symbol, max_articles=20)
    news_dicts = [
        {"title": a.title, "summary": a.summary or "", "sentiment": a.sentiment or "neutral",
         "published_at": a.published_at, "source": a.source, "sentiment_score": a.sentiment_score or 0}
        for a in news_articles
    ]

    for model_name, model in models_to_test:
        print(f"\n  --- {model_name} ---")

        t0 = time.time()
        try:
            kwargs = {}
            if model_name == "xgboost_shap":
                kwargs = {"spy_df": spy_df, "av_news": news_dicts, "global_news": [],
                          "historical_av_news": {}, "historical_global_news": {}}

            result = model.predict(test_symbol, ohlcv, **kwargs)
            pred_time = time.time() - t0

            log(PASS, f"{model_name}", f"Prediction: {result.decision} (conf={result.confidence:.2f}, p_up={result.up_probability:.2f}) in {pred_time:.1f}s")

            # Validate decision
            if result.decision in ("BUY", "SELL", "HOLD"):
                log(PASS, f"{model_name}-valid", f"Valid decision: {result.decision}")
            else:
                log(FAIL, f"{model_name}-valid", f"Invalid decision: {result.decision}")

            # Confidence sanity check
            if 0 <= result.confidence <= 1:
                log(PASS, f"{model_name}-conf", f"Confidence in valid range: {result.confidence:.3f}")
            else:
                log(FAIL, f"{model_name}-conf", f"Confidence out of range: {result.confidence}")

            # Check for errors
            if result.error:
                log(WARN, f"{model_name}-err", f"Model reported error: {result.error}")

            # Model-specific checks
            if model_name == "kronos_mini" and result.predicted_close:
                current_price = ohlcv["Close"].iloc[-1]
                pct_diff = abs(result.predicted_close - current_price) / current_price * 100
                log(
                    PASS if pct_diff < 20 else WARN,
                    f"{model_name}-price",
                    f"Predicted close: ${result.predicted_close:.2f} (current: ${current_price:.2f}, diff: {pct_diff:.1f}%)",
                )

            if result.details:
                detail_keys = list(result.details.keys())
                log(INFO, f"{model_name}-details", f"Details keys: {detail_keys}")

        except Exception as e:
            pred_time = time.time() - t0
            log(FAIL, f"{model_name}", f"Prediction failed in {pred_time:.1f}s: {e}")
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: STOCK DATA & TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def test_stock_data_and_indicators():
    section("TEST 4: STOCK DATA & TECHNICAL INDICATORS")
    from services.stock_data import fetch_stock_data, get_stock_info, validate_symbol
    from services.analytics import add_indicators_to_df, get_latest_signals, calculate_performance_metrics

    # 4a. Symbol validation
    for sym, expected in [("AAPL", True), ("XYZXYZXYZ", False)]:
        valid = validate_symbol(sym)
        if valid == expected:
            log(PASS, "Validation", f"{sym}: valid={valid} (expected {expected})")
        else:
            log(FAIL, "Validation", f"{sym}: valid={valid} (expected {expected})")

    # 4b. Stock data fetch
    for symbol in TEST_SYMBOLS[:2]:
        print(f"\n  --- {symbol} Data Quality ---")

        try:
            df = fetch_stock_data(symbol, period="1y")

            # Data completeness
            expected_cols = {"Open", "High", "Low", "Close", "Volume"}
            missing = expected_cols - set(df.columns)
            if not missing:
                log(PASS, "Data-cols", f"{symbol}: All OHLCV columns present")
            else:
                log(FAIL, "Data-cols", f"{symbol}: Missing columns: {missing}")

            # Row count sanity
            if len(df) >= 200:
                log(PASS, "Data-rows", f"{symbol}: {len(df)} trading days (1y)")
            else:
                log(WARN, "Data-rows", f"{symbol}: Only {len(df)} rows — expected ~250 for 1y")

            # NaN check
            nan_pct = df[list(expected_cols)].isna().mean()
            worst_nan = nan_pct.max()
            if worst_nan < 0.01:
                log(PASS, "Data-nan", f"{symbol}: Max NaN rate: {worst_nan:.1%}")
            else:
                log(WARN, "Data-nan", f"{symbol}: NaN rates: {nan_pct.to_dict()}")

            # Freshness
            last_date = df.index[-1]
            days_stale = (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days
            if days_stale <= 3:  # weekend buffer
                log(PASS, "Data-fresh", f"{symbol}: Latest data: {last_date.date()} ({days_stale}d ago)")
            else:
                log(WARN, "Data-fresh", f"{symbol}: Latest data: {last_date.date()} ({days_stale}d ago — stale)")

            # 4c. Technical indicators
            df_ind = add_indicators_to_df(df)
            indicator_cols = [c for c in df_ind.columns if c not in df.columns]
            log(INFO, "Indicators", f"{symbol}: {len(indicator_cols)} indicators added", ", ".join(indicator_cols[:10]))

            # Check key indicators are computed. Names must match what
            # add_indicators_to_df actually emits ("RSI", not "RSI_14") —
            # the old list made this check permanently WARN, which is how
            # an RSI formula drift went unnoticed.
            key_indicators = ["SMA_20", "SMA_50", "RSI", "MACD"]
            for ind in key_indicators:
                matches = [c for c in indicator_cols if ind.lower() in c.lower()]
                if matches:
                    last_val = df_ind[matches[0]].dropna().iloc[-1] if not df_ind[matches[0]].dropna().empty else None
                    # `is not None`: a legitimate 0.0 (e.g. MACD at crossover)
                    # must not report as "computed but empty".
                    log(PASS, "Indicator", f"{ind}: {last_val:.2f}" if last_val is not None else f"{ind}: computed but empty")
                else:
                    log(WARN, "Indicator", f"{ind}: not found in output columns")

            # 4d. Signals
            signals = get_latest_signals(df_ind)
            if signals:
                log(PASS, "Signals", f"{symbol}: {len(signals)} signal groups")
                for key, val in list(signals.items())[:3]:
                    if isinstance(val, dict):
                        log(INFO, "Signal", f"  {key}: {val.get('signal', val)}")
            else:
                log(WARN, "Signals", f"{symbol}: No signals generated")

            # 4e. Performance metrics
            metrics = calculate_performance_metrics(df)
            if metrics:
                log(PASS, "Perf-metrics", f"{symbol}: return={metrics.get('total_return', 'N/A')}%, "
                    f"vol={metrics.get('volatility', 'N/A')}%, maxDD={metrics.get('max_drawdown', 'N/A')}%")
            else:
                log(WARN, "Perf-metrics", f"{symbol}: No performance metrics")

            # 4f. Stock info
            info = get_stock_info(symbol)
            log(PASS, "Stock-info", f"{symbol}: {info.name}, ${info.current_price:.2f}, "
                f"sector={info.sector}, change={info.day_change_percent:+.2f}%")

        except Exception as e:
            log(FAIL, "Data", f"{symbol}: {e}")
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_summary():
    section("EVALUATION SUMMARY")

    total = len(results)
    passes = sum(1 for r in results if r["icon"] == PASS)
    warns = sum(1 for r in results if r["icon"] == WARN)
    fails = sum(1 for r in results if r["icon"] == FAIL)
    infos = sum(1 for r in results if r["icon"] == INFO)

    print(f"\n  Total checks: {total}")
    print(f"  {PASS} Passed:  {passes}")
    print(f"  {WARN} Warnings: {warns}")
    print(f"  {FAIL} Failed:  {fails}")
    print(f"  {INFO} Info:    {infos}")

    score = (passes / (passes + warns + fails)) * 100 if (passes + warns + fails) > 0 else 0
    print(f"\n  Platform Health Score: {score:.0f}%")

    if fails > 0:
        print(f"\n  {FAIL} FAILURES:")
        for r in results:
            if r["icon"] == FAIL:
                print(f"    - [{r['category']}] {r['message']}")

    if warns > 0:
        print(f"\n  {WARN} WARNINGS:")
        for r in results:
            if r["icon"] == WARN:
                print(f"    - [{r['category']}] {r['message']}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'#'*70}")
    print(f"  QUANT-NEWS PLATFORM EVALUATION")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Symbols: {', '.join(TEST_SYMBOLS)}")
    print(f"{'#'*70}")

    try:
        test_news_fetching()
    except Exception as e:
        log(FAIL, "NEWS-TEST", f"Test crashed: {e}")
        traceback.print_exc()

    try:
        test_llm_analysis()
    except Exception as e:
        log(FAIL, "LLM-TEST", f"Test crashed: {e}")
        traceback.print_exc()

    try:
        test_stock_data_and_indicators()
    except Exception as e:
        log(FAIL, "DATA-TEST", f"Test crashed: {e}")
        traceback.print_exc()

    try:
        test_prediction_models()
    except Exception as e:
        log(FAIL, "MODEL-TEST", f"Test crashed: {e}")
        traceback.print_exc()

    print_summary()
