#!/usr/bin/env python3
"""Test the Postgres-backed CacheService.

Run with Docker infra up:
    docker compose up -d postgres minio
    DATABASE_URL=postgresql://quantnews:quantnews@localhost:5433/quantnews \
        python scripts/test_cache_postgres.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("DATABASE_URL", "postgresql://quantnews:quantnews@localhost:5433/quantnews")

from dotenv import load_dotenv
load_dotenv()

# Never against a shared database: these tests write junk rows (symbol TEST)
# and load_dotenv() above happily points them at whatever .env names --
# which is how a TEST symbol ended up in production stock_prices.
_db = os.environ.get("DATABASE_URL", "")
if "localhost" not in _db and "127.0.0.1" not in _db:
    raise SystemExit(f"Refusing to run against non-local DATABASE_URL: {_db.split('@')[-1]}")

from datetime import date, datetime
from services.cache_service import get_cache, CacheService


def test_cache_metadata():
    print("1. Testing cache metadata...")
    cache = get_cache()
    assert not cache.is_cached("TEST", "prices")
    print("   ✓ is_cached returns False for empty DB")


def test_news_cache():
    print("2. Testing news cache...")
    cache = get_cache()

    cache.cache_news("TEST", [
        {"id": "n1", "title": "Test headline", "source": "Reuters",
         "url": "http://example.com/1", "published_at": datetime.now(),
         "summary": "Summary", "sentiment": "Bullish", "sentiment_score": 0.8,
         "topics": ["earnings"], "overall_sentiment_score": 0.7,
         "overall_sentiment_label": "Bullish", "ticker_relevance_score": 0.9,
         "impact": "HIGH"},
    ])
    print("   ✓ cache_news stored")

    cached = cache.get_cached_news("TEST", max_age_minutes=5)
    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["title"] == "Test headline"
    assert cached[0]["topics"] == ["earnings"]
    print("   ✓ get_cached_news returns correct data")


def test_prediction_store():
    print("3. Testing prediction store/retrieve...")
    cache = get_cache()

    cache.store_prediction("TEST", "kronos_mini", {
        "decision": "SELL",
        "confidence": 0.85,
        "up_probability": 0.15,
        "details": {"sample_count": 20},
    }, prediction_date_str="2026-07-13")
    print("   ✓ store_prediction OK")

    today_preds = cache.get_predictions_for_today("TEST")
    # May or may not have today's predictions depending on date
    print(f"   ✓ get_predictions_for_today returned {len(today_preds)} results")

    history = cache.get_prediction_history("TEST", "kronos_mini", limit=5)
    assert len(history) >= 1
    assert history[0]["decision"] == "SELL"
    print(f"   ✓ get_prediction_history: {len(history)} records")


def test_model_accuracy():
    print("4. Testing model accuracy (empty evaluations)...")
    cache = get_cache()
    acc = cache.get_model_accuracy("kronos_mini")
    assert acc["total"] == 0  # No evaluations yet
    print(f"   ✓ get_model_accuracy: total={acc['total']}, accuracy={acc['accuracy']}")


def test_strategy_operations():
    print("5. Testing strategy operations...")
    cache = get_cache()

    # Store a strategy evaluation
    count = cache.store_strategy_evaluations([{
        "id": "TEST_kronos_mini_20260713_directional",
        "prediction_id": "TEST_kronos_mini_20260713",
        "strategy_name": "directional",
        "strategy_version": "v1",
        "action": "SELL",
        "position_size": 1000.0,
        "entry_price": 100.0,
        "exit_price": 95.0,
        "pnl_dollars": 50.0,
        "was_correct": True,
        "metadata": {"signal": "strong_sell"},
    }])
    assert count == 1
    print("   ✓ store_strategy_evaluations OK")

    evals = cache.get_strategy_evaluations("directional", symbol="TEST")
    assert len(evals) >= 1
    print(f"   ✓ get_strategy_evaluations: {len(evals)} records")

    symbols = cache.get_strategy_symbols("directional")
    assert "TEST" in symbols
    print(f"   ✓ get_strategy_symbols: {symbols}")


def test_strategy_metrics():
    print("6. Testing strategy metrics...")
    cache = get_cache()

    cache.store_strategy_metrics("directional", "TEST", "all", {
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.1,
        "max_drawdown": -0.15,
        "win_rate": 0.62,
        "total_pnl": 500.0,
        "total_trades": 20,
    })
    print("   ✓ store_strategy_metrics OK")

    metrics = cache.get_strategy_metrics(strategy_name="directional", symbol="TEST")
    assert len(metrics) >= 1
    assert metrics[0]["sharpe_ratio"] == 1.5
    print(f"   ✓ get_strategy_metrics: {len(metrics)} records")


def test_trading_agent_reports():
    print("7. Testing trading agent reports...")
    cache = get_cache()

    cache.save_trading_agent_report(
        symbol="TEST", trade_date="2026-07-13",
        decision="SELL", confidence=0.9,
        report_text="# Detailed Analysis\n\nSELL recommendation based on...",
        model_name="gpt-4o", input_tokens=1000, output_tokens=500,
    )
    print("   ✓ save_trading_agent_report OK")

    reports = cache.get_trading_agent_reports("TEST", limit=5)
    assert len(reports) >= 1
    assert reports[0]["decision"] == "SELL"
    print(f"   ✓ get_trading_agent_reports: {len(reports)} records")

    all_reports = cache.get_all_trading_agent_reports(limit=10)
    assert len(all_reports) >= 1
    print(f"   ✓ get_all_trading_agent_reports: {len(all_reports)} records")


def test_historical_news():
    print("8. Testing historical news...")
    cache = get_cache()

    count = cache.store_historical_news([{
        "id": "hist1",
        "symbol": "TEST",
        "published_date": "2026-06-01",
        "title": "Historical headline",
        "summary": "Old news",
        "url": "http://example.com/old",
        "source": "AP",
        "topics": ["tech"],
        "overall_sentiment_score": 0.5,
        "overall_sentiment_label": "Neutral",
        "ticker_sentiment_score": 0.4,
        "ticker_relevance_score": 0.6,
    }])
    assert count == 1
    print("   ✓ store_historical_news OK")

    articles = cache.get_historical_news("TEST", start_date="2026-01-01")
    assert len(articles) >= 1
    print(f"   ✓ get_historical_news: {len(articles)} articles")


def test_cache_status():
    print("9. Testing cache status...")
    cache = get_cache()
    status = cache.get_cache_status()
    assert len(status) >= 1  # news was cached above
    print(f"   ✓ get_cache_status: {len(status)} entries")

    symbols = cache.get_all_cached_symbols()
    print(f"   ✓ get_all_cached_symbols: {symbols}")


def test_clear():
    print("10. Testing clear operations...")
    cache = get_cache()
    cache.clear_symbol("TEST")
    assert not cache.is_cached("TEST", "prices")
    assert not cache.is_cached("TEST", "news")
    print("   ✓ clear_symbol OK")


def main():
    print("=" * 60)
    print("CacheService (Postgres) — Integration Test")
    print("=" * 60)

    tests = [
        test_cache_metadata,
        test_news_cache,
        test_prediction_store,
        test_model_accuracy,
        test_strategy_operations,
        test_strategy_metrics,
        test_trading_agent_reports,
        test_historical_news,
        test_cache_status,
        test_clear,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
