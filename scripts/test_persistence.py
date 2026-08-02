#!/usr/bin/env python3
"""End-to-end test for the persistence layer.

Tests: Postgres connection, Alembic migrations, S3 upload/download,
cache invalidation logic (store, hit, miss after data change).

Run with Docker infra up:
    docker compose up -d postgres minio
    python scripts/test_persistence.py
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()


def test_db_connection():
    """Test Postgres connection."""
    print("1. Testing database connection...")
    from db.session import get_engine
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        assert result.scalar() == 1
    print("   ✓ Postgres connection OK")


def test_alembic_migrations():
    """Test that Alembic can reach 'head' (creates tables via migration)."""
    print("2. Testing Alembic migrations...")
    from alembic.config import Config
    from alembic import command
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", os.getenv(
        "DATABASE_URL",
        "postgresql://quantnews:quantnews@localhost:5432/quantnews",
    ))
    command.upgrade(alembic_cfg, "head")
    print("   ✓ Migrations applied (at head)")

    from db.session import get_engine
    engine = get_engine()
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    for expected in ["data_snapshots", "prediction_runs", "report_catalog", "recommendation_runs"]:
        assert expected in tables, f"Table {expected} not found"
    print(f"   ✓ All 4 tables verified: {', '.join(sorted(tables))}")


def test_s3_storage():
    """Test MinIO/S3 upload, download, list, delete."""
    print("3. Testing S3 object storage...")
    from services import storage_service

    storage_service.ensure_bucket()
    print("   ✓ Bucket exists")

    key = "test/hello.md"
    content = "# Test Report\n\nThis is a test report for EIX."
    size = storage_service.upload_report(key, content)
    assert size > 0
    print(f"   ✓ Upload OK ({size} bytes)")

    downloaded = storage_service.download_report(key)
    assert downloaded.decode("utf-8") == content
    print("   ✓ Download OK (content matches)")

    assert storage_service.report_exists(key)
    print("   ✓ Exists check OK")

    objects = storage_service.list_reports(prefix="test/")
    assert any(o["key"] == key for o in objects)
    print(f"   ✓ List OK ({len(objects)} objects)")

    storage_service.delete_report(key)
    assert not storage_service.report_exists(key)
    print("   ✓ Delete OK")


def test_prediction_cache():
    """Test prediction store → cache hit → invalidation on data change."""
    print("4. Testing prediction cache...")
    from services import persistence_service as ps

    symbol = "TEST"
    trade_date = "2026-07-13"
    model_name = "kronos_mini"
    data_hash_v1 = ps.compute_data_hash({"close": [100, 101, 102]})
    data_hash_v2 = ps.compute_data_hash({"close": [100, 101, 103]})  # changed

    result = {
        "decision": "SELL",
        "confidence": 0.85,
        "up_probability": 0.15,
        "details": {"sample_count": 20},
    }

    # Store
    ps.store_prediction(symbol, trade_date, model_name, data_hash_v1, result)
    print("   ✓ Prediction stored")

    # Cache hit (same hash)
    cached = ps.get_cached_prediction(symbol, trade_date, model_name, data_hash_v1)
    assert cached is not None
    assert cached["decision"] == "SELL"
    print("   ✓ Cache HIT (same data hash)")

    # Cache miss (different hash = data changed)
    cached2 = ps.get_cached_prediction(symbol, trade_date, model_name, data_hash_v2)
    assert cached2 is None
    print("   ✓ Cache MISS (data hash changed → stale)")


def test_report_cache():
    """Test report store (S3 + catalog) → cache hit → invalidation."""
    print("5. Testing report cache (Postgres + S3)...")
    from services import persistence_service as ps
    from services import storage_service

    storage_service.ensure_bucket()

    symbol = "TEST"
    trade_date = "2026-07-13"
    report_type = "ai_report"
    data_hash = ps.compute_data_hash({"articles": ["headline1", "headline2"]})
    content = "# AI Report for TEST\n\nBullish outlook based on earnings beat."

    # Store
    key = ps.store_report(symbol, trade_date, report_type, data_hash, content)
    assert key.startswith("reports/")
    print(f"   ✓ Report stored at {key}")

    # Cache hit
    cached = ps.get_cached_report(symbol, trade_date, report_type, data_hash)
    assert cached == content
    print("   ✓ Cache HIT (report content matches)")

    # Cache miss (different hash)
    bad_hash = ps.compute_data_hash({"articles": ["different"]})
    cached2 = ps.get_cached_report(symbol, trade_date, report_type, bad_hash)
    assert cached2 is None
    print("   ✓ Cache MISS (data hash changed)")


def test_recommendation_cache():
    """Test recommendation store → cache hit."""
    print("6. Testing recommendation cache...")
    from services import persistence_service as ps

    trade_date = "2026-07-13"
    symbols = ["EIX", "EPAM"]
    data_hash = ps.compute_data_hash({"combined": "hash_of_all_inputs"})
    result = {
        "overall": {"summary": "Sell EIX, hold EPAM"},
        "by_symbol": {
            "EIX": {"action": "SELL", "conviction": "HIGH"},
            "EPAM": {"action": "HOLD", "conviction": "MEDIUM"},
        },
    }

    ps.store_recommendation(
        trade_date, symbols, data_hash, result,
        model_used="gpt-5.6-luna", provider_used="openai",
    )
    print("   ✓ Recommendation stored")

    cached = ps.get_cached_recommendation(trade_date, data_hash)
    assert cached is not None
    assert cached["by_symbol"]["EIX"]["action"] == "SELL"
    print("   ✓ Cache HIT")


def test_data_snapshot_invalidation():
    """Test that data snapshots detect changes."""
    print("7. Testing data snapshot invalidation...")
    from services import persistence_service as ps

    symbol = "TEST"
    trade_date = "2026-07-13"
    hash_v1 = ps.compute_data_hash({"prices": [100, 101]})
    hash_v2 = ps.compute_data_hash({"prices": [100, 102]})

    changed = ps.upsert_data_snapshot(symbol, trade_date, "stock_prices", hash_v1)
    assert changed is True
    print("   ✓ First insert → changed=True")

    changed = ps.upsert_data_snapshot(symbol, trade_date, "stock_prices", hash_v1)
    assert changed is False
    print("   ✓ Same hash → changed=False (no recomputation needed)")

    changed = ps.upsert_data_snapshot(symbol, trade_date, "stock_prices", hash_v2)
    assert changed is True
    print("   ✓ New hash → changed=True (data updated, recompute)")


def main():
    print("=" * 60)
    print("Persistence Layer — End-to-End Test")
    print("=" * 60)

    tests = [
        test_db_connection,
        test_alembic_migrations,
        test_s3_storage,
        test_prediction_cache,
        test_report_cache,
        test_recommendation_cache,
        test_data_snapshot_invalidation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
