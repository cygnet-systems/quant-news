"""A scheduled run owns an analysis_runs row from before its first stage to
after its last, and closes it from inside the runner.

The row is what the pill and the Home tabs use to tell a scheduled run
from a manual one, so it must exist before start_run opens the feed, link
to the JobRun the scheduler passed by env, and reach a terminal status on
every exit: a clean run, a degraded one, an early abort, and a stage that
raises. The stages themselves are stubbed; nothing here fetches or models.
"""

import diskcache
import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import AnalysisRun, Base
from services import analysis_runner as ar
from services import progress_service as prog
from services import run_service as rs
from services.news_window import fetch_run_news


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    # Test-only: lets the Postgres models build on in-memory SQLite.
    return "JSON"


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


@pytest.fixture
def feed(monkeypatch, tmp_path):
    monkeypatch.setenv(prog._ENV_FLAG, "1")
    monkeypatch.setattr(prog, "_cache", diskcache.Cache(str(tmp_path)))
    monkeypatch.setattr(prog, "_local_run", {"id": None, "title": None})
    monkeypatch.setattr(prog, "_write_audit", lambda *a, **kw: None)
    token = prog._run_ctx.set(None)
    yield prog
    prog._run_ctx.reset(token)


@pytest.fixture
def stages(monkeypatch):
    """Every stage of _run_stages replaced by a stub that reports as the
    real one would; the dict records what ran."""
    seen = {}

    def _market(symbols, period="2y", force_refresh=False):
        seen["market"] = list(symbols)
        return {s: {"prices": "{}"} for s in symbols if s != "NOPX"}

    def _news(symbols, as_of, target, *, overnight, lookback_days,
              max_articles, on_symbol=None, **kw):
        by = {s: [{"title": f"{s} story"}] for s in symbols}
        stats = {s: {"kept": 1, "status": "ok"} for s in symbols}
        for s in symbols:
            on_symbol(s, by[s], stats[s])
        return by, stats

    def _predict(symbols, *a, **kw):
        prog.emit_progress("models", done=len(symbols), total=len(symbols),
                           state="done")
        return {s: {"ensemble": {"decision": "BUY"}} for s in symbols} | {
            "_meta": {"is_backtest": False, "skipped_symbols": {},
                      "training_news_failures": []}}

    def _persist(signals, run_id=None):
        n = sum(1 for k in signals if k != "_meta")
        prog.emit_progress("models", predictions=n)
        return n, 0

    def _report(symbols, *a, **kw):
        prog.emit_progress("report", state="done", done=len(symbols),
                           total=len(symbols))
        return {"overall": {"recommendation": "HOLD"}, "by_symbol": {}}

    def _recs(ai, signals, symbols, trade_date, run_id=None):
        seen["recs"] = list(symbols)
        prog.emit_progress("synthesis", state="done", done=len(symbols),
                           total=len(symbols))
        return {"by_symbol": {s: {"action": "BUY"} for s in symbols}}

    class _PS:
        @staticmethod
        def store_report(**kw):
            seen.setdefault("archived", []).append(kw["report_type"])

        @staticmethod
        def compute_data_hash(obj):
            return "h"

    class _Cache:
        @staticmethod
        def evaluate_predictions():
            return 0

    monkeypatch.setattr(ar, "load_market_data", _market)
    monkeypatch.setattr("services.news_window.fetch_run_news", _news)
    monkeypatch.setattr(ar, "run_predictions", _predict)
    monkeypatch.setattr(ar, "persist_predictions", _persist)
    monkeypatch.setattr(ar, "build_ai_report", _report)
    monkeypatch.setattr(ar, "attach_positioning_quality",
                        lambda ai, *a, **kw: ai)
    monkeypatch.setattr(ar, "run_recommendations", _recs)
    monkeypatch.setattr(ar, "_assess_completeness",
                        lambda *a, **kw: ({}, []))
    monkeypatch.setattr("services.persistence_service.store_report",
                        _PS.store_report)
    monkeypatch.setattr("services.persistence_service.compute_data_hash",
                        _PS.compute_data_hash)
    monkeypatch.setattr("services.cache_service.get_cache", lambda: _Cache())
    monkeypatch.setattr(prog, "emit_memory", lambda *a, **kw: None)
    return seen


def _only_run():
    (run,) = rs.list_runs()
    return run


def test_clean_run_creates_links_and_closes_the_row(db, feed, stages, monkeypatch):
    monkeypatch.setenv("QUANTNEWS_JOB_RUN_ID", "41")
    monkeypatch.setenv("QUANTNEWS_RUN_OWNER", "owner-1")

    summary = ar.run_full_analysis(["NVDA", "AMD"], lookback_days=7,
                                   max_articles=50, recs_mode="auto")

    run = _only_run()
    assert run["kind"] == "scheduled"
    assert run["job_run_id"] == 41
    assert run["owner_uid"] == "owner-1"
    assert run["status"] == "done" and run["error"] is None
    assert run["symbols"] == ["NVDA", "AMD"]
    assert run["prediction_date"] == summary["as_of"]
    assert run["target_date"] == summary["target_date"]
    assert run["config"]["lookback_days"] == 7
    assert run["stages"]["news"] == {"state": "done", "done": 2, "total": 2}
    assert run["stages"]["models"]["state"] == "done"
    assert run["stages"]["report"]["state"] == "done"
    assert run["stages"]["synthesis"]["state"] == "done"
    assert run["counters"] == {"articles": 2, "predictions": 2, "archived": 1}
    assert stages["archived"] == ["news_snapshot", "ai_report"]
    # The feed opened under the row's id and closed with it.
    assert feed._active_runs() == []
    assert feed.get_feed(run["run_id"])["events"][-1]["stage"] == "done"
    assert summary["predictions_stored"] == 2


def test_cli_run_without_a_job_has_no_link(db, feed, stages, monkeypatch):
    monkeypatch.delenv("QUANTNEWS_JOB_RUN_ID", raising=False)
    monkeypatch.delenv("QUANTNEWS_RUN_OWNER", raising=False)
    ar.run_full_analysis(["AAPL"], lookback_days=3, max_articles=0,
                         recs_mode="off")
    run = _only_run()
    assert run["job_run_id"] is None and run["owner_uid"] is None
    assert run["status"] == "done"
    assert run["stages"]["synthesis"] == {"state": "skipped"}


def test_early_abort_fails_the_row(db, feed, stages):
    summary = ar.run_full_analysis(["NOPX"], lookback_days=3, max_articles=0)
    assert summary["error"] == "no price data"
    run = _only_run()
    assert run["status"] == "failed"
    assert run["error"] == "no price data"
    assert feed._active_runs() == []


def test_raising_stage_fails_the_row_and_reraises(db, feed, stages, monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("model weights missing")

    monkeypatch.setattr(ar, "run_predictions", _boom)
    with pytest.raises(RuntimeError):
        ar.run_full_analysis(["NVDA"], lookback_days=3, max_articles=0)
    run = _only_run()
    assert run["status"] == "failed"
    assert "model weights missing" in run["error"]
    assert feed._active_runs() == []
    assert feed.get_feed(run["run_id"])["events"][-1]["stage"] == "done"


def test_row_failure_does_not_stop_the_run(db, feed, stages, monkeypatch):
    monkeypatch.setattr(rs, "create_run",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("db down")))
    summary = ar.run_full_analysis(["NVDA"], lookback_days=3, max_articles=0)
    assert summary["predictions_stored"] == 1
    assert rs.list_runs() == []
    assert feed._active_runs() == []


def test_news_hook_reports_each_symbol(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "services.news_window.fetch_point_in_time_news_with_stats",
        lambda sym, as_of, lookback_days, max_articles: (
            [] if sym == "QUIET" else [object()],
            {"kept": 0 if sym == "QUIET" else 1}))
    monkeypatch.setattr("services.news_window.article_to_dict",
                        lambda a: {"title": "x"})
    by, stats = fetch_run_news(
        ["NVDA", "QUIET"], "2026-09-01", "2026-09-02", overnight=False,
        lookback_days=7, max_articles=50,
        on_symbol=lambda s, arts, st: calls.append((s, len(arts), st["status"])))
    assert calls == [("NVDA", 1, "ok"), ("QUIET", 0, "empty")]
