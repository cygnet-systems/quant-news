"""The three memory levers from the 2026-09-02 container deaths.

Three silent reboots mid daily_analysis, no traceback, each catch-up
re-billing the run. The audit found no leak: the first symbol of the model
loop loads torch, transformers and the DeBERTa weights while XGBoost and
LightGBM train beside it, and the investigation prefetch pool had started
N workers (yfinance histories, statements, an uncapped news window each)
on top of that same spike. These tests pin the three changes:

  - the red-flag scan reads the run's own capped news window instead of
    fetching a second, uncapped one per symbol;
  - the investigation prefetch starts only after the first symbol has run
    (weights loaded), and covers the symbols still to come;
  - stage boundaries emit an RSS figure so the next incident has numbers.
"""

from contextlib import nullcontext
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from config import MODEL
from services import bad_apples_service as ba

AS_OF = "2026-09-01"
HEADLINE = "CFO resigns as SEC subpoena lands; auditor raises going concern doubt"


def _article_dict(title=HEADLINE):
    return {"title": title, "summary": "", "url": "https://x/1",
            "published_at": "2026-08-30T14:00:00+00:00"}


class _ArticleObj:
    def __init__(self, title=HEADLINE):
        self.title, self.summary, self.url = title, "", "https://x/1"
        self.published_at = "2026-08-30T14:00:00+00:00"


class TestRedFlagScanReadsTheRunWindow:
    def test_supplied_articles_are_scanned_without_a_fetch(self):
        with patch("services.news_window.fetch_point_in_time_news",
                   side_effect=AssertionError("must not fetch")):
            hits = ba.scan_news_red_flags("X", AS_OF, articles=[_article_dict()])
        assert isinstance(hits, list) and hits, "keyword scan found nothing"
        assert hits[0]["headline"].startswith("CFO resigns")
        assert hits[0]["date"] == "2026-08-30"

    def test_dict_and_object_articles_scan_the_same(self):
        as_dicts = ba.scan_news_red_flags("X", AS_OF, articles=[_article_dict()])
        as_objs = ba.scan_news_red_flags("X", AS_OF, articles=[_ArticleObj()])
        assert as_dicts == as_objs

    def test_no_articles_still_fetches_so_outage_stays_visible(self):
        with patch("services.news_window.fetch_point_in_time_news",
                   side_effect=RuntimeError("source down")):
            assert ba.scan_news_red_flags("X", AS_OF) is None

    def test_analyze_symbol_records_the_scope_and_skips_the_fetch(self):
        ba._CACHE.pop(("ZZZ", AS_OF), None)
        with patch("services.stock_data.get_ticker",
                   side_effect=RuntimeError("no yfinance in tests")), \
             patch("services.terminal_cache.get_info", return_value={}), \
             patch("services.news_window.fetch_point_in_time_news",
                   side_effect=AssertionError("must not fetch")):
            out = ba.analyze_symbol("ZZZ", AS_OF, articles=[_article_dict()])
        assert out["red_flag_scope"] == "run news window (1 articles)"
        assert out["red_flags"]
        assert ba.summarize(out)["red_flag_scope"] == out["red_flag_scope"]
        assert "run news window (1 articles)" in ba.format_bad_apples_block("ZZZ", out)
        ba._CACHE.pop(("ZZZ", AS_OF), None)


class _Pool:
    def shutdown(self, wait=False):
        pass


def _stock_data(symbols):
    idx = pd.date_range("2026-06-01", periods=60, freq="B")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                       "Volume": 1}, index=idx)
    return {s: {"prices": df.to_json(date_format="iso")} for s in symbols}


def _run(symbols, events, reused=None):
    """run_predictions with every model stubbed; ``events`` records the
    order of model runs and prefetch starts."""
    from services import analysis_runner

    class _Service:
        def predict_symbol_no_store(self, symbol, df, **kw):
            events.append(f"predict:{symbol}")
            return {}

    def _prefetch(remaining, as_of, **kw):
        events.append(f"prefetch:{','.join(remaining)}")
        return _Pool()

    spy = pd.DataFrame({"Close": [1.0] * 10},
                       index=pd.date_range("2026-08-01", periods=10, freq="B"))
    with patch("services.prediction_service.get_prediction_service",
               return_value=_Service()), \
         patch("services.stock_data.fetch_stock_data", return_value=spy), \
         patch("services.investigation_service.prefetch_many", side_effect=_prefetch), \
         patch("services.analysis_runner._reusable_predictions",
               side_effect=lambda symbol, *a, **k: (reused or {}).get(symbol)), \
         patch("services.usage_service.track", return_value=nullcontext()):
        return analysis_runner.run_predictions(
            symbols, _stock_data(symbols), {s: [] for s in symbols},
            target_date=date(2026, 9, 2), cutoff_date=date(2026, 9, 1),
            models={"trading_agents"}, evidence=["investigation"],
            news_lookback_days=14, tools=["web_research"])


class TestPrefetchStartsAfterTheLoadSpike:
    """The pool only exists when investigation is NOT gated on anomalies.

    With the gate on (the default since 2026-09-05) there is no pool at all:
    the gate cannot know which symbols are quiet until each one's evidence
    blocks are built inside the loop, so investigation moved in-loop and only
    flagged names pay for it. These tests pin the pool's ordering for the
    ungated configuration, which is still what runs when a deployment turns
    the gate off.
    """

    @pytest.fixture(autouse=True)
    def _ungated(self, monkeypatch):
        # MODEL is a frozen dataclass; swap the module's reference for a copy
        # with the gate off, the same way test_news_availability does.
        import dataclasses

        from services import analysis_runner as ar

        monkeypatch.setattr(ar, "MODEL", dataclasses.replace(
            MODEL, INVESTIGATE_ONLY_ANOMALIES=False))

    def test_pool_starts_after_the_first_symbol_and_covers_the_rest(self):
        events = []
        _run(["A", "B", "C"], events)
        assert events == ["predict:A", "prefetch:B,C", "predict:B", "predict:C"]

    def test_a_reused_first_symbol_does_not_count_as_the_load(self):
        events = []
        _run(["A", "B", "C"], events, reused={"A": {"trading_agents": {}}})
        assert events == ["predict:B", "prefetch:C", "predict:C"]

    def test_single_symbol_never_starts_a_pool(self):
        events = []
        _run(["A"], events)
        assert events == ["predict:A"]

    def test_no_pool_without_the_investigation_block(self):
        from services import analysis_runner
        events = []

        class _Service:
            def predict_symbol_no_store(self, symbol, df, **kw):
                events.append(f"predict:{symbol}")
                return {}

        with patch("services.prediction_service.get_prediction_service",
                   return_value=_Service()), \
             patch("services.stock_data.fetch_stock_data",
                   side_effect=RuntimeError("no spy")), \
             patch("services.investigation_service.prefetch_many",
                   side_effect=AssertionError("must not start")), \
             patch("services.analysis_runner._reusable_predictions", return_value=None), \
             patch("services.usage_service.track", return_value=nullcontext()):
            analysis_runner.run_predictions(
                ["A", "B"], _stock_data(["A", "B"]), {"A": [], "B": []},
                target_date=date(2026, 9, 2), cutoff_date=date(2026, 9, 1),
                models={"trading_agents"}, evidence=[], news_lookback_days=14)
        assert events == ["predict:A", "predict:B"]


class TestMemoryTelemetry:
    def test_stage_boundary_emits_rss(self):
        from services import progress_service as prog
        seen = []
        with patch("services.progress_service.emit",
                   side_effect=lambda stage, msg, payload=None, **k: seen.append((stage, msg, payload))):
            prog.emit_memory("news fetch", articles=12)
        assert len(seen) == 1
        stage, msg, payload = seen[0]
        assert payload["event"] == "memory" and payload["stage"] == "news fetch"
        assert payload["rss_mb"] > 0 and payload["articles"] == 12
        assert "MB RSS after news fetch" in msg

    def test_rss_is_a_positive_number_here(self):
        from services.progress_service import rss_mb
        assert rss_mb() > 0


class TestTorchThreadCap:
    def test_env_pins_torch_threads(self, monkeypatch):
        torch = pytest.importorskip("torch")
        from models.base import apply_torch_thread_cap
        before = torch.get_num_threads()
        try:
            monkeypatch.setenv("TORCH_NUM_THREADS", "1")
            apply_torch_thread_cap()
            assert torch.get_num_threads() == 1
        finally:
            torch.set_num_threads(before)

    def test_unset_env_leaves_torch_alone(self, monkeypatch):
        torch = pytest.importorskip("torch")
        from models.base import apply_torch_thread_cap
        monkeypatch.delenv("TORCH_NUM_THREADS", raising=False)
        before = torch.get_num_threads()
        apply_torch_thread_cap()
        assert torch.get_num_threads() == before
