"""The Run dialog and the scheduled run are ONE pipeline.

2026-09-01 audit: the interactive path in app.py had re-implemented the
model, persist and synthesis stages and drifted in eight places (no
pipeline-epoch/news-status stamps, NULL synthesis confidence, a different
recommendation cache key, the asked-for model recorded instead of the one
that answered, ...). app.py now calls services.analysis_runner for all of
them; these tests pin the shared contracts so a second copy cannot grow back
unnoticed.
"""

import ast
import pathlib

import pytest
from unittest.mock import patch

import pandas as pd

from services import analysis_runner as ar

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _app_source() -> str:
    return (ROOT / "app.py").read_text()


class TestOneImplementation:
    def test_app_delegates_the_three_stages_to_the_runner(self):
        src = _app_source()
        for name in ("run_predictions", "persist_predictions as _persist",
                     "run_recommendations", "report_cache_key as _report_key",
                     "load_market_data"):
            assert f"from services.analysis_runner import" in src and name in src, name

    def test_app_has_no_private_copies_of_runner_logic(self):
        """Names that only existed in the interactive re-implementation."""
        src = _app_source()
        for relic in ("_conv2conf", "_art_dict", "predict_symbol_no_store(",
                      '"schema": "v5-flowq"', "_volatile = ("):
            assert relic not in src, relic

    def test_recommendation_key_treats_model_and_basis_as_inputs(self):
        assert "recs_model" not in ar.RECOMMENDATION_KEY_VOLATILE
        assert "recs_request" not in ar.RECOMMENDATION_KEY_VOLATILE
        assert {"generated_at", "from_cache", "run_seq", "scope"} <= set(
            ar.RECOMMENDATION_KEY_VOLATILE)

    def test_report_cache_key_covers_every_input_both_paths_set(self):
        key = ar.report_cache_key(
            {"TYL": []}, ["TYL"], "2026-08-31", "gpt-5.6-luna", 30, True,
            evidence=["quality"], max_articles=100, overnight=False,
            include_research=True, recs_mode="signals")
        assert key["lookback"] == 30 and key["max_articles"] == 100
        assert key["research"] is True and key["recs"] == "signals"
        assert key["evidence"] == ["quality"]
        overnight = ar.report_cache_key(
            {"TYL": []}, ["TYL"], "2026-08-31", "m", 1, True, overnight=True,
            max_articles=0)
        assert overnight["lookback"] == "overnight"

    def test_report_cache_key_refuses_a_missing_cap(self):
        """No default: the cap is the frontend's, or the key cannot be built."""
        from services.news_window import RunParameterMissing
        with pytest.raises(RunParameterMissing):
            ar.report_cache_key({"TYL": []}, ["TYL"], "2026-08-31", "m", 1, True)


class TestReuseGuardKnowsTheWindow:
    """Same article count must not serve a different window or depth."""

    def _stored(self, **details):
        base = {"previous_close": 10.0,
                "details": {"pipeline_epoch": ar.PIPELINE_EPOCH, "model": "m",
                            "evidence": ["options", "quality"],
                            "news_count": 6, "news_status": "ok", **details}}
        return {"trading_agents": dict(base), "ensemble": dict(base),
                "kronos_mini": dict(base)}

    def _df(self):
        return pd.DataFrame({"Close": [10.0]}, index=pd.to_datetime(["2026-08-31"]))

    def _reuse(self, stored, **kw):
        class _Cache:
            def get_predictions_for_today(self, *a, **k):
                return stored
        with patch.object(ar, "get_cache", create=True):
            with patch("services.cache_service.get_cache", return_value=_Cache()):
                return ar._reusable_predictions(
                    "TYL", pd.Timestamp("2026-08-31").date(), {"kronos_mini"},
                    self._df(), research_model="m", news_count=6, **kw)

    def test_same_window_and_depth_is_reused(self):
        stored = self._stored(news_window_days=30, include_thesis=True)
        assert self._reuse(stored, news_lookback_days=30, include_thesis=True)

    def test_different_window_reruns(self):
        stored = self._stored(news_window_days=30, include_thesis=True)
        assert self._reuse(stored, news_lookback_days=7, include_thesis=True) is None

    def test_different_depth_reruns(self):
        stored = self._stored(news_window_days=30, include_thesis=True)
        assert self._reuse(stored, news_lookback_days=30, include_thesis=False) is None

    def test_rows_from_before_the_stamps_rerun_once(self):
        stored = self._stored()  # no news_window_days / include_thesis
        assert self._reuse(stored, news_lookback_days=30, include_thesis=True) is None


class TestSchedulerCommand:
    """A scheduled job is a saved Run dialog: every setting reaches the CLI."""

    def test_every_dialog_setting_reaches_the_cli(self):
        from services import scheduler_service as ss
        params = {**ss.default_run_params(), "lookback": 30, "max_articles": 0,
                  "models": ["kronos_mini"], "run_ensemble": False,
                  "depth": "standard", "recs": "off", "evidence": [],
                  "tools": []}
        cmd = ss._build_command({"kind": "analysis", "symbols_csv": "TYL,BAC",
                                 "params": params}, None)
        flag = lambda f: cmd[cmd.index(f) + 1]  # noqa: E731
        assert flag("--max-articles") == "0" and flag("--lookback") == "30"
        assert flag("--news-filter") == "lookback"
        assert flag("--models") == "kronos_mini" and "--no-ensemble" in cmd
        assert flag("--depth") == "standard" and flag("--recs") == "off"
        assert flag("--evidence") == "none" and "--tools" not in cmd
        assert flag("--ensemble-json").startswith("{")

    def test_overnight_window_maps_to_the_news_filter(self):
        from services import scheduler_service as ss
        params = {**ss.default_run_params(), "lookback": "overnight"}
        cmd = ss._build_command({"kind": "analysis", "symbols_csv": "TYL",
                                 "params": params}, None)
        assert cmd[cmd.index("--news-filter") + 1] == "overnight"
        assert cmd[cmd.index("--lookback") + 1] == "1"

    def test_pre_modal_job_still_maps_web_research(self):
        from services import scheduler_service as ss
        cmd = ss._build_command({"kind": "analysis", "symbols_csv": "TYL",
                                 "params": {"lookback": 7, "max_articles": 500,
                                            "web_research": 1}}, None)
        assert cmd[cmd.index("--tools") + 1] == "web_research"

    def test_job_without_window_or_cap_is_refused(self):
        from services import scheduler_service as ss
        with pytest.raises(ValueError, match="missing"):
            ss._build_command({"kind": "analysis", "symbols_csv": "TYL",
                               "params": {"lookback": 7}}, None)

    def test_defaults_mirror_the_run_dialog(self):
        from config import MODEL
        from layouts.modals import RUN_MODELS
        from services import scheduler_service as ss
        d = ss.default_run_params()
        assert d["lookback"] == MODEL.NEWS_LOOKBACK_DAYS
        assert d["max_articles"] == MODEL.NEWS_MAX_ARTICLES
        assert d["models"] == [m for m, _, _ in RUN_MODELS]
        assert set(d["evidence"]) == set(MODEL.DEFAULT_EVIDENCE)
        assert d["recs"] == "auto" and d["depth"] == "thesis"


class TestNoNewestFirstSlices:
    """No prompt builder may take ``articles[:n]`` off a newest-first list."""

    def test_prompt_builders_use_select_spread(self):
        for rel in ("models/single_agent.py", "services/llm_service.py"):
            src = (ROOT / rel).read_text()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in ("articles", "news_articles", "shown")):
                    raise AssertionError(
                        f"{rel}:{node.lineno} slices {node.value.id}: use select_spread")
            assert "select_spread" in src, rel
