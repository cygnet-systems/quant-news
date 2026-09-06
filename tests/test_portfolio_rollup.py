"""The portfolio view is arithmetic over the research epilogues, and the
per-symbol action is the research verdict with one HOLD rule. No model
decides direction here any more (prod: the synthesis re-vote agreed with the
report 68% of the time and hit 49.8% vs the report's 46.1% on the same
days, both coin-flip)."""
from unittest.mock import patch

from services import portfolio_rollup as pr


def _entry(decision="SELL", conviction=0.56, gaps=(), situation=None,
           watch=("close above $159.70",)):
    return {
        "recommendation": pr.STANCE_OF[decision],
        "stance_source": "research_verdict",
        "research": {
            "decision": decision, "stated_conviction": conviction,
            "investigation": {"situation": situation} if situation else {},
            "evidence": {"gaps": [{"severity": "expected", "label": g}
                                  for g in gaps]},
            "structured": {"watch_items": list(watch)},
        },
    }


def _signals(*decisions):
    sig = {f"m{i}": {"decision": d} for i, d in enumerate(decisions)}
    sig["trading_agents"] = {"decision": "SELL"}
    sig["ensemble"] = {"decision": "BUY"}
    return sig


class TestDecideAction:
    def test_the_verdict_stands_when_evidence_was_complete(self):
        d = pr.decide_action("ORCL", _entry(), _signals("BUY", "BUY"))
        assert d["action"] == "SELL" and d["rule"] == pr.RULE_VERDICT
        assert d["models_opposed"] == 2 and d["models_agreeing"] == 0
        assert d["p_correct"] == 0.56

    def test_the_verdict_stands_when_models_agree_despite_gaps(self):
        d = pr.decide_action("ORCL", _entry(gaps=("options positioning",)),
                             _signals("SELL", "SELL", "BUY"))
        assert d["action"] == "SELL" and d["rule"] == pr.RULE_VERDICT

    def test_hold_when_models_mostly_disagree_and_evidence_was_missing(self):
        d = pr.decide_action("ORCL", _entry(gaps=("options positioning",)),
                             _signals("BUY", "BUY"))
        assert d["action"] == "HOLD" and d["rule"] == pr.RULE_HOLD
        assert d["p_correct"] == 0.5
        assert d["missing_evidence"] == ["options positioning"]

    def test_trading_agents_and_ensemble_are_not_second_opinions(self):
        # Only the ensemble and the report itself: nothing independent
        # opposes, so the verdict stands even with a gap.
        d = pr.decide_action("ORCL", _entry(gaps=("quality screen",)),
                             {"trading_agents": {"decision": "SELL"},
                              "ensemble": {"decision": "BUY"}})
        assert d["action"] == "SELL" and d["models_opposed"] == 0

    def test_no_research_means_hold(self):
        d = pr.decide_action("ORCL", {}, _signals("BUY"))
        assert d["action"] == "HOLD" and d["rule"] == pr.RULE_NO_RESEARCH


class TestRollup:
    def test_counts_stance_and_held_symbols(self):
        by = {"ORCL": _entry("SELL", 0.56, gaps=("situation & investigation",),
                             situation="EARNINGS_EVENT"),
              "BAC": _entry("SELL", 0.6),
              "MCD": _entry("BUY", 0.52)}
        sig = {"ORCL": _signals("BUY", "BUY"), "BAC": _signals("SELL"),
               "MCD": _signals("BUY")}
        o = pr.build_overall_rollup(by, sig)
        assert o["recommendation"] == "BEARISH" and o["market_sentiment"] == "BEARISH"
        assert o["confidence"] == 0.56
        assert "3 symbols researched: 1 bullish, 2 bearish, 0 neutral" in o["key_developments"]
        assert "ORCL SELL (0.56), earnings event, held back" in o["key_developments"]
        assert "1 held back to HOLD" in o["developments_read"]
        assert "MCD" in o["risk_factors"]  # conviction below 0.55
        assert o["watch_items"][0].startswith("BAC: ")
        assert o["rollup"] is True and o["source"] == "research_epilogues"

    def test_two_thirds_is_the_line_between_cautious_and_full(self):
        by = {s: _entry("BUY", 0.6) for s in ("A", "B")}
        by["C"] = _entry("SELL", 0.6)
        assert pr.build_overall_rollup(by)["recommendation"] == "BULLISH"
        by["D"] = _entry("HOLD", 0.5)
        assert pr.build_overall_rollup(by)["recommendation"] == "CAUTIOUS_BULLISH"

    def test_nothing_researched_is_none(self):
        assert pr.build_overall_rollup({}) is None
        assert pr.build_overall_rollup({"X": {"recommendation": "NEUTRAL"}}) is None


class TestSynthesisTakesTheFixedAction:
    def test_the_model_cannot_change_the_action(self):
        from services.llm_service import LLMService
        svc = LLMService.__new__(LLMService)
        raw = ('{"overall": {"summary": "s", "portfolio_action": "a", '
               '"key_conflicts": [], "risk_assessment": "r", "watch_items": []}, '
               '"by_symbol": {"ORCL": {"action": "BUY", "p_correct": 0.7, '
               '"reasoning": "the model wanted to buy"}}}')
        seen = {}

        def fake_generate(prompt, system_prompt, **kw):
            seen["prompt"] = prompt
            seen["system"] = system_prompt
            return raw

        fixed = {"ORCL": {"action": "SELL", "p_correct": 0.56,
                          "rule": pr.RULE_VERDICT, "verdict": "SELL",
                          "models_agreeing": 0, "models_opposed": 2}}
        with patch.object(LLMService, "generate", side_effect=fake_generate), \
                patch("services.llm_service.MODEL") as m:
            m.RECOMMENDATIONS_MODEL = "gpt-5.6-luna"
            m.RECOMMENDATIONS_PROVIDER = "openai"
            m.RECOMMENDATIONS_FALLBACK_MODEL = ""
            m.RECOMMENDATIONS_MAX_TOKENS = 100
            m.RECOMMENDATIONS_TEMPERATURE = 0.2
            m.RECOMMENDATIONS_REASONING_EFFORT = "low"
            out = svc.generate_recommendations(
                {"by_symbol": {"ORCL": _entry()}}, {"ORCL": _signals("BUY", "BUY")},
                ["ORCL"], basis="research+signals", fixed_actions=fixed)
        rec = out["by_symbol"]["ORCL"]
        assert rec["action"] == "SELL" and rec["p_correct"] == 0.56
        assert rec["action_source"] == pr.RULE_VERDICT
        assert rec["model_would_have"] == "BUY"
        assert "ACTION (set by the platform, the research verdict): SELL" in seen["prompt"]
        assert "THE ACTION IS NOT YOURS TO CHOOSE" in seen["system"]
        assert "OVERALL AI REPORT" not in seen["prompt"]


class TestRunRecommendationsFixesActionsAfterTheMerge:
    def test_the_verdict_reaches_the_memo_model(self, monkeypatch):
        """The research verdict lives on the trading_agents signal until
        merge_research_into_analysis copies it into the analysis. Deciding
        the actions before that merge saw no verdict and sent every symbol
        to the memo model as HOLD (the first live run did exactly that)."""
        from services import analysis_runner as ar
        from services import persistence_service as ps
        seen = {}

        class Stub:
            def generate_recommendations(self, ai_analysis, signals, symbols,
                                         basis=None, model_override=None,
                                         fixed_actions=None):
                seen["fixed"] = fixed_actions
                seen["overall"] = ai_analysis.get("overall")
                return {"overall": {}, "by_symbol": {
                    "ORCL": {"action": fixed_actions["ORCL"]["action"],
                             "p_correct": fixed_actions["ORCL"]["p_correct"],
                             "action_source": fixed_actions["ORCL"]["rule"]}},
                    "model_used": "stub", "basis": basis}

        class Cache:
            def store_prediction(self, *a, **k):
                seen.setdefault("stored", []).append(a[:2])

        monkeypatch.setattr("services.llm_service.get_llm", lambda: Stub())
        monkeypatch.setattr("services.cache_service.get_cache", lambda: Cache())
        monkeypatch.setattr(ps, "get_cached_recommendation", lambda *a, **k: None)
        monkeypatch.setattr(ps, "store_recommendation", lambda **k: None)
        monkeypatch.setattr("services.progress_service.emit", lambda *a, **k: None)
        monkeypatch.setattr("services.progress_service.emit_progress",
                            lambda *a, **k: None)
        signals = {"ORCL": {
            "trading_agents": {"decision": "SELL", "confidence": 0.5,
                               "details": {"raw_response": "## Verdict\nSELL",
                                           "stated_conviction": 0.56,
                                           "structured": {"stance": "BEARISH"},
                                           "evidence": {"gaps": []}}},
            "xgboost_shap": {"decision": "BUY", "confidence": 0.56},
            "lightgbm": {"decision": "BUY", "confidence": 0.82},
        }}
        ai = {"by_symbol": {}, "recs_request": "news+signals",
              "recs_model": "stub", "as_of": "2026-09-04"}
        out = ar.run_recommendations(ai, signals, ["ORCL"], "2026-09-04")
        assert seen["fixed"]["ORCL"]["action"] == "SELL"
        assert seen["fixed"]["ORCL"]["rule"] == pr.RULE_VERDICT
        assert seen["overall"]["recommendation"] == "BEARISH"
        assert out["by_symbol"]["ORCL"]["action"] == "SELL"
        assert ("ORCL", "recommendation_synthesis") in seen["stored"]
