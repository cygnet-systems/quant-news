"""The paths that carry the investigation and the evidence gaps downstream:
the run completeness check and the synthesis evidence lines."""

from services.analysis_runner import _assess_completeness, merge_research_into_analysis
from services.llm_service import LLMService


def _signals(gaps=None, investigation=None):
    details = {
        "raw_response": "## Verdict\nFINAL TRANSACTION PROPOSAL: **SELL**\n\n"
                        "### 1. Situation & Key Figures, pending deal\ntext\n"
                        "### 8. Bull vs Bear, x\ntext\n",
        "evidence": {"present": ["metrics"], "gaps": gaps or [],
                     "degraded": bool(gaps)},
        "investigation": investigation or {},
        "stated_conviction": 0.6,
        "structured": {"stance": "BEARISH"},
    }
    return {"BHF": {
        "trading_agents": {"decision": "SELL", "confidence": 0.5, "details": details},
        "kronos_mini": {"decision": "BUY", "confidence": 0.7},
        "ensemble": {"decision": "HOLD", "confidence": 0.1},
    }}


def test_completeness_marks_expected_gaps_as_degraded():
    gaps = [{"block": "options", "severity": "expected", "reason": "throttled",
             "label": "options positioning"},
            {"block": "political", "severity": "optional", "reason": "n/a",
             "label": "political & institutional flows"}]
    coverage, degraded = _assess_completeness(
        _signals(gaps), ["BHF"], {"kronos_mini", "trading_agents"},
        {"by_symbol": {"BHF": {"action": "SELL"}}})
    assert coverage["trading_agents"] == 1
    assert degraded == ["research written without expected evidence: "
                        "BHF (options positioning)"]
    _, clean = _assess_completeness(
        _signals(), ["BHF"], {"kronos_mini", "trading_agents"},
        {"by_symbol": {"BHF": {"action": "SELL"}}})
    assert clean == []


def test_synthesis_lines_carry_situation_spread_and_gaps():
    inv = {"situation": "PENDING_ACQUISITION", "situation_confidence": "high",
           "one_line": "Merger arb: will Delaware approve?", "web": True,
           "deal": {"present": True, "acquirer": "Aquarian", "offer_price": 70.0},
           "spread_pct": 32.1}
    gaps = [{"block": "options", "severity": "expected", "reason": "throttled",
             "label": "options positioning"}]
    signals = _signals(gaps, inv)
    merged, changed = merge_research_into_analysis({}, signals, ["BHF"])
    assert changed
    research = merged["by_symbol"]["BHF"]["research"]
    assert research["investigation"]["situation"] == "PENDING_ACQUISITION"
    assert research["evidence"]["degraded"] is True

    svc = LLMService.__new__(LLMService)  # no provider init needed
    captured = {}

    def fake_generate(prompt, system_prompt=None, **kw):
        captured["prompt"] = prompt
        captured["system"] = system_prompt
        return None

    svc.generate = fake_generate
    svc._client_cache, svc._active = {}, (None, None)
    svc.generate_recommendations(merged, signals, ["BHF"], basis="research+signals")
    prompt = captured["prompt"]
    assert ("Situation (web-researched): PENDING_ACQUISITION (high): "
            "Merger arb: will Delaware approve?; deal: Aquarian at $70.00, "
            "gross spread +32.1%") in prompt
    assert "WRITTEN WITHOUT expected evidence: options positioning (throttled)" in prompt
    assert "### 1. Situation & Key Figures" in prompt  # digest leads with it
    assert 'Where a "Situation" line is present' in captured["system"]


def test_synthesis_tolerates_string_offer_and_missing_spread():
    inv = {"situation": "STRATEGIC_REVIEW", "situation_confidence": "low",
           "one_line": "bid rumour", "web": False,
           "deal": {"present": True, "acquirer": None, "offer_price": "n/a"}}
    merged, _ = merge_research_into_analysis({}, _signals(None, inv), ["BHF"])
    svc = LLMService.__new__(LLMService)
    captured = {}
    svc.generate = lambda prompt, system_prompt=None, **kw: captured.setdefault("p", prompt) and None
    svc._client_cache, svc._active = {}, (None, None)
    svc.generate_recommendations(merged, _signals(None, inv), ["BHF"], basis="research+signals")
    assert "Situation (classified from supplied evidence): STRATEGIC_REVIEW (low): bid rumour; deal: n/a" in captured["p"]
