"""Situation classification + investigation: parsing, point-in-time gating,
block formatting, and how the research model wires it in."""

import json
from unittest.mock import patch

import pytest

from services import investigation_service as inv_svc
from services.investigation_service import (
    Investigation, _extract_json, format_investigation_block, investigate,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    inv_svc._CACHE.clear()
    yield
    inv_svc._CACHE.clear()


BHF_JSON = {
    "situation": "PENDING_ACQUISITION",
    "situation_confidence": "high",
    "one_line": "Merger-arb name: will Delaware approve Aquarian's $70 cash deal?",
    "deal": {
        "present": True, "acquirer": "Aquarian Capital", "offer_price": 70.0,
        "consideration": "cash", "announced": "2025-11-06",
        "expected_close": "2026",
        "approvals": [{"body": "Delaware Department of Insurance",
                       "status": "Form A review opened", "date": "2026-08-17",
                       "source": "news.delaware.gov, 2026-08-17, https://news.delaware.gov/x"}],
        "break_risk": "Regulator ties the review to the Walter probe.",
    },
    "key_figures": [
        {"name": "Rudy Sahay", "role": "Aquarian founder",
         "relevance": "Guggenheim Partners alumnus (sourced). inference: probe overhang transfers.",
         "source": "InvestmentNews, 2025-11-06, https://example.com/a"},
    ],
    "findings": [
        {"claim": "Federal prosecutors are probing Mark Walter's insurers.",
         "source": "WSJ", "date": "2026-08-16", "url": "https://wsj.com/x"},
    ],
    "dated_events": [{"date": "TBD", "what": "Delaware public hearing", "source": "x"}],
    "open_questions": ["Does Aquarian's financing depend on Guggenheim entities?"],
}


def _wrapped(payload: dict) -> str:
    return "Searching...\n```json\n" + json.dumps(payload) + "\n```"


class _FakeLLM:
    def __init__(self, text: str, searches: int = 5):
        self.text, self.searches = text, searches
        self.web_calls = 0
        self.plain_calls = 0

    def generate_with_web_search(self, prompt, system, **kw):
        self.web_calls += 1
        kw.get("usage_out", {}).update({"input_tokens": 1000, "output_tokens": 500,
                                        "model": "m", "provider": "anthropic"})
        return {"text": self.text, "searches": self.searches,
                "sources": [{"url": "https://wsj.com/x", "title": "WSJ"}],
                "stop_reason": "end_turn", "model": "m"}

    def generate(self, prompt, system, **kw):
        self.plain_calls += 1
        kw.get("usage_out", {}).update({"input_tokens": 100, "output_tokens": 50,
                                        "model": "m", "provider": "anthropic"})
        return self.text


def test_extract_json_takes_the_last_fence_and_tolerates_prose():
    text = "thinking ```json\n{\"a\": 1}\n``` more ```json\n{\"b\": 2}\n```"
    assert _extract_json(text) == {"b": 2}
    assert _extract_json("no json here") is None
    assert _extract_json("prefix {\"c\": 3} suffix") == {"c": 3}


def test_live_run_uses_web_and_computes_spread():
    fake = _FakeLLM(_wrapped(BHF_JSON))
    with patch("services.llm_service.get_llm", return_value=fake):
        inv = investigate("bhf", "2026-09-01", live=True, target="2026-09-02",
                          last_close=52.99, mode="auto")
    assert fake.web_calls == 1 and fake.plain_calls == 0
    assert inv.web is True
    assert inv.situation == "PENDING_ACQUISITION"
    assert inv.spread_pct == pytest.approx(32.1, abs=0.05)
    assert inv.searches == 5
    assert inv.input_tokens == 1000
    # Cached per (symbol, as_of, web, model): a retry does not re-search.
    with patch("services.llm_service.get_llm", return_value=fake):
        again = investigate("BHF", "2026-09-01", live=True, last_close=52.99, mode="auto")
    assert again is inv and fake.web_calls == 1


def test_backtest_never_touches_the_web():
    fake = _FakeLLM(_wrapped({**BHF_JSON, "deal": {"present": False}}))
    with patch("services.llm_service.get_llm", return_value=fake):
        inv = investigate("BHF", "2026-06-01", live=False, last_close=60.0, mode="auto")
    assert fake.web_calls == 0 and fake.plain_calls == 1
    assert inv.web is False
    assert inv.spread_pct is None
    block = format_investigation_block(inv, 60.0)
    assert "no web access on a historical as-of" in block


def test_mode_off_and_always_override_liveness():
    fake = _FakeLLM(_wrapped(BHF_JSON))
    with patch("services.llm_service.get_llm", return_value=fake):
        investigate("BHF", "2026-09-01", live=True, mode="off")
        assert fake.web_calls == 0
        investigate("BHF", "2026-06-01", live=False, mode="always")
        assert fake.web_calls == 1


def test_unparseable_response_raises_instead_of_a_blank_block():
    fake = _FakeLLM("I could not determine anything.")
    with patch("services.llm_service.get_llm", return_value=fake):
        with pytest.raises(RuntimeError, match="not parseable"):
            investigate("BHF", "2026-09-01", live=True, mode="auto")


def test_failed_investigation_is_cached_so_the_search_is_not_paid_twice():
    fake = _FakeLLM("I could not determine anything.")
    with patch("services.llm_service.get_llm", return_value=fake):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="not parseable"):
                investigate("BHF", "2026-09-01", live=True, mode="auto")
    assert fake.web_calls == 1


def test_prefetch_warms_the_cache_for_the_in_loop_call():
    from services.investigation_service import prefetch_many
    fake = _FakeLLM(_wrapped(BHF_JSON))
    with patch("services.llm_service.get_llm", return_value=fake), \
         patch("services.stock_data.get_company_profile", return_value="BHF"), \
         patch("services.bad_apples_service.analyze_symbol", side_effect=RuntimeError("no yf")):
        pool = prefetch_many(["BHF"], "2026-09-01", live=True, target="2026-09-02",
                             news_by_symbol={"BHF": []}, workers=2)
        pool.shutdown(wait=True)
        inv = investigate("BHF", "2026-09-01", live=True, mode="auto", last_close=52.99)
    assert fake.web_calls == 1
    assert inv.situation == "PENDING_ACQUISITION"


def test_unknown_situation_label_falls_back_to_other():
    fake = _FakeLLM(_wrapped({**BHF_JSON, "situation": "TAKEOVER_TARGET"}))
    with patch("services.llm_service.get_llm", return_value=fake):
        inv = investigate("BHF", "2026-09-01", live=True, mode="auto")
    assert inv.situation == "OTHER"


def test_block_carries_deal_terms_spread_figures_and_sources():
    inv = Investigation(symbol="BHF", as_of="2026-09-01", web=True,
                        situation="PENDING_ACQUISITION", situation_confidence="high",
                        one_line=BHF_JSON["one_line"], deal=BHF_JSON["deal"],
                        key_figures=BHF_JSON["key_figures"],
                        findings=BHF_JSON["findings"],
                        dated_events=BHF_JSON["dated_events"],
                        open_questions=BHF_JSON["open_questions"],
                        sources=[{"url": "u", "title": "t"}], searches=4,
                        spread_pct=32.1)
    block = format_investigation_block(inv, 52.99)
    assert block.startswith("[BHF — situation & investigation as of 2026-09-01 (web research")
    assert "Deal: Aquarian Capital, $70.00 per share cash, announced 2025-11-06" in block
    assert "$52.99 vs $70.00 = +32.1%" in block
    assert "Delaware Department of Insurance — Form A review opened (2026-08-17)" in block
    assert "Rudy Sahay — Aquarian founder" in block
    assert "(src: WSJ | 2026-08-16 | https://wsj.com/x)" in block
    assert "Open questions: Does Aquarian's financing" in block
    assert "4 web searches, 1 sources consulted" in block


def test_research_model_records_gap_when_investigation_fails():
    """A failed investigation is an EXPECTED gap: the report is written,
    but the ledger, the emitted event and the details all say so."""
    from services.evidence_contract import EvidenceLedger
    from models.trading_agents_model import TradingAgentsModel
    import pandas as pd

    idx = pd.bdate_range("2025-09-01", "2026-09-01")
    df = pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.0,
                       "Volume": 1_000_000}, index=idx)
    ledger = EvidenceLedger("BHF")

    def boom(*a, **k):
        raise RuntimeError("provider down")

    model = TradingAgentsModel()
    with patch("services.investigation_service.investigate", side_effect=boom), \
         patch.object(TradingAgentsModel, "_is_live", return_value=True), \
         patch("utils.events.get_upcoming_events", return_value={}), \
         patch("services.stock_data.fetch_stock_data", return_value=df), \
         patch("services.stock_data.get_company_profile", return_value="BHF — insurer"):
        blocks, inv = model._build_extra_context(
            "BHF", df, "2026-09-01", evidence={"investigation"}, ledger=ledger)
    assert inv is None
    gap = next(g for g in ledger.expected_gaps() if g.block == "investigation")
    assert "provider down" in gap.reason
    assert any(b.startswith("[SPY regime") for b in blocks)
