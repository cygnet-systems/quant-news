"""Situation classification + investigation: parsing, point-in-time gating,
block formatting, and how the research model wires it in."""

import json
import time
from unittest.mock import patch

import pytest

from config import MODEL
from services import investigation_service as inv_svc
from services.investigation_service import (
    Investigation, _extract_json, format_investigation_block, investigate,
)
from services.rate_limiter import FileSemaphore


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


class _TriageLLM(_FakeLLM):
    """Web-free call returns one label, the web call another."""
    def __init__(self, plain_label, web_label="PENDING_ACQUISITION"):
        super().__init__(_wrapped({**BHF_JSON, "situation": web_label}))
        self.plain_text = _wrapped({**BHF_JSON, "situation": plain_label,
                                    "deal": {"present": False}})
    def generate(self, prompt, system, **kw):
        self.plain_calls += 1
        kw.get("usage_out", {}).update({"input_tokens": 100, "output_tokens": 50,
                                        "model": "m", "provider": "anthropic"})
        return self.plain_text


def test_extract_json_takes_the_last_fence_and_tolerates_prose():
    text = "thinking ```json\n{\"a\": 1}\n``` more ```json\n{\"b\": 2}\n```"
    assert _extract_json(text) == {"b": 2}
    assert _extract_json("no json here") is None
    assert _extract_json("prefix {\"c\": 3} suffix") == {"c": 3}


def test_live_run_uses_web_and_computes_spread():
    fake = _FakeLLM(_wrapped(BHF_JSON))
    with patch("services.llm_service.get_llm", return_value=fake):
        inv = investigate("bhf", "2026-09-01", web=True, target="2026-09-02",
                          last_close=52.99)
    # One web-free triage call, then the search (the triage label comes
    # from the same fake JSON, which is not a skip label).
    assert fake.web_calls == 1 and fake.plain_calls == 1
    assert inv.web is True
    assert inv.situation == "PENDING_ACQUISITION"
    assert inv.spread_pct == pytest.approx(32.1, abs=0.05)
    assert inv.searches == 5
    assert inv.input_tokens == 1000
    # Cached per (symbol, as_of, web, model): a retry does not re-search.
    with patch("services.llm_service.get_llm", return_value=fake):
        again = investigate("BHF", "2026-09-01", web=True, last_close=52.99)
    assert again is inv and fake.web_calls == 1


def test_backtest_never_touches_the_web():
    fake = _FakeLLM(_wrapped({**BHF_JSON, "deal": {"present": False}}))
    with patch("services.llm_service.get_llm", return_value=fake):
        inv = investigate("BHF", "2026-06-01", web=False, last_close=60.0)
    assert fake.web_calls == 0 and fake.plain_calls == 1
    assert inv.web is False
    assert inv.spread_pct is None
    block = format_investigation_block(inv, 60.0)
    assert "no web access on a historical as-of" in block


def test_web_is_off_unless_the_run_asked_for_the_tool():
    fake = _FakeLLM(_wrapped(BHF_JSON))
    with patch("services.llm_service.get_llm", return_value=fake):
        investigate("BHF", "2026-09-01", web=False)
        assert fake.web_calls == 0
        investigate("BHF", "2026-06-01", web=True)
        assert fake.web_calls == 1


def test_unparseable_response_raises_instead_of_a_blank_block():
    fake = _FakeLLM("I could not determine anything.")
    with patch("services.llm_service.get_llm", return_value=fake):
        with pytest.raises(RuntimeError, match="not parseable"):
            investigate("BHF", "2026-09-01", web=True)


def test_failed_investigation_is_cached_so_the_search_is_not_paid_twice():
    # Triage classifies fine; the web call comes back unparseable. The
    # failure is cached under the web key, so a retry costs nothing.
    fake = _TriageLLM("LEADERSHIP_CHANGE")
    fake.text = "I could not determine anything."
    with patch("services.llm_service.get_llm", return_value=fake):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="not parseable"):
                investigate("BHF", "2026-09-01", web=True)
    assert fake.web_calls == 1 and fake.plain_calls == 1


def test_failed_triage_does_not_block_the_search():
    fake = _FakeLLM(_wrapped(BHF_JSON))
    fake.generate = lambda *a, **k: "garbage"          # triage unparseable
    with patch("services.llm_service.get_llm", return_value=fake):
        # Same evidence the prefetch assembled: the cache key carries a digest
        # of it, so this is what the in-loop call actually passes.
        inv = investigate("BHF", "2026-09-01", web=True, target="2026-09-02",
                          profile="BHF", last_close=52.99)
    assert fake.web_calls == 1 and inv.web is True


def test_prefetch_warms_the_cache_for_the_in_loop_call():
    from services.investigation_service import prefetch_many
    fake = _FakeLLM(_wrapped(BHF_JSON))
    with patch("services.llm_service.get_llm", return_value=fake), \
         patch("services.stock_data.get_company_profile", return_value="BHF"), \
         patch("services.bad_apples_service.analyze_symbol", side_effect=RuntimeError("no yf")):
        pool = prefetch_many(["BHF"], "2026-09-01", web=True, target="2026-09-02",
                             news_by_symbol={"BHF": []}, workers=2)
        pool.shutdown(wait=True)
        # Same evidence the prefetch assembled: the cache key carries a digest
        # of it, so this is what the in-loop call actually passes.
        inv = investigate("BHF", "2026-09-01", web=True, target="2026-09-02",
                          profile="BHF", last_close=52.99)
    assert fake.web_calls == 1
    assert inv.situation == "PENDING_ACQUISITION"


def test_backend_default_is_no_web():
    fake = _FakeLLM(_wrapped(BHF_JSON))
    with patch("services.llm_service.get_llm", return_value=fake):
        investigate("BHF", "2026-09-01")
    assert fake.web_calls == 0 and fake.plain_calls == 1


def test_default_tools_follow_the_target_date():
    from datetime import date, timedelta
    from services.investigation_service import default_tools
    today = date.today()
    assert default_tools(today) == ["web_research"]
    assert default_tools(today + timedelta(days=1)) == ["web_research"]
    assert default_tools(today - timedelta(days=1)) == []
    assert default_tools("2026-01-05") == []
    assert default_tools(None) == ["web_research"]


def test_momentum_only_names_never_pay_for_a_search():
    fake = _TriageLLM("MOMENTUM_ONLY")
    with patch("services.llm_service.get_llm", return_value=fake):
        inv = investigate("BHF", "2026-09-01", web=True, last_close=52.99)
    assert fake.plain_calls == 1 and fake.web_calls == 0
    assert inv.web is False and inv.situation == "MOMENTUM_ONLY"
    assert "web research not spent" in inv.web_skipped
    assert "web research not spent" in format_investigation_block(inv, 52.99)
    # Cached under the web=True key: a second call costs nothing.
    with patch("services.llm_service.get_llm", return_value=fake):
        investigate("BHF", "2026-09-01", web=True, last_close=52.99)
    assert fake.plain_calls == 1 and fake.web_calls == 0


def test_special_situations_still_get_the_web():
    fake = _TriageLLM("LEADERSHIP_CHANGE")
    with patch("services.llm_service.get_llm", return_value=fake):
        inv = investigate("BHF", "2026-09-01", web=True, last_close=52.99)
    assert fake.plain_calls == 1 and fake.web_calls == 1
    assert inv.web is True and inv.situation == "PENDING_ACQUISITION"
    assert inv.spread_pct == pytest.approx(32.1, abs=0.05)


def test_unknown_situation_label_falls_back_to_other():
    fake = _FakeLLM(_wrapped({**BHF_JSON, "situation": "TAKEOVER_TARGET"}))
    with patch("services.llm_service.get_llm", return_value=fake):
        inv = investigate("BHF", "2026-09-01", web=True)
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
    assert block.startswith("[BHF: situation & investigation as of 2026-09-01 (web research")
    assert "Deal: Aquarian Capital, $70.00 per share cash, announced 2025-11-06" in block
    assert "$52.99 vs $70.00 = +32.1%" in block
    assert "Delaware Department of Insurance: Form A review opened (2026-08-17)" in block
    assert "Rudy Sahay: Aquarian founder" in block
    assert "(src: WSJ | 2026-08-16 | https://wsj.com/x)" in block
    assert "Open questions: Does Aquarian's financing" in block
    assert "4 web searches, 1 sources consulted" in block


def test_research_model_records_gap_when_investigation_fails(monkeypatch):
    """A failed investigation is an EXPECTED gap: the report is written,
    but the ledger, the emitted event and the details all say so.

    Gate off: this is about the FAILURE path, and a gated run never calls
    the investigator at all. The distinction is the point of
    test_a_gated_off_investigation_is_a_skip_not_a_gap below.
    """
    import dataclasses

    from config import MODEL
    from services.evidence_contract import EvidenceLedger
    from models import trading_agents_model as tam
    from models.trading_agents_model import TradingAgentsModel
    import pandas as pd

    monkeypatch.setattr(tam, "MODEL", dataclasses.replace(
        MODEL, INVESTIGATE_ONLY_ANOMALIES=False))

    idx = pd.bdate_range("2025-09-01", "2026-09-01")
    df = pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.0,
                       "Volume": 1_000_000}, index=idx)
    ledger = EvidenceLedger("BHF")

    def boom(*a, **k):
        raise RuntimeError("provider down")

    model = TradingAgentsModel()
    with patch("services.investigation_service.investigate", side_effect=boom), \
         patch("utils.events.get_upcoming_events", return_value={}), \
         patch("services.stock_data.fetch_stock_data", return_value=df), \
         patch("services.stock_data.get_company_profile", return_value="BHF: insurer"):
        blocks, inv, _, _, _ = model._build_extra_context(
            "BHF", df, "2026-09-01", evidence={"investigation"}, ledger=ledger,
            tools=["web_research"])
    assert inv is None
    gap = next(g for g in ledger.expected_gaps() if g.block == "investigation")
    assert "provider down" in gap.reason
    assert any(b.startswith("[SPY regime") for b in blocks)


def test_a_gated_off_investigation_is_a_skip_not_a_gap():
    """A symbol nothing stood out for was not investigated BY DECISION.

    That must not read as an evidence gap. A gap means the run wanted the
    block and could not get it, marks the report degraded, and tells the
    model to say the evidence was unavailable. Calling a deliberate saving a
    failure would mark every quiet symbol degraded and put a false statement
    about the run's own provenance in the report -- the exact class of thing
    the evidence contract exists to prevent.
    """
    from services.evidence_contract import EvidenceLedger
    from models.trading_agents_model import TradingAgentsModel
    import pandas as pd

    idx = pd.bdate_range("2025-09-01", "2026-09-01")
    df = pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.0,
                       "Volume": 1_000_000}, index=idx)
    ledger = EvidenceLedger("BHF")
    called = []

    def _investigate(*a, **k):
        called.append(a)
        raise AssertionError("a quiet symbol must not buy an investigation")

    model = TradingAgentsModel()
    with patch("services.investigation_service.investigate", _investigate), \
         patch("utils.events.get_upcoming_events", return_value={}), \
         patch("services.stock_data.fetch_stock_data", return_value=df), \
         patch("services.stock_data.get_company_profile", return_value="BHF: insurer"):
        _, inv, anomalies, _, _ = model._build_extra_context(
            "BHF", df, "2026-09-01", evidence={"investigation"}, ledger=ledger,
            tools=["web_research"])

    assert not anomalies, "fixture is meant to be a quiet symbol"
    assert not called, "the gate let a paid investigation through"
    assert inv is None
    assert not [g for g in ledger.expected_gaps() if g.block == "investigation"], (
        "a deliberate skip was recorded as a gap; every quiet symbol would "
        "mark its run degraded")
    # Not `ledger.degraded is False`: this fixture mocks the events calendar
    # away, which is its own unrelated gap. The claim under test is narrower
    # and exact -- the skip contributes nothing to degradation.
    assert "investigation" not in [g.block for g in ledger.gaps]
    skip = next(g for g in ledger.skipped if g.block == "investigation")
    assert "nothing stood out" in skip.reason
    assert skip.severity == "optional", (
        "a skip must never carry a severity that can degrade a run")


def test_the_dossier_supersedes_the_political_blocks_congress_half():
    """Both keys are default-on, and both cover congressional trades over the
    same window from the same store. Building both would put two counts of
    the same disclosures in one prompt."""
    from services.evidence_contract import EvidenceLedger
    from models.trading_agents_model import TradingAgentsModel
    import pandas as pd

    idx = pd.bdate_range("2025-09-01", "2026-09-01")
    df = pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.0,
                       "Volume": 1_000_000}, index=idx)
    seen = {}

    def political(symbol, as_of, include_congress=True):
        seen["include_congress"] = include_congress
        return (["[BHF: institutional (13F) holder flows]\nnothing"], [])

    model = TradingAgentsModel()
    with patch("services.political_service.political_blocks", political), \
         patch("services.politician_dossier.politician_block",
               return_value=("[BHF: congressional dossier]\nnothing", [])), \
         patch("utils.events.get_upcoming_events", return_value={}), \
         patch("services.stock_data.fetch_stock_data", return_value=df):
        blocks, _, _, _, _ = model._build_extra_context(
            "BHF", df, "2026-09-01", evidence={"political", "politicians"},
            ledger=EvidenceLedger("BHF"))
    assert seen["include_congress"] is False
    assert sum("congressional" in b for b in blocks) == 1

    with patch("services.political_service.political_blocks", political), \
         patch("utils.events.get_upcoming_events", return_value={}), \
         patch("services.stock_data.fetch_stock_data", return_value=df):
        model._build_extra_context("BHF", df, "2026-09-01",
                                   evidence={"political"},
                                   ledger=EvidenceLedger("BHF"))
    assert seen["include_congress"] is True


def test_a_rendered_block_is_present_evidence_even_with_a_stale_source():
    """politician_block returns a dossier AND a problem when one of its two
    stored sources did not refresh. The dossier is in the prompt, so the
    ledger has to say the report was built with it; recording a gap against
    text the model is reading is how a complete report gets labelled
    degraded."""
    from services.evidence_contract import EvidenceLedger
    from models.trading_agents_model import TradingAgentsModel
    import pandas as pd

    idx = pd.bdate_range("2025-09-01", "2026-09-01")
    df = pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.0,
                       "Volume": 1_000_000}, index=idx)
    model = TradingAgentsModel()
    ledger = EvidenceLedger("BHF")
    dossier = "[BHF: congressional dossier]\nDan Newhouse bought."
    stale = "[BHF: insider transactions (SEC Form 4)]\nJensen Huang sold."

    with patch("services.politician_dossier.politician_block",
               return_value=(dossier, ["roster: unavailable (throttled)"])), \
         patch("services.insider_service.insider_block",
               return_value=(stale, ["insider top-up: unavailable"])), \
         patch("utils.events.get_upcoming_events", return_value={}), \
         patch("services.stock_data.fetch_stock_data", return_value=df):
        blocks, _, _, _, _ = model._build_extra_context(
            "BHF", df, "2026-09-01", evidence={"politicians", "insiders"},
            ledger=ledger)

    assert "Dan Newhouse bought." in "\n".join(blocks)
    assert "Jensen Huang sold." in "\n".join(blocks)
    assert {"politicians", "insiders"} <= set(ledger.present)
    assert [g.block for g in ledger.gaps
            if g.block in ("politicians", "insiders")] == []


def test_the_political_block_follows_the_same_rule_as_its_siblings():
    """political_blocks' two halves fail independently, so a 13F block can be
    written while the congressional fetch raised. Requiring an empty problems
    list put that rendered block in the prompt AND a gap saying the same
    evidence was unavailable next to it."""
    from services.evidence_contract import EvidenceLedger
    from models.trading_agents_model import TradingAgentsModel
    import pandas as pd

    idx = pd.bdate_range("2025-09-01", "2026-09-01")
    df = pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.0,
                       "Volume": 1_000_000}, index=idx)
    model = TradingAgentsModel()
    ledger = EvidenceLedger("BHF")
    holders = "[BHF: institutional (13F) holder flows]\nTwo holders added."

    with patch("services.political_service.political_blocks",
               return_value=([holders], ["congressional trades: throttled"])), \
         patch("utils.events.get_upcoming_events", return_value={}), \
         patch("services.stock_data.fetch_stock_data", return_value=df):
        blocks, _, _, _, _ = model._build_extra_context(
            "BHF", df, "2026-09-01", evidence={"political"}, ledger=ledger)

    assert "Two holders added." in "\n".join(blocks)
    assert "political" in set(ledger.present)
    assert [g.block for g in ledger.gaps if g.block == "political"] == []


def test_the_political_caveat_rides_in_the_block_text():
    """Where the model reading the rows can see it, which is the only place
    a caveat about those rows does any work."""
    from services import political_service

    with patch.object(political_service, "get_congress_trades",
                      side_effect=RuntimeError("throttled")), \
         patch.object(political_service, "get_institutional_holdings",
                      return_value={"symbol": "BHF"}), \
         patch.object(political_service, "format_institutional_block",
                      return_value="[BHF: institutional (13F) holder flows]\nrows"):
        blocks, problems = political_service.political_blocks("BHF", "2026-09-01")

    assert len(blocks) == 1 and problems
    assert "Not every source behind this section refreshed" in blocks[0]
    assert "throttled" in blocks[0]


class TestWebResearchSlots:
    """One box, however many runs and processes: the open-web calls are
    bounded.

    The Alpha Vantage token bucket does not cover this stage (different
    vendor, and it paces calls per minute rather than counting the ones in
    flight), so this semaphore is the only thing standing between ten
    simultaneous runs and ten prefetch pools' worth of provider connections.
    It has to be the file-backed one: a manual run's model stage is a forked
    background-callback subprocess, so a threading.Semaphore here bounds one
    run's own fan-out and nothing between two people. The bound itself is
    tested in test_rate_limiter; what is tested here is that these three
    call sites take it and the triage does not.
    """

    @pytest.fixture(autouse=True)
    def _clear_answers(self):
        inv_svc._ANSWER_CACHE.clear()
        yield
        inv_svc._ANSWER_CACHE.clear()

    def _bounded(self, monkeypatch, tmp_path, size=1):
        monkeypatch.setattr(inv_svc, "_WEB_SLOTS", FileSemaphore(
            "web_research_test", size, state_dir=tmp_path, poll_s=0.01))

    def test_the_shipped_bound_is_the_cross_process_one(self):
        assert isinstance(inv_svc._WEB_SLOTS, FileSemaphore)
        assert inv_svc._WEB_SLOTS.slots == MODEL.WEB_RESEARCH_CONCURRENCY

    def test_web_investigations_do_not_exceed_the_slot_count(self, monkeypatch,
                                                             tmp_path):
        import threading
        self._bounded(monkeypatch, tmp_path, size=1)
        seen, live = [], []
        gate = threading.Lock()

        class _Counting(_FakeLLM):
            def generate_with_web_search(self, prompt, system, **kw):
                with gate:
                    live.append(1)
                    seen.append(len(live))
                # Long enough that unbounded threads WOULD overlap: without
                # the semaphore this test sees 4.
                time.sleep(0.05)
                try:
                    return super().generate_with_web_search(prompt, system, **kw)
                finally:
                    with gate:
                        live.pop()

        fake = _Counting(_wrapped(BHF_JSON))
        with patch("services.llm_service.get_llm", return_value=fake):
            threads = [threading.Thread(
                target=investigate, args=(sym, "2026-09-01"),
                kwargs={"web": True, "target": "2026-09-02"})
                for sym in ("AAA", "BBB", "CCC", "DDD")]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert fake.web_calls == 4
        assert max(seen) == 1

    def test_question_research_shares_the_same_slots(self, monkeypatch,
                                                     tmp_path):
        # The three questions a symbol raises fan out concurrently; they are
        # web searches like any other and queue with the investigations.
        import threading
        self._bounded(monkeypatch, tmp_path, size=1)
        seen, live = [], []
        gate = threading.Lock()

        def _one(symbol, as_of, question, **kw):
            with gate:
                live.append(1)
                seen.append(len(live))
            time.sleep(0.05)
            with gate:
                live.pop()
            return {"question": question, "finding": "x", "citations": []}

        monkeypatch.setattr(inv_svc, "_research_one", _one)
        answers = inv_svc.research_questions(
            "AAA", "2026-09-01", ["q1", "q2", "q3"], web=True)

        assert len(answers) == 3
        assert max(seen) == 1

    def test_the_web_free_triage_never_takes_a_slot(self, monkeypatch, tmp_path):
        # Held by someone else for the whole call: a triage that waited on
        # a slot it does not need would deadlock a run behind a run.
        self._bounded(monkeypatch, tmp_path, size=1)
        inv_svc._WEB_SLOTS.acquire()
        try:
            fake = _FakeLLM(_wrapped(BHF_JSON))
            with patch("services.llm_service.get_llm", return_value=fake):
                inv = investigate("BHF", "2026-09-01", web=False)
        finally:
            inv_svc._WEB_SLOTS.release()
        assert fake.plain_calls == 1 and inv.web is False

    def test_a_failed_search_gives_its_slot_back(self, monkeypatch, tmp_path):
        self._bounded(monkeypatch, tmp_path, size=1)

        class _Boom(_FakeLLM):
            def generate_with_web_search(self, prompt, system, **kw):
                raise RuntimeError("provider down")

        with patch("services.llm_service.get_llm", return_value=_Boom("")):
            with pytest.raises(RuntimeError):
                investigate("AAA", "2026-09-01", web=True)
        fake = _FakeLLM(_wrapped(BHF_JSON))
        with patch("services.llm_service.get_llm", return_value=fake):
            assert investigate("BBB", "2026-09-01", web=True).web is True


class TestTheBacktestRuleIsEnforcedOnThePath:
    """The open web is off for a backtest wherever the run came from.

    Run with the anomaly gate OFF, because the witness these tests read is
    the prefetch pool's ``web`` argument and the gate removes the pool. The
    rule under test is about the tools list run_predictions builds, which is
    the same either way.

    The rule had two witnesses and neither was on the wire: the dialog's
    ``preset_run_tools`` drops web research for a past target and
    ``apply_run_preset`` rewrites the control when the picker moves, but a
    box ticked BEFORE the date was moved back reached the confirm intact,
    and the confirm records the checklist verbatim. A retry, a stored run
    row and a saved schedule whose target has since gone by all copy that
    config forward. run_predictions is where every surface meets, so that
    is where the tools are stripped, and both readers of the switch -- the
    prefetch pool and trading_agents_model, which takes the same list
    through run_report_for_symbol -- see the stripped set.
    """

    @pytest.fixture(autouse=True)
    def _ungated(self, monkeypatch):
        import dataclasses

        from config import MODEL
        from services import analysis_runner as ar

        monkeypatch.setattr(ar, "MODEL", dataclasses.replace(
            MODEL, INVESTIGATE_ONLY_ANOMALIES=False))

    def _run(self, target, cutoff, tools):
        """run_predictions with the models stubbed; returns the web flag the
        prefetch pool was started with and the tools the report stage got."""
        from contextlib import nullcontext
        from datetime import date

        import pandas as pd

        from services import analysis_runner

        seen = {}

        class _Service:
            def predict_symbol_no_store(self, symbol, df, **kw):
                seen["report_tools"] = kw.get("tools")
                return {}

        class _Pool:
            def shutdown(self, wait=False):
                pass

        def _prefetch(remaining, as_of, **kw):
            seen["prefetch_web"] = kw.get("web")
            return _Pool()

        idx = pd.date_range("2026-06-01", periods=60, freq="B")
        frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                              "Close": 1.0, "Volume": 1}, index=idx)
        symbols = ["A", "B"]
        spy = pd.DataFrame({"Close": [1.0] * 10},
                           index=pd.date_range("2026-08-01", periods=10,
                                               freq="B"))
        with patch("services.prediction_service.get_prediction_service",
                   return_value=_Service()), \
             patch("services.stock_data.fetch_stock_data", return_value=spy), \
             patch("services.investigation_service.prefetch_many",
                   side_effect=_prefetch), \
             patch("services.analysis_runner._reusable_predictions",
                   return_value=None), \
             patch("services.usage_service.track", return_value=nullcontext()):
            analysis_runner.run_predictions(
                symbols,
                {s: {"prices": frame.to_json(date_format="iso")}
                 for s in symbols},
                {s: [] for s in symbols},
                target_date=target, cutoff_date=cutoff,
                models={"trading_agents"}, evidence=["investigation"],
                news_lookback_days=14, tools=tools)
        return seen

    def test_a_ticked_box_cannot_put_the_web_into_a_backtest(self):
        from datetime import date, timedelta

        past = date.today() - timedelta(days=30)
        seen = self._run(past, past - timedelta(days=1), ["web_research"])
        assert seen["prefetch_web"] is False
        assert "web_research" not in (seen["report_tools"] or [])

    def test_a_forward_run_keeps_the_tool_it_asked_for(self):
        from datetime import date, timedelta

        ahead = date.today() + timedelta(days=1)
        seen = self._run(ahead, date.today(), ["web_research"])
        assert seen["prefetch_web"] is True
        assert "web_research" in (seen["report_tools"] or [])
