"""The dialog's cost switches switch off what they say they switch off.

Found 2026-09-05 while explaining the evidence checkboxes: the web-research
tool was described as the cost-saving switch, and with it off the pipeline
still bought a web-free classification call per flagged symbol (and per
symbol per day in a backtest, where the path strips the tool). The other
three cases here share the shape: a spend that no switch reached.
"""

from unittest.mock import patch

import pandas as pd
import pytest


def _bars():
    idx = pd.bdate_range("2025-09-01", "2026-09-01")
    return pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0,
                         "Close": 50.0, "Volume": 1_000_000}, index=idx)


def _flagged(*a, **k):
    """An anomaly scan that found something, so the gate would be open."""
    anomaly = {"key": "options_skew", "title": "Options skew",
               "severity": 0.6, "facts": [], "question": "why?",
               "evidence_key": "options"}
    return [anomaly], ["the options chain"], False


class TestWebResearchOffBuysNoInvestigation:

    def test_a_flagged_symbol_is_not_investigated_without_the_tool(self):
        from services.evidence_contract import EvidenceLedger
        from models.trading_agents_model import TradingAgentsModel

        df = _bars()
        ledger = EvidenceLedger("ETR")
        called = []

        def _investigate(*a, **k):
            called.append(k)
            raise AssertionError("web research is off: no model call may be bought")

        model = TradingAgentsModel()
        with patch.object(TradingAgentsModel, "_detect_anomalies", _flagged), \
             patch("services.investigation_service.investigate", _investigate), \
             patch("utils.events.get_upcoming_events", return_value={}), \
             patch("services.stock_data.fetch_stock_data", return_value=df), \
             patch("services.stock_data.get_company_profile", return_value="ETR: utility"):
            _, inv, anomalies, _, _ = model._build_extra_context(
                "ETR", df, "2026-09-01", evidence={"investigation"},
                ledger=ledger, tools=[])

        assert anomalies, "fixture is meant to be a flagged symbol"
        assert not called
        assert inv is None
        skip = next(g for g in ledger.skipped if g.block == "investigation")
        assert "web research is off" in skip.reason
        assert "investigation" not in [g.block for g in ledger.gaps], (
            "a deliberate saving was recorded as a gap")

    def test_the_tool_on_still_buys_it(self):
        from services.evidence_contract import EvidenceLedger
        from services.investigation_service import Investigation
        from models.trading_agents_model import TradingAgentsModel

        df = _bars()
        ledger = EvidenceLedger("ETR")
        called = []

        def _investigate(symbol, as_of, **k):
            called.append(k)
            return Investigation(symbol=symbol, as_of=as_of, web=k["web"],
                                 situation="LEGAL_REGULATORY_OVERHANG")

        model = TradingAgentsModel()
        with patch.object(TradingAgentsModel, "_detect_anomalies", _flagged), \
             patch("services.investigation_service.investigate", _investigate), \
             patch("services.investigation_service.research_questions",
                   return_value=[]), \
             patch("utils.events.get_upcoming_events", return_value={}), \
             patch("services.stock_data.fetch_stock_data", return_value=df), \
             patch("services.stock_data.get_company_profile", return_value="ETR: utility"):
            _, inv, _, _, _ = model._build_extra_context(
                "ETR", df, "2026-09-01", evidence={"investigation"},
                ledger=ledger, tools=["web_research"])

        assert len(called) == 1
        assert called[0]["web"] is True
        # The scan already flagged the name: no web-free triage in front.
        assert called[0]["triage"] is False
        assert inv is not None and inv.situation == "LEGAL_REGULATORY_OVERHANG"
        assert "investigation" in ledger.present


class TestTriageIsASwitch:

    @pytest.fixture(autouse=True)
    def _clean(self):
        from services import investigation_service as inv_svc
        inv_svc._CACHE.clear()
        yield
        inv_svc._CACHE.clear()

    def _record(self, monkeypatch):
        from services import investigation_service as inv_svc
        seen = []

        def fake(key, symbol, as_of, *, web, **k):
            seen.append(web)
            return inv_svc.Investigation(symbol=symbol, as_of=as_of, web=web,
                                         situation="OTHER")

        monkeypatch.setattr(inv_svc, "_investigate_uncached", fake)
        # The default skip list is non-empty, which is what arms the triage.
        assert inv_svc.MODEL.INVESTIGATION_WEB_SKIP
        return seen

    def test_default_still_triages_before_searching(self, monkeypatch):
        from services import investigation_service as inv_svc
        seen = self._record(monkeypatch)
        inv_svc.investigate("TRIAGEA", "2026-09-01", web=True)
        assert seen == [False, True]

    def test_a_caller_that_knows_better_skips_the_triage(self, monkeypatch):
        from services import investigation_service as inv_svc
        seen = self._record(monkeypatch)
        inv_svc.investigate("TRIAGEB", "2026-09-01", web=True, triage=False)
        assert seen == [True]


class TestTheCeilingFailsClosed:

    def test_a_priced_call_counts_at_its_price(self):
        from services import usage_service as us
        assert us._ceiling_charge(0.01, 0.02, model="gpt-5.6-luna",
                                  provider="openai", input_tokens=1, output_tokens=1,
                                  searches=2) == pytest.approx(0.03)

    def test_an_unpriced_model_counts_at_the_dearest_rate(self):
        from config import LLM_PRICING, WEB_SEARCH_PRICING
        from services import usage_service as us
        charge = us._ceiling_charge(None, None, model="mystery-model",
                                    provider="nowhere", input_tokens=1_000_000,
                                    output_tokens=0, searches=10)
        dearest_in = max(r["input"] for r in LLM_PRICING.values())
        dearest_search = max(WEB_SEARCH_PRICING.values())
        assert charge == pytest.approx(dearest_in + 10 * dearest_search / 1000)
        assert charge > 0

    def test_nothing_used_costs_nothing(self):
        from services import usage_service as us
        assert us._ceiling_charge(None, None, model="mystery", provider=None,
                                  input_tokens=0, output_tokens=0, searches=0) == 0.0


class TestContinuityPostPassIsNotBoughtOnBacktests:

    def _run(self, **kw):
        from models.trading_agents_model import TradingAgentsModel

        seen = {}

        class _Agent:
            def analyze(self, symbol, as_of, **kwargs):
                seen.update(kwargs)
                return {"decision": "HOLD", "confidence": 0.5,
                        "raw_response": "", "structured": {}, "provenance": {}}

        model = TradingAgentsModel()
        with patch("models.research_backend.get_research_agent",
                   return_value=_Agent()), \
             patch.object(TradingAgentsModel, "_build_extra_context",
                          return_value=([], None, [], [], False)), \
             patch.object(TradingAgentsModel, "_track_record_line",
                          return_value=""), \
             patch.object(TradingAgentsModel, "_grounded_confidence",
                          return_value=(0.5, 0.5, "none", 0)):
            model._run_analysis("X", _bars(), as_of="2026-09-01",
                                news_lookback_days=14, **kw)
        return seen

    def test_a_backtest_keeps_the_deterministic_record(self):
        assert self._run(is_backtest=True)["use_continuity"] is False

    def test_a_live_run_still_writes_the_line(self):
        assert self._run(is_backtest=False)["use_continuity"] is True

    def test_an_explicit_choice_wins(self):
        assert self._run(is_backtest=True, use_continuity=True)["use_continuity"] is True
