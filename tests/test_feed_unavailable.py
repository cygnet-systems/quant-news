"""A feed that did not answer stops the symbol's report and lands in the
activity trail. A feed that answered with nothing is still a gap.

The distinction is the whole contract: no Form 4 rows this window is a
finding about the symbol, a throttled Form 4 endpoint is not, and the two
used to be one ``None`` and one gap. Every site below is exercised through
``_build_extra_context`` with the outside world stubbed, so what is pinned
is the site's classification of its source's answer, not the vendor.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from models.trading_agents_model import TradingAgentsModel
from services import bad_apples_service, options_service
from services.alpha_vantage import AlphaVantageUnavailable
from services.evidence_contract import (
    EvidenceLedger, FeedUnavailable, MissingRequiredEvidence,
)

AS_OF = "2026-09-01"


def frame():
    idx = pd.bdate_range("2025-09-01", AS_OF)
    return pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0,
                         "Close": 50.0, "Volume": 1_000_000}, index=idx)


def build(evidence, **patches):
    """_build_extra_context with the price feed answering and every other
    door stubbed as the test says."""
    df = frame()
    ledger = EvidenceLedger("BHF")
    stubs = {
        "utils.events.get_upcoming_events": {"return_value": {
            "next_earnings": None, "ex_dividend": None, "in_window": False,
            "earnings_in_window": False, "exdiv_in_window": False,
            "window_end": "2026-09-08"}},
        "services.stock_data.fetch_stock_data": {"return_value": df},
        "services.stock_data.get_company_profile": {"return_value": "BHF"},
        "services.terminal_data.filings_block": {"return_value": ""},
        "services.terminal_data.market_context_block": {"return_value": ""},
    }
    stubs.update(patches)
    from contextlib import ExitStack
    with ExitStack() as stack:
        for target, kw in stubs.items():
            stack.enter_context(patch(target, **kw))
        blocks, *_ = TradingAgentsModel()._build_extra_context(
            "BHF", df, AS_OF, evidence=set(evidence), ledger=ledger)
    return blocks, ledger


class TestContract:
    def test_unavailable_always_raises_whatever_the_blocks_severity(self):
        ledger = EvidenceLedger("BHF")
        for block in ("insiders", "political", "options", "events"):
            with pytest.raises(FeedUnavailable) as exc:
                ledger.unavailable(block, "vendor throttled")
            assert exc.value.block == block and exc.value.kind == "feed_unavailable"
            assert "feed unavailable" in str(exc.value)
        assert ledger.gaps == [] and ledger.skipped == []

    def test_it_is_a_missing_required_evidence_so_the_one_handler_catches_it(self):
        assert issubclass(FeedUnavailable, MissingRequiredEvidence)
        assert MissingRequiredEvidence("BHF", "spy", "x").kind == "required_missing"

    def test_an_answered_empty_window_is_still_a_gap(self):
        ledger = EvidenceLedger("BHF")
        ledger.missing("insiders", "no Form 4 rows stored for this symbol")
        assert [g.block for g in ledger.gaps] == ["insiders"]


class TestOptions:
    def test_a_throttle_is_unavailable_not_no_chain(self, monkeypatch):
        options_service._CACHE.clear()
        monkeypatch.setattr(options_service, "API",
                            SimpleNamespace(ALPHA_VANTAGE_API_KEY="k"))
        monkeypatch.setattr("services.terminal_cache.get_putcall_summary",
                            lambda *a, **k: None)

        def throttled(*a, **k):
            raise AlphaVantageUnavailable("Note: rate limit")
        monkeypatch.setattr(options_service, "fetch", throttled)
        with pytest.raises(options_service.OptionsUnavailable, match="throttled"):
            options_service.get_put_call_metrics("BHF", AS_OF)
        # Nothing cached: the next call asks again.
        assert options_service._CACHE == {}

    def test_no_key_and_no_warm_chain_is_unavailable(self, monkeypatch):
        options_service._CACHE.clear()
        monkeypatch.setattr(options_service, "API",
                            SimpleNamespace(ALPHA_VANTAGE_API_KEY=""))
        monkeypatch.setattr("services.terminal_cache.get_putcall_summary",
                            lambda *a, **k: None)
        with pytest.raises(options_service.OptionsUnavailable, match="no Alpha Vantage key"):
            options_service.get_put_call_metrics("BHF", AS_OF)

    def test_a_vendor_answer_with_no_contracts_is_none(self, monkeypatch):
        options_service._CACHE.clear()
        monkeypatch.setattr(options_service, "API",
                            SimpleNamespace(ALPHA_VANTAGE_API_KEY="k"))
        monkeypatch.setattr("services.terminal_cache.get_putcall_summary",
                            lambda *a, **k: None)
        monkeypatch.setattr(options_service, "fetch", lambda *a, **k: {"data": []})
        assert options_service.get_put_call_metrics("BHF", AS_OF) is None

    def test_the_report_stops_on_a_throttle_and_writes_around_no_chain(self):
        def throttled(*a, **k):
            raise options_service.OptionsUnavailable("vendor throttled or down")
        with pytest.raises(FeedUnavailable) as exc:
            build({"options"}, **{
                "services.options_service.get_put_call_metrics":
                    {"side_effect": throttled}})
        assert exc.value.block == "options"
        assert "throttled" in exc.value.reason

        blocks, ledger = build({"options"}, **{
            "services.options_service.get_put_call_metrics": {"return_value": None}})
        gap = next(g for g in ledger.gaps if g.block == "options")
        assert "no listed options" in gap.reason
        assert any(b.startswith("[SPY regime") for b in blocks)


class TestQuality:
    def test_the_screen_says_when_its_source_did_not_answer(self, monkeypatch):
        bad_apples_service._CACHE.clear()
        monkeypatch.setattr("services.stock_data.get_ticker",
                            lambda s: (_ for _ in ()).throw(RuntimeError("no yf")))
        monkeypatch.setattr("services.terminal_cache.get_info", lambda *a, **k: None)
        monkeypatch.setattr(bad_apples_service, "scan_news_red_flags",
                            lambda *a, **k: [])
        out = bad_apples_service.analyze_symbol("BHF", AS_OF, articles=[])
        assert "did not answer" in out["unavailable"]
        # Never raises still holds: the result is complete, just flagged.
        assert out["total_checks"] == len(out["checks"]) > 0

    def test_a_thin_filer_is_a_gap_and_a_dead_source_stops_the_report(self):
        thin = {"symbol": "BHF", "as_of": AS_OF, "checks": [
            {"category": "growth", "check": "x", "status": "n/a",
             "value": "", "threshold": "", "note": ""}],
            "scores": {"growth": {"fail": 0, "pass": 0, "n/a": 1}},
            "total_fails": 0, "total_checks": 1, "flag": "clean"}
        blocks, ledger = build({"quality"}, **{
            "services.bad_apples_service.analyze_symbol": {"return_value": thin}})
        assert next(g for g in ledger.gaps if g.block == "quality")

        dead = dict(thin, unavailable="fundamentals source did not answer")
        with pytest.raises(FeedUnavailable) as exc:
            build({"quality"}, **{
                "services.bad_apples_service.analyze_symbol": {"return_value": dead}})
        assert exc.value.block == "quality"


class TestSparseFeeds:
    """Insider, political and congressional windows are empty most weeks.
    Empty stays a gap; a top-up that failed with nothing stored is dead."""

    def test_insiders(self):
        blocks, ledger = build({"insiders"}, **{
            "services.insider_service.insider_block": {"return_value": ("", [])}})
        assert next(g for g in ledger.gaps if g.block == "insiders")
        with pytest.raises(FeedUnavailable) as exc:
            build({"insiders"}, **{
                "services.insider_service.insider_block": {
                    "return_value": ("", ["insider transactions: unavailable: rate limit"])}})
        assert exc.value.block == "insiders" and "rate limit" in exc.value.reason

    def test_politicians(self):
        with pytest.raises(FeedUnavailable) as exc:
            build({"politicians"}, **{
                "services.politician_dossier.politician_block": {
                    "return_value": ("", ["congressional trades: unavailable: 429"])}})
        assert exc.value.block == "politicians"

    def test_political(self):
        _, ledger = build({"political"}, **{
            "services.political_service.political_blocks": {"return_value": ([], [])}})
        assert next(g for g in ledger.gaps if g.block == "political")
        with pytest.raises(FeedUnavailable):
            build({"political"}, **{
                "services.political_service.political_blocks": {
                    "return_value": ([], ["institutional holdings: unavailable"])}})

    def test_a_rendered_block_over_a_stale_source_is_still_present(self):
        stale = "[BHF: insider transactions (SEC Form 4)]\nJensen Huang sold."
        blocks, ledger = build({"insiders"}, **{
            "services.insider_service.insider_block": {
                "return_value": (stale, ["insider top-up: unavailable"])}})
        assert "insiders" in ledger.present
        assert [g for g in ledger.gaps if g.block == "insiders"] == []


class TestCalendarAndPeers:
    def test_a_calendar_that_could_not_be_read_stops_the_report(self):
        with pytest.raises(FeedUnavailable) as exc:
            build(set(), **{"utils.events.get_upcoming_events":
                            {"return_value": {"unavailable": True}}})
        assert exc.value.block == "events"

    def test_mapped_peers_whose_prices_did_not_come_back_stop_the_report(self):
        with pytest.raises(FeedUnavailable) as exc:
            build(set(), **{
                "models.sector_map.get_peers": {"return_value": ["MET", "PRU"]},
                "utils.metrics.compute_peer_relative_strength": {"return_value": None}})
        assert exc.value.block == "peers" and "MET" in exc.value.reason

    def test_no_peer_mapping_is_still_only_logged(self):
        _, ledger = build(set(), **{"models.sector_map.get_peers": {"return_value": []}})
        gap = next(g for g in ledger.gaps if g.block == "peers")
        assert gap.severity == "optional"


class TestActivity:
    def test_predict_emits_the_refusal_with_block_and_reason(self, monkeypatch):
        events = []
        monkeypatch.setattr("services.progress_service.emit",
                            lambda stage, msg, payload=None, **k: events.append(
                                (stage, msg, payload)))
        model = TradingAgentsModel()
        monkeypatch.setattr(model, "is_ready", lambda: True)

        def boom(*a, **k):
            raise FeedUnavailable("BHF", "options", "vendor throttled or down")
        monkeypatch.setattr(model, "_run_analysis", boom)
        out = model.predict("BHF", frame())
        assert out.error and out.decision == "HOLD" and out.confidence == 0.0
        assert out.details["missing_kind"] == "feed_unavailable"
        stage, msg, payload = events[-1]
        assert stage == "error"
        assert "NOT written" in msg and "feed unavailable" in msg
        assert payload == {"symbol": "BHF", "block": "options",
                           "reason": "vendor throttled or down",
                           "kind": "feed_unavailable"}
