"""The 2026-09-06 spend levers: ask the web only what it can answer, reuse
a sourced answer across days for the same figures, and record per section
whether the search produced a lead."""
from datetime import date

import pytest

from services import anomaly_service, investigation_service as inv


class TestOnlyResearchableKindsAreAsked:
    def test_the_split(self):
        assert anomaly_service.researchable({"key": "news_spike"})
        assert anomaly_service.researchable({"key": "insider_cluster"})
        assert not anomaly_service.researchable({"key": "options_skew"})
        assert not anomaly_service.researchable({"key": "quality_failures"})
        assert not anomaly_service.researchable({"key": "positioning_vs_price"})

    def test_a_flag_only_kind_says_why_in_its_block(self):
        a = {"key": "quality_failures", "title": "6 checks", "facts": ["f"],
             "question": "q", "evidence_key": "quality",
             "unresearched": "not_researchable"}
        block = anomaly_service.format_anomaly_block("ORCL", a, None)
        assert "the open web cannot answer this" in block
        assert "no search was bought" in block

    def test_the_model_asks_only_the_askable(self, monkeypatch):
        from models.trading_agents_model import TradingAgentsModel
        from services.evidence_contract import EvidenceLedger
        asked = {}

        def fake_rq(symbol, as_of, questions, **kw):
            asked["questions"] = list(questions)
            return [{"question": q, "finding": "found", "citations": [
                {"source": "Reuters", "date": as_of}], "searches": 2}
                for q in questions]

        monkeypatch.setattr(inv, "research_questions", fake_rq)
        model = TradingAgentsModel()
        found = [
            {"key": "news_spike", "title": "spike", "severity": 0.6,
             "facts": ["13 articles"], "question": "why the spike?",
             "evidence_key": "news_source"},
            {"key": "options_skew", "title": "tilt", "severity": 0.5,
             "facts": ["0.38 p/c"], "question": "why the tilt?",
             "evidence_key": "options"},
        ]
        anomalies, _, failed = model._scan_and_research(
            "ORCL", "2026-09-04", detected=(found, ["news volume"], False),
            ledger=EvidenceLedger("ORCL"), target="2026-09-08", web=True)
        assert failed is False
        assert asked["questions"] == ["why the spike?"]
        by = {a["key"]: a for a in anomalies}
        assert by["news_spike"]["researched"] and by["news_spike"]["sourced"]
        assert by["options_skew"]["unresearched"] == "not_researchable"
        assert by["options_skew"]["answer"] is None


class TestAnswersAreReusedAcrossDays:
    @pytest.fixture
    def reuse_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(inv, "_REUSE_DIR", str(tmp_path / "answers"))
        monkeypatch.setattr(inv, "_reuse_disk", None)
        inv._ANSWER_CACHE.clear()
        token = inv._BUDGET.set(None)
        yield tmp_path
        inv._BUDGET.reset(token)
        inv._ANSWER_CACHE.clear()

    def _run(self, monkeypatch, as_of, calls):
        def fake_one(symbol, as_of_, question, **kw):
            calls.append(as_of_)
            return inv._answer(question, finding="EU review opened",
                               citations=[{"source": "Reuters", "date": as_of_}],
                               searches=3)
        monkeypatch.setattr(inv, "_research_one", fake_one)
        monkeypatch.setattr(inv, "_web_slot", lambda: __import__("contextlib").nullcontext())
        return inv.research_questions(
            "ORCL", as_of, ["why the spike?"], web=True, model="gpt-5.6-luna",
            context_by_question={"why the spike?": "same figures"})

    def test_a_later_day_reads_monday_back_and_pays_nothing(self, monkeypatch, reuse_dir):
        calls = []
        first = self._run(monkeypatch, "2026-09-01", calls)
        assert first[0]["finding"] == "EU review opened" and calls == ["2026-09-01"]
        inv._ANSWER_CACHE.clear()
        again = self._run(monkeypatch, "2026-09-03", calls)
        assert calls == ["2026-09-01"]          # no second search
        assert again[0]["reused_from"] == "2026-09-01"
        assert again[0]["searches"] == 0
        block = anomaly_service.format_anomaly_block(
            "ORCL", {"key": "news_spike", "title": "t", "facts": [],
                     "question": "why the spike?", "evidence_key": "n"},
            again[0])
        assert "Researched on 2026-09-01" in block

    def test_an_earlier_as_of_never_sees_its_future(self, monkeypatch, reuse_dir):
        calls = []
        self._run(monkeypatch, "2026-09-03", calls)
        inv._ANSWER_CACHE.clear()
        back = self._run(monkeypatch, "2026-09-01", calls)
        assert calls == ["2026-09-03", "2026-09-01"]
        assert "reused_from" not in back[0]

    def test_reuse_off_buys_every_time(self, monkeypatch, reuse_dir):
        from config import MODEL
        saved = MODEL.ANOMALY_ANSWER_REUSE_DAYS
        object.__setattr__(MODEL, "ANOMALY_ANSWER_REUSE_DAYS", 0)
        try:
            calls = []
            self._run(monkeypatch, "2026-09-01", calls)
            inv._ANSWER_CACHE.clear()
            self._run(monkeypatch, "2026-09-03", calls)
            assert calls == ["2026-09-01", "2026-09-03"]
        finally:
            object.__setattr__(MODEL, "ANOMALY_ANSWER_REUSE_DAYS", saved)
