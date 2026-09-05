"""The report's shape follows what the run actually found.

Every report used to render the same twelve sections whether or not there was
anything in them, which is why they all read alike and why none of them said
anything a chart could not. What is pinned here is the fix:

* two things standing out for a symbol produce two sections in the prompt and
  one web search each, aimed at the specific question rather than the ticker;
* a symbol with nothing standing out produces NO anomaly sections and a prompt
  that says so and asks for a short report, because padding a quiet name out
  is the failure this replaces;
* what a run SCREENED is read off the evidence it gathered, not off whether
  rows came back. A category that returned nothing was still looked at, and a
  scan that raised is a third state that must not borrow either wording;
* the investigation cache is keyed on what was asked. A changed question is a
  new search and a repeated one is free, or a changed prompt reads back the
  previous run's answer (the trap this phase exists to close);
* the web-free triage that spares plain momentum names a paid search must not
  swallow an anomaly question, which is exactly the case that should search;
* the block fitting cannot trim the anomaly blocks away to make room for the
  SPY regime block;
* a report quoting an anomaly's figures is fully grounded in its own prompt.

Nothing here reaches the network, a vendor or a database: the fixture below
fails the test if it tries.
"""

from unittest.mock import patch

import pytest

from models.single_agent import (
    SINGLE_AGENT_PROMPT, render_output_sections,
)
from models.trading_agents_model import (
    MIN_BLOCK_CHARS, TradingAgentsModel, _fit_blocks,
)
from services import anomaly_service, av_store, investigation_service, options_service
from services.evidence_contract import EvidenceLedger
from services.investigation_service import research_questions
from utils.figure_check import check_figures

AS_OF = "2026-09-02"


@pytest.fixture(autouse=True)
def no_outside_world(monkeypatch):
    """Every door out of the process except the stubbed LLM."""
    def forbidden(*args, **kwargs):
        raise AssertionError("the dynamic report path reached outside the process")

    import requests
    import db.session as dbs
    from services import alpha_vantage

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(alpha_vantage, "fetch", forbidden)
    monkeypatch.setattr(av_store, "fetch", forbidden)
    monkeypatch.setattr(dbs, "get_session", forbidden)


@pytest.fixture(autouse=True)
def _clear_caches():
    investigation_service._CACHE.clear()
    investigation_service._ANSWER_CACHE.clear()
    investigation_service.begin_research_budget(None)
    yield
    investigation_service._CACHE.clear()
    investigation_service._ANSWER_CACHE.clear()
    investigation_service.begin_research_budget(None)


def test_the_outside_world_fixture_is_not_vacuous():
    import requests
    with pytest.raises(AssertionError):
        requests.get("https://example.com")


PHASE_MODULES = ("services/anomaly_service.py",
                 "services/investigation_service.py",
                 "models/single_agent.py",
                 "models/trading_agents_model.py")


def _pre_312_interpreter():
    """An interpreter old enough to hold the f-string grammar production
    runs. Prefers the one running the tests when it already qualifies."""
    import subprocess
    import sys

    candidates = [sys.executable, "/usr/bin/python3", "python3.11", "python3.9"]
    for exe in candidates:
        if not exe:
            continue
        try:
            out = subprocess.run(
                [exe, "-c", "import sys; print(sys.version_info[0] * 100 + "
                            "sys.version_info[1])"],
                capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and (out.stdout.strip() or "999").isdigit() \
                and int(out.stdout.strip()) < 312:
            return exe
    return None


def test_this_phases_modules_parse_under_the_containers_python():
    """The Dockerfile pins python:3.11-slim, and 3.12 relaxed f-string
    grammar (PEP 701): a backslash or a same-quote nesting inside an f-string
    expression compiles on the dev machine and is a SyntaxError in
    production. Both imports of anomaly_service are lazy and swallowed, so
    there the app still boots, every report silently loses its anomaly
    sections, and the only trace is one warning per symbol.
    """
    import subprocess

    exe = _pre_312_interpreter()
    if exe is None:
        pytest.skip("no interpreter older than 3.12 to check the grammar with")
    for path in PHASE_MODULES:
        out = subprocess.run(
            [exe, "-c",
             "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')",
             path],
            capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, f"{path} does not parse on {exe}:\n{out.stderr}"


# --- seeds ---------------------------------------------------------------

def put_tilted_chain():
    """A liquid chain positioned nearly two to one against the tape, through
    the vendor aggregation so the keys are the vendor's."""
    contracts = ([{"type": "put", "volume": 62_000, "open_interest": 41_000}]
                 + [{"type": "call", "volume": 34_000, "open_interest": 30_000}])
    return options_service._aggregate(contracts, AS_OF)


INSIDER_ROWS = [
    {"symbol": "NVDA", "executive": "Jensen Huang", "title": "President and CEO",
     "security_type": "Common Stock", "side": "D",
     "transaction_date": "2026-08-10", "visible_from": "2026-08-12",
     "shares": 100_000.0, "share_price": 180.50, "value_usd": 18_050_000.0},
    {"symbol": "NVDA", "executive": "Colette Kress", "title": "EVP and CFO",
     "security_type": "Common Stock", "side": "D",
     "transaction_date": "2026-08-05", "visible_from": "2026-08-07",
     "shares": 40_000.0, "share_price": 176.25, "value_usd": 7_050_000.0},
    {"symbol": "NVDA", "executive": "Debora Shoquist", "title": "EVP Operations",
     "security_type": "Common Stock", "side": "D",
     "transaction_date": "2026-08-18", "visible_from": "2026-08-20",
     "shares": 12_000.0, "share_price": 179.00, "value_usd": 2_148_000.0},
]


def insider_summary(monkeypatch):
    """The real summariser over the seeded rows; only the reader is stubbed."""
    monkeypatch.setattr(av_store, "insider_transactions_for",
                        lambda *a, **k: list(INSIDER_ROWS))
    from services.insider_service import summarize_insiders
    return summarize_insiders("NVDA", AS_OF)


ANSWER_JSON = ('```json\n{"finding": "Dealers were hedging the 2026-09-18 '
               'expiry into the antitrust ruling.", "citations": [{"source": '
               '"Reuters", "date": "2026-08-30", "url": "https://r.co/x"}], '
               '"unresolved": ""}\n```')


class WebLLM:
    """Records every web search and refuses a web-free call: the anomaly path
    must not be routed through the momentum triage."""

    def __init__(self, text=ANSWER_JSON):
        self.text = text
        self.prompts = []

    def generate_with_web_search(self, prompt, system, **kw):
        self.prompts.append(prompt)
        kw.get("usage_out", {}).update(
            {"input_tokens": 900, "output_tokens": 300, "model": "m"})
        return {"text": self.text, "searches": 2, "model": "m",
                "sources": [{"url": "https://r.co/x", "title": "Reuters"}],
                "stop_reason": "end_turn"}

    def generate(self, *a, **kw):
        raise AssertionError("an anomaly question was sent to the web-free "
                             "triage instead of a search")

    @property
    def searches(self):
        return len(self.prompts)


def scan(monkeypatch, llm, *, web=True, options=None, insiders=None,
         congress=None, evidence=("options", "insiders"), present=()):
    model = TradingAgentsModel()
    ledger = EvidenceLedger("NVDA")
    for block in present:
        ledger.have(block)
    # The ledger is what says a category was looked at, so the fixture has to
    # record the blocks the real builder would have recorded before the scan.
    if options is not None:
        ledger.have("options")
    if insiders is not None:
        ledger.have("insiders")
        monkeypatch.setattr("services.insider_service.summarize_insiders",
                            lambda *a, **k: insiders)
    # congress=[] is the ordinary case and is NOT "no dossier": the block
    # renders, says nobody filed, and hands back zero rows.
    if congress is not None:
        ledger.have("politicians" if "politicians" in evidence else "political")
        monkeypatch.setattr(av_store, "congress_trades_for",
                            lambda *a, **k: list(congress))
    with patch("services.llm_service.get_llm", return_value=llm):
        # Detection and research are two steps now: the caller detects first
        # so the investigation gate can read the result before spending.
        detected = model._detect_anomalies(
            "NVDA", AS_OF, evidence=set(evidence), ledger=ledger, news=None,
            ohlcv_df=None, options=options, by_expiry=None, quality=None)
        return model._scan_and_research(
            "NVDA", AS_OF, detected=detected, ledger=ledger,
            target="2026-09-03", web=web)


def anomalies_from(*args, **kwargs) -> list:
    """``scan`` returns (anomalies, screened, scan_failed); most tests want
    the first."""
    return scan(*args, **kwargs)[0]


# --- two anomalies -------------------------------------------------------

class TestTwoThingsStandOut:
    def test_two_anomalies_get_two_sections_and_one_question_each(
            self, monkeypatch):
        llm = WebLLM()
        found = anomalies_from(monkeypatch, llm, options=put_tilted_chain(),
                               insiders=insider_summary(monkeypatch))

        # Most severe first: the chain is nearly two to one against the tape,
        # the cluster is three officers on a 180-day window.
        assert [a["key"] for a in found] == ["options_skew", "insider_cluster"]
        # One search per anomaly, each carrying that anomaly's own question.
        # Which prompt arrives first is a race (the questions fan out over a
        # thread pool and each takes a web slot on the way in), so what is
        # pinned is that every question was searched exactly once.
        assert llm.searches == 2
        for anomaly in found:
            carrying = [p for p in llm.prompts if anomaly["question"] in p]
            assert len(carrying) == 1, anomaly["key"]
        assert all(a["researched"] for a in found)

        sections = render_output_sections("NVDA", AS_OF, "XLK", found)
        # Numbered INTO the fixed frame, right after Situation, and the frame
        # renumbers around them rather than being replaced.
        assert '"options_skew"' in sections and '"insider_cluster"' in sections
        assert sections.index('"options_skew"') < sections.index('"insider_cluster"')
        assert "7. Positioning & Flows" in sections
        assert "14. Trade Plan" in sections

    def test_the_researched_finding_and_its_citation_reach_the_block(
            self, monkeypatch):
        llm = WebLLM()
        found = anomalies_from(monkeypatch, llm, options=put_tilted_chain())
        block = anomaly_service.format_anomaly_block(
            "NVDA", found[0], found[0]["answer"])

        assert "Dealers were hedging" in block
        assert "Reuters, 2026-08-30, https://r.co/x" in block
        assert "NOT RESEARCHED" not in block
        # Severity orders the questions, it is not a fact about the company.
        assert "severity" not in block.lower()

    def test_a_web_free_run_says_the_anomaly_is_unresearched(self, monkeypatch):
        llm = WebLLM()
        found = anomalies_from(monkeypatch, llm, web=False,
                               options=put_tilted_chain())

        assert llm.searches == 0
        assert found and not found[0]["researched"]
        block = anomaly_service.format_anomaly_block("NVDA", found[0])
        assert "NOT RESEARCHED" in block
        assert "do not supply one of your own" in block
        # The figures are still there: an unexplained anomaly is evidence.
        assert "puts traded per call" in block

    def test_a_failed_search_states_the_failure_rather_than_vanishing(
            self, monkeypatch):
        class Broken(WebLLM):
            def generate_with_web_search(self, prompt, system, **kw):
                raise RuntimeError("provider 429")

        found = anomalies_from(monkeypatch, Broken(), options=put_tilted_chain())
        assert found and not found[0]["researched"]
        block = anomaly_service.format_anomaly_block(
            "NVDA", found[0], found[0]["answer"])
        assert "NOT RESEARCHED: the search failed" in block
        assert "provider 429" in block


class TestWhyAQuestionWentUnresearched:
    """The reason is stamped by the stage that knows it, never inferred from
    a missing answer. A block that guessed "web research was off" on a run
    where it was on had the report state something false about its own
    provenance, contradicted by the same report's footer.
    """

    def test_a_capped_question_does_not_blame_the_web_switch(self, monkeypatch):
        # Two anomalies, a budget of one: the second is dropped by the cap on
        # a run whose web research is ON.
        investigation_service.begin_research_budget(1)
        llm = WebLLM()
        found = anomalies_from(monkeypatch, llm, options=put_tilted_chain(),
                               insiders=insider_summary(monkeypatch))

        assert llm.searches == 1
        assert [a["researched"] for a in found] == [True, False]
        block = anomaly_service.format_anomaly_block("NVDA", found[1],
                                                     found[1]["answer"])
        assert "NOT RESEARCHED" in block
        assert "research budget went to higher-ranked questions" in block
        assert "web research was off" not in block

    def test_the_per_symbol_cap_reads_the_same_way(self, monkeypatch):
        # MAX_ANOMALIES is 4 and MAX_RESEARCH_QUESTIONS is 3, so a symbol at
        # the anomaly cap always has one question it never asks.
        monkeypatch.setattr(investigation_service, "MAX_RESEARCH_QUESTIONS", 1)
        llm = WebLLM()
        found = anomalies_from(monkeypatch, llm, options=put_tilted_chain(),
                               insiders=insider_summary(monkeypatch))

        assert llm.searches == 1
        block = anomaly_service.format_anomaly_block("NVDA", found[1],
                                                     found[1]["answer"])
        assert "research budget went to higher-ranked questions" in block

    def test_a_web_free_run_still_says_the_web_was_off(self, monkeypatch):
        found = anomalies_from(monkeypatch, WebLLM(), web=False,
                               options=put_tilted_chain())
        block = anomaly_service.format_anomaly_block("NVDA", found[0])
        assert "web research was off for this run" in block

    def test_a_crashed_research_stage_says_so(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("pool exhausted")

        monkeypatch.setattr(investigation_service, "research_questions", boom)
        found = anomalies_from(monkeypatch, WebLLM(),
                               options=put_tilted_chain())
        block = anomaly_service.format_anomaly_block("NVDA", found[0])
        assert "the research stage failed for this symbol" in block

    def test_an_unstamped_anomaly_claims_nothing_about_the_run(self):
        """The default wording is the only one that is true whatever
        happened: a caller that forgot to stamp a reason must not have the
        block invent one."""
        anomaly = {"key": "options_skew", "title": "t", "facts": ["1 fact"],
                   "question": "why?", "evidence_key": "options"}
        block = anomaly_service.format_anomaly_block("NVDA", anomaly)
        assert "this question was not researched on this run" in block
        assert "web research was off" not in block


class TestRunLevelResearchBudget:
    """Anomaly questions are asked inside the serial per-symbol model loop,
    so the per-symbol cap alone does not bound a 20-symbol scheduled job."""

    def test_the_budget_is_spent_across_symbols_not_per_symbol(self, monkeypatch):
        investigation_service.begin_research_budget(2)
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            first = research_questions("AAA", AS_OF, ["q1", "q2"], web=True)
            second = research_questions("BBB", AS_OF, ["q3", "q4"], web=True)

        assert len(first) == 2 and second == []
        assert llm.searches == 2
        assert investigation_service.research_budget_left() == 0

    def test_a_cached_answer_does_not_spend_a_slot(self, monkeypatch):
        """The ceiling counts SEARCHES. Claiming before the cache lookup let
        an answer that cost nothing burn a slot, and the question refused
        afterwards was labelled "capped" and rendered as "this run's research
        budget went to higher-ranked questions", which had not happened."""
        investigation_service.begin_research_budget(2)
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            research_questions("AAA", AS_OF, ["q1"], web=True)
            assert investigation_service.research_budget_left() == 1
            # Same symbol, same date, same question: served from the cache.
            again = research_questions("AAA", AS_OF, ["q1"], web=True)
            assert len(again) == 1 and llm.searches == 1
            assert investigation_service.research_budget_left() == 1
            # And the slot the cache hit did not take is still there to spend.
            fresh = research_questions("BBB", AS_OF, ["q2"], web=True)

        assert len(fresh) == 1 and llm.searches == 2
        assert investigation_service.research_budget_left() == 0

    def test_a_cache_hit_beside_a_new_question_leaves_the_ceiling_for_the_new_one(
            self, monkeypatch):
        """One warm question and one cold one, with a single slot left: the
        cold one must get it rather than be refused by the warm one."""
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            investigation_service.begin_research_budget(None)
            research_questions("AAA", AS_OF, ["warm"], web=True)
            investigation_service.begin_research_budget(1)
            answers = research_questions("AAA", AS_OF, ["warm", "cold"],
                                         web=True)

        assert [a["question"] for a in answers] == ["warm", "cold"]
        assert llm.searches == 2
        assert investigation_service.research_budget_left() == 0

    def test_no_budget_set_means_no_ceiling(self, monkeypatch):
        investigation_service.begin_research_budget(None)
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            answers = research_questions("AAA", AS_OF, ["q1", "q2", "q3"],
                                         web=True)
        assert len(answers) == 3
        assert investigation_service.research_budget_left() is None

    def test_the_questions_for_one_symbol_run_concurrently(self, monkeypatch):
        """Three searches one after another would add their full latency to
        every symbol of a scheduled run."""
        import threading

        started = threading.Barrier(3, timeout=5)

        class Concurrent(WebLLM):
            def generate_with_web_search(self, prompt, system, **kw):
                # Deadlocks (BrokenBarrier) unless all three are in flight.
                started.wait()
                return super().generate_with_web_search(prompt, system, **kw)

        with patch("services.llm_service.get_llm", return_value=Concurrent()):
            answers = research_questions("AAA", AS_OF, ["q1", "q2", "q3"],
                                         web=True)
        assert [a["finding"][:7] for a in answers] == ["Dealers"] * 3

    def test_usage_attribution_survives_the_fan_out(self, monkeypatch):
        """Workers run in a copy of the caller's context: a plain thread
        would bill these searches to no stage at all."""
        from services import usage_service

        seen = []

        class Tagged(WebLLM):
            def generate_with_web_search(self, prompt, system, **kw):
                seen.append(usage_service.current().section)
                return super().generate_with_web_search(prompt, system, **kw)

        with patch("services.llm_service.get_llm", return_value=Tagged()):
            with usage_service.track("investigation", symbol="AAA",
                                     section="anomaly_research:AAA"):
                research_questions("AAA", AS_OF, ["q1", "q2"], web=True)
        assert seen == ["anomaly_research:AAA"] * 2

    def test_a_symbol_dispatched_through_a_copied_context_spends_the_ceiling(self):
        """prediction_service hands each symbol to a worker with
        ``copy_context().run`` and the report callback uses
        ``asyncio.to_thread``: both copy the mapping, so the run's ledger has
        to be the same OBJECT in the worker or every symbol would start with
        a fresh ceiling."""
        from contextvars import copy_context

        investigation_service.begin_research_budget(1)
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            first = copy_context().run(research_questions, "AAA", AS_OF,
                                       ["q1"], web=True)
            second = copy_context().run(research_questions, "BBB", AS_OF,
                                        ["q2"], web=True)
        assert len(first) == 1 and second == []
        assert llm.searches == 1

    def test_two_overlapping_runs_do_not_spend_each_others_ceiling(self):
        """The scheduler and the report callback share one process. A single
        module-wide counter meant the 07:00 watchlist job drained it and
        every interactive report afterwards was told its questions had been
        ranked out by a run it had nothing to do with."""
        from contextvars import copy_context

        scheduled, interactive = copy_context(), copy_context()
        scheduled.run(investigation_service.begin_research_budget, 2)
        interactive.run(investigation_service.begin_research_budget, 2)
        llm = WebLLM()

        def ask(ctx, symbol, question):
            with patch("services.llm_service.get_llm", return_value=llm):
                return ctx.run(research_questions, symbol, AS_OF, [question],
                               web=True)

        # Interleaved, the way an 8am job and a user clicking Run overlap.
        assert len(ask(scheduled, "AAA", "q1")) == 1
        assert len(ask(interactive, "BBB", "q2")) == 1
        assert len(ask(scheduled, "AAA", "q3")) == 1
        assert len(ask(interactive, "BBB", "q4")) == 1
        # Each run spent its own two and neither borrowed from the other.
        assert ask(scheduled, "AAA", "q5") == []
        assert ask(interactive, "BBB", "q6") == []
        assert llm.searches == 4
        assert scheduled.run(investigation_service.research_budget_left) == 0
        assert interactive.run(investigation_service.research_budget_left) == 0


class TestTheSearchIsAimedAtTheFigures:
    """An anomaly is a measurement. Sending the researcher the one-line
    question without the numbers that raised it, and telling it in writing
    that no figures were supplied, turns a question about a 1.8-to-1 chain
    into a general search on the ticker."""

    def test_the_facts_that_raised_the_question_reach_the_researcher(
            self, monkeypatch):
        llm = WebLLM()
        found = anomalies_from(monkeypatch, llm, options=put_tilted_chain())

        prompt = llm.prompts[0]
        assert found[0]["facts"]
        for fact in found[0]["facts"]:
            assert fact in prompt
        assert "no supporting figures supplied" not in prompt

    def test_the_same_question_on_different_figures_is_a_new_search(self):
        """The question names the count, the facts name what is behind it, so
        the two move independently and only the pair identifies an answer."""
        llm = WebLLM()
        question = "Why is the NVDA chain positioned against the tape?"

        def ask(context):
            with patch("services.llm_service.get_llm", return_value=llm):
                return research_questions(
                    "NVDA", AS_OF, [question], web=True,
                    context_by_question={question: context})

        ask("1.82 puts traded per call")
        ask("1.82 puts traded per call")
        assert llm.searches == 1, "the same question and figures re-searched"
        ask("0.41 puts traded per call")
        assert llm.searches == 2


# --- nothing stands out --------------------------------------------------

class TestNothingStandsOut:
    def test_a_quiet_symbol_gets_no_anomaly_sections_and_a_short_report(
            self, monkeypatch):
        llm = WebLLM()
        # A balanced, liquid chain and no other evidence: the correct outcome
        # is silence, not a low-severity section to fill a slot.
        balanced = options_service._aggregate(
            [{"type": "put", "volume": 40_000, "open_interest": 0},
             {"type": "call", "volume": 52_000, "open_interest": 0}], AS_OF)
        found, screened, failed = scan(monkeypatch, llm, options=balanced)

        assert found == []
        assert llm.searches == 0

        sections = render_output_sections("NVDA", AS_OF, "XLK", found, screened)
        # The template wraps, so read it the way the model does.
        flat = " ".join(sections.split())
        assert "ANOMALY block headed" not in sections
        assert "Nothing stands out for NVDA in what this run screened" in flat
        assert "THIS REPORT IS SHORT" in sections
        assert ("Padding a quiet symbol out with technical recitation is a "
                "failure of this report") in flat
        # It names the one category it looked at and every one it did not,
        # instead of asserting an absence over blocks never fetched.
        assert "screened: the options chain." in flat
        assert "did NOT screen insider filings" in flat
        assert "congressional filings" in flat
        # The fixed frame is intact and numbered as it always was.
        assert "1. Situation & Key Figures" in sections
        assert "5. Positioning & Flows" in sections
        assert "12. Trade Plan" in sections

    def test_a_run_that_screened_nothing_does_not_claim_a_quiet_symbol(
            self, monkeypatch):
        """detect() returns [] both for a quiet symbol and for a symbol whose
        blocks were never fetched. Only the first is "nothing stands out"."""
        found, screened, failed = scan(monkeypatch, WebLLM(), options=None,
                                       evidence=())
        assert found == [] and screened == [] and failed is False

        sections = render_output_sections("NVDA", AS_OF, "XLK", found, screened)
        flat = " ".join(sections.split())
        assert "gathered none of the evidence the anomaly scan reads" in flat
        assert "you must not write that nothing does" in flat
        assert "no insider cluster" not in flat
        assert "Nothing stands out" not in flat

    def test_a_caller_with_no_screen_at_all_gets_the_same_treatment(self):
        """The default is the honest one: a caller that passes no screen has
        not told us a single category was ruled out."""
        flat = " ".join(render_output_sections("NVDA", AS_OF, "XLK", []).split())
        assert "gathered none of the evidence the anomaly scan reads" in flat

    def test_a_screened_category_that_returned_nothing_is_still_screened(
            self, monkeypatch):
        """The headline case. Congressional disclosure is sparse, so the
        common outcome is a dossier that rendered, said nobody filed, and
        returned zero rows. Reading gathered-ness off those rows told the
        model the filings "were not looked at" on the same prompt as the
        block saying they were checked, and no report could ever say that no
        member of Congress traded the name."""
        found, screened, failed = scan(
            monkeypatch, WebLLM(), congress=[],
            evidence=("politicians",))

        assert found == [] and failed is False
        assert "congressional filings" in screened

        flat = " ".join(render_output_sections(
            "NVDA", AS_OF, "XLK", found, screened).split())
        assert "screened: congressional filings" in flat
        assert "did NOT screen congressional filings" not in flat
        # And the categories this run really did skip are still named.
        assert "the options chain" in flat

    def test_the_political_blocks_congress_half_counts_as_a_screen(
            self, monkeypatch):
        """With the dossier off, political_blocks renders the congressional
        half itself, so those rows ARE in the prompt. Tying the screen to the
        dossier key alone put "this run did NOT screen congressional filings"
        beside a block listing them."""
        found, screened, failed = scan(monkeypatch, WebLLM(), congress=[],
                                       evidence=("political",))
        assert "congressional filings" in screened and failed is False

    def test_the_dossier_key_wins_when_both_are_selected(self, monkeypatch):
        """With both keys on, political_blocks is called with
        include_congress=False, so a political block that rendered carries
        13F holders only, and a dossier that failed leaves the congressional
        filings genuinely unscreened."""
        found, screened, failed = scan(monkeypatch, WebLLM(), congress=None,
                                       evidence=("political", "politicians"),
                                       present=("political",))
        assert "congressional filings" not in screened

    def test_a_category_the_run_never_fetched_is_not_screened(
            self, monkeypatch):
        """The other side of the same contract: politicians was not in the
        run's evidence, so nothing about congressional filings is known."""
        found, screened, failed = scan(monkeypatch, WebLLM(), congress=None,
                                       evidence=("politicians",))

        assert screened == [] and failed is False
        flat = " ".join(render_output_sections(
            "NVDA", AS_OF, "XLK", found, screened).split())
        assert "gathered none of the evidence the anomaly scan reads" in flat

    def test_rows_that_came_back_are_screened_and_researched(self, monkeypatch):
        """Gathered and found: the same category, with rows in it, still
        reads as screened and raises its question."""
        llm = WebLLM()
        trade = {"symbol": "NVDA", "bioguide_id": "P000001",
                 "politician": "Rep. A", "party": "D", "chamber": "House",
                 "state": "CA-11", "type": "PURCHASE", "owner": "self",
                 "asset_name": "NVIDIA Corp", "transaction_date": "2026-08-20",
                 "filed_date": "2026-08-30", "amount_min": 15001.0,
                 "amount_max": 50000.0}
        found, screened, failed = scan(monkeypatch, llm, congress=[trade],
                                       evidence=("politicians",))

        assert [a["key"] for a in found] == ["congress_activity"]
        assert "congressional filings" in screened and failed is False
        assert llm.searches == 1

    def test_a_scan_that_raised_is_neither_quiet_nor_unscreened(
            self, monkeypatch):
        """The third state. A crash AFTER the evidence was gathered knows
        neither that the symbol is quiet nor that there was nothing to
        screen, and reusing either wording puts a false account of the run
        in the report."""
        def boom(*a, **k):
            raise RuntimeError("detector blew up")

        monkeypatch.setattr(anomaly_service, "detect", boom)
        found, screened, failed = scan(monkeypatch, WebLLM(), congress=[],
                                       evidence=("politicians",))

        assert found == [] and failed is True
        # What the run HAD gathered is still known and still named.
        assert "congressional filings" in screened

        flat = " ".join(render_output_sections(
            "NVDA", AS_OF, "XLK", found, screened, failed).split())
        assert "The anomaly scan for NVDA failed on this run" in flat
        assert "congressional filings" in flat
        assert "Nothing stands out" not in flat
        assert "gathered none of the evidence" not in flat

    def test_the_footer_names_what_was_screened(self, monkeypatch):
        from models.single_agent import _join
        assert _join(["a"]) == "a"
        assert _join(["a", "b"]) == "a and b"
        assert _join(["a", "b", "c"]) == "a, b and c"

    def test_the_prompt_says_movement_is_not_a_forecast(self):
        assert ("PRICE AND INDICATOR MOVEMENT DESCRIBE WHAT ALREADY HAPPENED"
                in SINGLE_AGENT_PROMPT)
        assert ("the evidence does not support a directional call"
                in SINGLE_AGENT_PROMPT)


# --- the cache key -------------------------------------------------------

class TestCacheKey:
    def test_a_different_question_is_a_different_search(self, monkeypatch):
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            first = research_questions("NVDA", AS_OF, ["Why is the chain skewed?"],
                                       web=True)
            again = research_questions("NVDA", AS_OF, ["Why is the chain skewed?"],
                                       web=True)
            other = research_questions(
                "NVDA", AS_OF, ["Why are three officers selling?"], web=True)

        assert llm.searches == 2, "the repeated question must be free"
        assert first[0]["finding"] == again[0]["finding"]
        assert other[0]["question"] == "Why are three officers selling?"

    def test_changed_evidence_is_not_served_the_previous_answer(self):
        """The trap this phase closes: the key used to be (symbol, date, web,
        model), so feeding the investigator new headlines read back the answer
        it gave before them."""
        from services.investigation_service import _evidence_digest, investigate

        assert (_evidence_digest("p", "h", "f", "q")
                != _evidence_digest("p", "h2", "f", "q"))

        calls = []

        class Classifier:
            def generate(self, prompt, system, **kw):
                calls.append(prompt)
                kw.get("usage_out", {}).update({"input_tokens": 1,
                                                "output_tokens": 1, "model": "m"})
                return ('```json\n{"situation": "MOMENTUM_ONLY", '
                        '"one_line": "flow and trend"}\n```')

        with patch("services.llm_service.get_llm", return_value=Classifier()):
            investigate("NVDA", AS_OF, headlines="- 2026-09-01 [AP] quiet")
            investigate("NVDA", AS_OF, headlines="- 2026-09-01 [AP] quiet")
            investigate("NVDA", AS_OF, headlines="- 2026-09-02 [AP] CEO resigns")

        assert len(calls) == 2
        assert "CEO resigns" in calls[1]

    def test_the_momentum_triage_never_swallows_an_anomaly_question(self):
        """``investigate`` spares a plain momentum name a paid search. A
        question raised by three officers selling is exactly the case that
        should search, so it does not go through that gate: WebLLM.generate
        (the web-free triage call) fails the test if it is reached."""
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            answers = research_questions(
                "NVDA", AS_OF, ["Why are three officers selling NVDA?"],
                web=True)
        assert llm.searches == 1
        assert answers[0]["citations"][0]["url"] == "https://r.co/x"

    def test_the_question_cap_bounds_the_spend(self):
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            answers = research_questions(
                "NVDA", AS_OF,
                [f"Question {i}?" for i in range(8)], web=True)
        assert len(answers) == investigation_service.MAX_RESEARCH_QUESTIONS
        assert llm.searches == investigation_service.MAX_RESEARCH_QUESTIONS

    def test_a_web_free_run_researches_nothing(self):
        llm = WebLLM()
        with patch("services.llm_service.get_llm", return_value=llm):
            assert research_questions("NVDA", AS_OF, ["Why?"], web=False) == []
        assert llm.searches == 0


# --- fitting -------------------------------------------------------------

def test_the_anomaly_blocks_survive_a_prompt_that_is_over_budget():
    anomaly = "[NVDA ANOMALY \"options_skew\"]\n" + "fact line\n" * 300
    filler = ["[filler block]\n" + "noise line\n" * 300 for _ in range(8)]
    fitted = _fit_blocks([anomaly] + filler, protected=1)

    assert 'ANOMALY "options_skew"' in fitted
    # The protected block keeps more than the floor every other block is cut
    # to; without protection the water-fill treats it like the rest.
    kept = fitted.split("\n\n")[0]
    assert len(kept) > MIN_BLOCK_CHARS
    assert len(kept) > len(_fit_blocks([anomaly] + filler).split("\n\n")[0])


def test_ordinary_fitting_is_unchanged_when_nothing_is_protected():
    blocks = ["[a]\n" + "x" * 5000, "[b]\n" + "y" * 5000, "[c]\n" + "z" * 5000]
    assert _fit_blocks(blocks, protected=0) == _fit_blocks(blocks)


# --- figure grounding ----------------------------------------------------

# A section of the shape the prompt now demands: the block's figures quoted
# exactly, the researched finding with its source, and the reading kept apart
# from both.
ANOMALY_REPORT = """### 2. Options chain is put-tilted: dealers hedging a dated event
The chain shows 1.82 puts traded per call against 96,000 contracts, with put
volume 62,000 against call volume 34,000 as of 2026-09-02. Open interest sits
at 1.37 puts per call, so today's flow matches the standing position.
Research: dealers were hedging the 2026-09-18 expiry into the antitrust ruling
(Reuters, 2026-08-30).
Reading, not observation: a hedge that dated unwinds after the ruling, so the
skew is not itself a directional signal for tomorrow.
**Read:** the positioning is event-shaped, so treat the ratio as a calendar
marker rather than a bearish vote.

### 3. Three insiders disposed in the same window: $27.2M, no buyers
Jensen Huang (President and CEO), Colette Kress (EVP and CFO) and Debora
Shoquist (EVP Operations) filed 3 disposals and nothing the other way,
$27.2M across the rows carrying a real share price. A reader expanding that
reads $27,200,000.
Reading, not observation: three officers on one side is the pattern worth
weighting, and the cause was not researched in this run.
**Read:** treat it as a size argument, not a timing one.
"""


def build_prompt(extra_context: str, anomalies) -> str:
    """The real template, filled the way ``analyze`` fills it: only the
    anomaly block carries content, so any figure the auditor flags below came
    from that block or from nowhere."""
    return SINGLE_AGENT_PROMPT.format(
        ticker="NVDA", date=AS_OF, sector_etf="XLK",
        situation_line="momentum only.", track_record_block="no rate yet",
        business_block="chips", spy_block="spy", sector_block="sector",
        price_block="price", tech_block="tech", fundamentals_block="fund",
        news_block="news",
        output_sections=render_output_sections("NVDA", AS_OF, "XLK", anomalies),
        extra_context=(
            "\n== PRECOMPUTED METRICS & EVENTS (validated. Prefer these "
            "numbers) ==\n" + extra_context + "\n"))


def anomaly_prompt(monkeypatch):
    """The prompt a two-anomaly run really builds: both blocks in the
    precomputed section, both sections in the output format."""
    llm = WebLLM()
    found = anomalies_from(monkeypatch, llm, options=put_tilted_chain(),
                           insiders=insider_summary(monkeypatch))
    blocks = "\n\n".join(
        anomaly_service.format_anomaly_block("NVDA", a, a["answer"])
        for a in found)
    return build_prompt(blocks, found)


def test_a_report_quoting_an_anomalys_numbers_is_grounded(monkeypatch):
    result = check_figures(ANOMALY_REPORT, anomaly_prompt(monkeypatch),
                           ignore_values=(0.6,))
    assert result.unmatched == []
    # The ratios, the open-interest reading and the window dollars, in both
    # the compressed and the expanded form, are audited rather than skipped.
    assert result.checked >= 4


def test_an_invented_figure_in_an_anomaly_section_is_still_caught(monkeypatch):
    prompt = anomaly_prompt(monkeypatch)
    invented = ANOMALY_REPORT.replace("1.82 puts", "3.41 puts")
    assert check_figures(invented, prompt).unmatched == ["3.41"]
    invented = ANOMALY_REPORT.replace("$27.2M", "$41.8M")
    assert check_figures(invented, prompt).unmatched == ["$41.8M"]


def test_the_prompt_that_carries_no_frame_still_formats():
    """Every placeholder the template declares is supplied by the renderer:
    a missing one is a KeyError at report time, on a live run."""
    prompt = build_prompt("", [])
    assert "{output_sections}" not in prompt
    assert "1. Situation & Key Figures" in prompt


def test_the_report_frame_is_intact(monkeypatch):
    """The dynamic middle is an addition. The Verdict block, the positioning
    section and the Trade Plan are the contract the rest of the app parses."""
    llm = WebLLM()
    found = anomalies_from(monkeypatch, llm, options=put_tilted_chain(),
                           insiders=insider_summary(monkeypatch))
    prompt = build_prompt("", found)

    assert "FINAL TRANSACTION PROPOSAL" in prompt
    assert "MEASURED ACCURACY" in prompt
    for name in ("Positioning & Flows", "Bull vs Bear", "Scenarios",
                 "Trade Plan", "Situation & Key Figures"):
        assert name in prompt
    from services.report_parse import _HEADING_RE
    numbered = [line for line in
                render_output_sections("NVDA", AS_OF, "XLK", found).split("\n")
                if line[:1].isdigit()]
    # Every section is numbered continuously, so the reader's parser
    # (_HEADING_RE wants "### <n>. ") still matches what the model writes.
    assert [int(line.split(".", 1)[0]) for line in numbered] == list(
        range(1, len(numbered) + 1))
    assert _HEADING_RE.match("### 2. Options chain is put-tilted: hedging")
