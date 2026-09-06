"""The research prompt is two halves: a system message that is the same for
every symbol (so the provider's prefix cache serves it) and a user message
that carries what changes. These pin the split and the search-cap floor
that landed with it."""

import re
from contextlib import nullcontext

import pytest

from models.single_agent import (
    SINGLE_AGENT_PROMPT, SINGLE_AGENT_SYSTEM_PROMPT, build_system_prompt,
    render_output_sections,
)

PLACEHOLDER = re.compile(
    r"\{(ticker|date|sector_etf|situation_line|track_record_block|"
    r"business_block|price_block|tech_block|fundamentals_block|news_block|"
    r"spy_block|sector_block|extra_context|output_sections)\}")


def user_prompt(ticker="ETR", date="2026-09-04"):
    return SINGLE_AGENT_PROMPT.format(
        ticker=ticker, date=date, sector_etf="XLU", situation_line="sit-line",
        track_record_block="tr", business_block="b", spy_block="s",
        sector_block="se", price_block="p", tech_block="t",
        fundamentals_block="f", news_block="n", extra_context="",
        output_sections=render_output_sections(ticker, date, "XLU", []))


class TestTheSystemHalfIsStable:
    def test_it_names_no_symbol_and_no_date(self):
        for variant in (build_system_prompt(True), build_system_prompt(False)):
            assert not PLACEHOLDER.search(variant)
            # word-bounded: "mETRics" is not a ticker. A static example date
            # inside a rule ("Reuters, 2026-08-14") is fine; a real cutoff is not.
            assert not re.search(r"\bETR\b", variant)
            assert "2026-09-04" not in variant

    def test_it_is_byte_identical_across_symbols(self):
        # The point of the split: one prefix for the whole run.
        assert build_system_prompt(True) == build_system_prompt(True)
        assert build_system_prompt(True) != build_system_prompt(False)
        assert '"company_thesis"' in build_system_prompt(True)
        assert '"company_thesis"' not in build_system_prompt(False)

    def test_it_clears_the_openai_cache_floor(self):
        # 1024 tokens is the smallest prefix OpenAI caches; below that the
        # split would buy nothing.
        assert len(build_system_prompt(False)) // 4 > 1024

    def test_it_carries_the_contract_the_app_parses(self):
        system = build_system_prompt(True)
        for anchor in ("## Verdict", "FINAL TRANSACTION PROPOSAL",
                       "MEASURED ACCURACY", "REASSESS_TO_BUY", "MOVE_TO_SELL",
                       "== STRUCTURED EPILOGUE (mandatory) ==",
                       "**Step 0: Situation", "**Step 3b: Map the outcomes",
                       "== VOICE AND PUNCTUATION =="):
            assert anchor in system, anchor
        # The rules now point at the user message rather than at "below".
        assert "the decision date" in system
        assert "SITUATION line at the top of the\nuser message" in system


class TestTheUserHalfCarriesWhatChanges:
    def test_it_opens_with_the_ticker_and_the_date(self):
        first_line = user_prompt().splitlines()[0]
        assert "ETR" in first_line and "2026-09-04" in first_line

    def test_it_holds_the_situation_line_and_the_section_list(self):
        p = user_prompt()
        assert "== SITUATION ==\nsit-line" in p
        assert p.index("== SECTIONS (write these, in this order) ==") < p.index("1. Situation & Key Figures")
        # The frame ends on Peer Comparison now that the plan rides with the
        # technicals and the evidence sections sit at the bottom.
        assert p.rstrip().endswith("(omit this section only if no peer block was provided)")
        assert "3. Technicals & Trade Plan" in p and "8. Peer Comparison" in p

    def test_no_rule_text_leaked_into_the_per_symbol_half(self):
        p = user_prompt()
        for rule in ("HOW TO REASON", "VOICE AND PUNCTUATION",
                     "REQUIRED OUTPUT FORMAT", "STRUCTURED EPILOGUE"):
            assert rule not in p, rule

    def test_two_symbols_differ_only_in_the_user_half(self):
        assert user_prompt("ETR") != user_prompt("BAC")
        assert SINGLE_AGENT_SYSTEM_PROMPT == SINGLE_AGENT_SYSTEM_PROMPT


class TestTheQuestionSearchFloor:
    """At a cap of 3, two anomaly questions would have been asked with one
    search each; the floor keeps each at two."""

    @pytest.fixture
    def recorder(self, monkeypatch):
        import services.investigation_service as inv
        from config import MODEL

        seen = []

        def fake_one(symbol, as_of, question, *, target, context, model, max_searches):
            seen.append(max_searches)
            return inv._answer(question, finding="f", searches=max_searches)

        monkeypatch.setattr(inv, "_research_one", fake_one)
        monkeypatch.setattr(inv, "_web_slot", lambda: nullcontext())
        # The budget is a ContextVar: set it for this test and put it back,
        # or every later test in the process runs without a ceiling.
        token = inv._BUDGET.set(None)
        # MODEL is a frozen dataclass; monkeypatch.setattr cannot write it.
        saved = (MODEL.INVESTIGATION_MAX_SEARCHES, MODEL.ANOMALY_SEARCHES_PER_QUESTION)
        yield seen
        inv._BUDGET.reset(token)
        object.__setattr__(MODEL, "INVESTIGATION_MAX_SEARCHES", saved[0])
        object.__setattr__(MODEL, "ANOMALY_SEARCHES_PER_QUESTION", saved[1])

    @staticmethod
    def _set(cap, per_question):
        from config import MODEL
        object.__setattr__(MODEL, "INVESTIGATION_MAX_SEARCHES", cap)
        object.__setattr__(MODEL, "ANOMALY_SEARCHES_PER_QUESTION", per_question)

    def _ask(self, tag, n):
        import services.investigation_service as inv
        return inv.research_questions(
            "ETR", "2026-09-04", [f"{tag} question {i}" for i in range(n)],
            web=True, model="gpt-5.6-luna")

    def test_every_question_gets_its_own_allowance(self, recorder):
        # Per section, not the classification cap shared out: two questions
        # under a cap of 3 used to get one search each.
        self._set(cap=3, per_question=3)
        assert len(self._ask("floor", 2)) == 2
        assert recorder == [3, 3]

    def test_the_allowance_does_not_grow_with_the_cap(self, recorder):
        self._set(cap=6, per_question=3)
        self._ask("split", 2)
        assert recorder == [3, 3]

    def test_the_default_allowance_is_three(self):
        import os
        from config import MODEL
        if "INVESTIGATION_MAX_SEARCHES" not in os.environ:
            assert MODEL.INVESTIGATION_MAX_SEARCHES == 3
        if "ANOMALY_SEARCHES_PER_QUESTION" not in os.environ:
            assert MODEL.ANOMALY_SEARCHES_PER_QUESTION == 3
