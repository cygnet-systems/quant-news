"""The prior stance must never reach the research call.

The SINCE LAST REPORT line is written by a separate post-hoc call once the
new decision is already extracted, so the fresh prediction cannot anchor to
the previous one. These tests pin the separation: the research prompt carries
no prior-stance block, and the line writer can only record, not decide.
"""

from models.single_agent import (
    SINGLE_AGENT_PROMPT,
    _insert_since_line,
    _since_last_report_line,
)


class _StubLLM:
    def __init__(self, reply=None, raise_=False):
        self.reply = reply
        self.raise_ = raise_
        self.calls = []

    def generate(self, prompt, *args, usage_out=None, **kwargs):
        self.calls.append(prompt)
        if self.raise_:
            raise RuntimeError("provider down")
        if usage_out is not None:
            usage_out.update({"input_tokens": 100, "output_tokens": 20})
        return self.reply


VERDICT = (
    "## Verdict\n"
    "FINAL TRANSACTION PROPOSAL: **SELL**\n"
    "CONVICTION: **0.62**\n"
    "MEASURED ACCURACY: 48% on 21 scored SELL calls\n"
    "REASSESS_TO_BUY: close above $150.00\n"
    "\n### 1. Situation: nothing situational\ntext\n"
)

PRIOR = {"trade_date": "2026-08-28", "decision": "BUY", "confidence": 0.55,
         "report_text": "FINAL TRANSACTION PROPOSAL: **BUY**\n"}


def test_research_prompt_carries_no_prior_stance():
    assert "{continuity_block}" not in SINGLE_AGENT_PROMPT
    assert "PRIOR STANCE" not in SINGLE_AGENT_PROMPT
    # The model is not asked to write the line; it is inserted afterwards.
    assert "SINCE LAST REPORT" not in SINGLE_AGENT_PROMPT


def test_insert_after_measured_accuracy():
    out = _insert_since_line(VERDICT, "prior report 2026-08-28 called BUY")
    lines = out.splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("MEASURED ACCURACY"))
    assert lines[i + 1] == "SINCE LAST REPORT: prior report 2026-08-28 called BUY"


def test_insert_is_idempotent_and_tolerates_missing_anchors():
    once = _insert_since_line(VERDICT, "x")
    assert _insert_since_line(once, "y") == once
    assert _insert_since_line("free-form text, no verdict block", "x") == \
        "free-form text, no verdict block"


def test_no_prior_yields_fixed_line_without_any_call():
    llm = _StubLLM(reply="should never be used")
    line, source = _since_last_report_line(
        llm, None, "", "AAPL", "2026-09-02", "SELL", 0.6, "m", "p", {})
    assert (line, source) == ("no prior report on record", "none")
    assert not llm.calls


def test_writer_sees_both_calls_but_output_is_just_the_line():
    llm = _StubLLM(reply="Prior 2026-08-28 BUY; no trigger met; call flipped.")
    usage = {"input_tokens": 0, "output_tokens": 0}
    line, source = _since_last_report_line(
        llm, PRIOR, "continuity text", "AAPL", "2026-09-02", "SELL", 0.62,
        "m", "p", usage)
    assert source == "model"
    assert line == "Prior 2026-08-28 BUY; no trigger met; call flipped."
    # The writer is told the already-fixed decision; it records, not decides.
    assert "SELL" in llm.calls[0] and "continuity text" in llm.calls[0]
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 20


def test_provider_failure_falls_back_to_deterministic_record():
    line, source = _since_last_report_line(
        _StubLLM(raise_=True), PRIOR, "c", "AAPL", "2026-09-02", "SELL", 0.62,
        "m", "p", {"input_tokens": 0, "output_tokens": 0})
    assert source == "fallback"
    assert "2026-08-28" in line and "BUY" in line and "SELL" in line


def test_same_cutoff_flip_is_labelled_in_fallback():
    prior = {**PRIOR, "trade_date": "2026-09-02"}
    line, source = _since_last_report_line(
        _StubLLM(raise_=True), prior, "c", "AAPL", "2026-09-02", "SELL", 0.62,
        "m", "p", {"input_tokens": 0, "output_tokens": 0})
    assert source == "fallback"
    assert "identical evidence" in line
