"""The defaults every entry point shares: one news window, one evidence
list, and an options block that names the by-expiry skew.

The news window default (14 days) lives in the FRONTEND only, the Run
dialog select, the job form and the seeded daily job all seed from
config.NEWS_LOOKBACK_DAYS. No code path may fall back to it: a run whose
window did not arrive raises RunParameterMissing (2026-09-02, owner's
rule: "if the frontend doesn't contain this parameter, let it crash")."""

import pytest

from config import MODEL
from services.news_window import RunParameterMissing, normalize_lookback
from services.options_service import format_options_block


def test_news_window_default_is_fourteen_days_on_the_frontend():
    assert MODEL.NEWS_LOOKBACK_DAYS == 14
    from layouts.modals import _report_param_selects
    assert _report_param_selects("run")["lookback"].value == "14"
    from services.scheduler_service import default_run_params
    assert default_run_params()["lookback"] == 14


def test_run_dialog_recommendations_default_matches_the_standard_preset():
    # The dialog's own default is the Standard preset, and the synthesis
    # is one of the two outputs a default run has to produce; the Schedule
    # modal seeds the same value from the config default.
    from layouts.modals import DEFAULT_RUN_PRESET, _report_param_selects, preset_fields
    assert DEFAULT_RUN_PRESET == "standard"
    assert preset_fields("standard")["recs"] == "auto"
    assert _report_param_selects("run", {"recs": "auto"})["recs"].value == "auto"
    assert _report_param_selects("sj")["recs"].value == "auto"


def test_a_missing_window_is_an_error_not_a_default():
    for raw in (None, ""):
        with pytest.raises(RunParameterMissing):
            normalize_lookback(raw)
    assert normalize_lookback("30") == (False, 30)
    assert normalize_lookback("overnight") == (True, 1)


def test_default_evidence_includes_investigation_and_political():
    assert set(MODEL.DEFAULT_EVIDENCE) >= {"options", "quality",
                                           "investigation", "political"}


def test_options_block_shows_by_expiry_skew():
    m = {"as_of": "2026-09-01", "put_volume": 3000, "call_volume": 400,
         "total_volume": 3400, "put_oi": 20000, "call_oi": 5000,
         "pc_volume": 7.5, "pc_oi": 4.0, "read": "put-tilted",
         "source": "alpha_vantage", "expiry_window": "all"}
    by_expiry = {"as_of": "2026-09-01", "full_chain": 8.12,
                 "by_expiry": [("2026-09-18", 12.94), ("2026-10-16", 0.08)]}
    block = format_options_block("BHF", m, by_expiry)
    assert "Put/Call volume ratio: 7.50" in block
    assert "(full chain 8.12): 2026-09-18 12.94 | 2026-10-16 0.08" in block
    assert "front-month skew" in block
    # Without the by-expiry data the block is unchanged in shape.
    assert "by expiration" not in format_options_block("BHF", m)
    assert format_options_block("BHF", None) == ""


def test_research_prompt_maps_outcomes_and_positioning():
    from models.single_agent import EPILOGUE_INSTRUCTIONS, SINGLE_AGENT_PROMPT
    assert "5. Positioning & Flows" in SINGLE_AGENT_PROMPT
    assert "10. Scenarios" in SINGLE_AGENT_PROMPT
    assert "12. Trade Plan" in SINGLE_AGENT_PROMPT
    assert "Step 3b: Map the outcomes" in SINGLE_AGENT_PROMPT
    assert '"scenarios"' in EPILOGUE_INSTRUCTIONS


def test_congress_block_states_the_skew_as_a_number():
    from services.political_service import format_congress_block
    c = {"as_of": "2026-09-02", "window_days": 180, "n": 22, "buys": 9, "sells": 13,
         "by_party": {"D": {"buys": 4, "sells": 6}, "R": {"buys": 5, "sells": 6}},
         "trades": [], "total_disclosed_visible": 405}
    block = format_congress_block("NVDA", c)
    assert "Skew: balanced overall; by party: D balanced, R balanced" in block
    c2 = {**c, "buys": 2, "sells": 20, "by_party": {"D": {"buys": 0, "sells": 12}}}
    assert "Skew: sell-skewed overall; by party: D sell-skewed" in format_congress_block("X", c2)
