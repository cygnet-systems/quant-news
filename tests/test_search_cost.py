"""A server-side web search is billed per call, so it has to be counted.

llm_usage priced tokens and nothing else. The investigation stage is the one
stage that pays a provider to browse, so it reported roughly half its true
cost: on 2026-09-04 the table read $9.67 for the day against a real bill near
$20, and 881 unrecorded searches were the difference. The count is observed
from the provider's response; only the RATE is a constant, so a wrong rate is
a one-line fix that re-prices every stored count.
"""

import pytest

from config import WEB_SEARCH_PRICING, get_web_search_rate
from services.usage_service import compute_tool_cost


class TestTheRate:
    def test_known_providers_are_priced(self):
        assert get_web_search_rate("openai") == WEB_SEARCH_PRICING["openai"]
        assert get_web_search_rate("anthropic") == WEB_SEARCH_PRICING["anthropic"]

    def test_case_and_padding_do_not_lose_the_rate(self):
        assert get_web_search_rate("  OpenAI ") == WEB_SEARCH_PRICING["openai"]

    def test_an_unknown_provider_is_unpriced_not_free(self):
        assert get_web_search_rate("lm_studio") is None


class TestTheCost:
    def test_searches_are_priced_per_thousand(self):
        rate = WEB_SEARCH_PRICING["openai"]
        assert compute_tool_cost("openai", 1000) == pytest.approx(rate)
        assert compute_tool_cost("openai", 8) == pytest.approx(rate * 8 / 1000)

    def test_no_searches_is_a_real_zero(self):
        """Every non-searching call in the table must read 0, not NULL:
        an unknown cost and a call that browsed nothing are different facts."""
        assert compute_tool_cost("openai", 0) == 0.0
        assert compute_tool_cost("who", 0) == 0.0

    def test_an_unpriced_provider_that_searched_is_NULL_not_zero(self):
        """A guessed zero is how the fee went missing in the first place."""
        assert compute_tool_cost("lm_studio", 5) is None

    def test_the_09_04_gap_is_explained_by_the_searches(self):
        """881 billed searches against a ledger that said $9.67 and a bill
        near $20. Pins the arithmetic that closed that gap."""
        tokens_only = 9.67
        tool = compute_tool_cost("openai", 881)
        assert tool == pytest.approx(8.81, abs=0.01)
        assert tokens_only + tool == pytest.approx(18.5, abs=0.2)


class TestTheLedgerColumn:
    def test_llm_usage_carries_both_costs_separately(self):
        """cost_usd keeps meaning TOKENS so historical rows still say what
        they said; the real number is the sum of the two."""
        from db.models import LLMUsage

        cols = LLMUsage.__table__.c
        assert "searches" in cols and "tool_cost_usd" in cols
        assert cols["searches"].nullable is False
        assert cols["tool_cost_usd"].nullable is True


class TestCacheVisibility:
    """Whether prompt caching lands must be a query, not an assumption.

    Measured 2026-09-05: the investigation prompt averages ~10,660 input
    tokens, of which the stable prefix (the system prompt) is ~458. OpenAI
    caches only prefixes over 1024 tokens, so caching cannot be hitting at
    all today, and moving the fixed TASK/schema block above the per-symbol
    evidence would still only reach ~856. That is a negative result worth
    keeping: the input cost here is per-symbol evidence, not repeated
    boilerplate, so caching is not the lever.
    """

    def test_the_column_exists_so_the_question_is_answerable(self):
        from db.models import LLMUsage

        assert "cached_input_tokens" in LLMUsage.__table__.c

    def test_the_stable_prefix_is_below_the_openai_cache_floor(self):
        """Pins the finding. If someone later moves the schema into the
        system prompt this fails, which is the moment to re-measure."""
        import services.investigation_service as inv

        system = inv._system_prompt(web=True)
        approx_tokens = len(system) // 4
        assert approx_tokens < 1024, (
            "the stable prefix now clears OpenAI's 1024-token cache floor: "
            "re-measure whether caching is worth enabling")
