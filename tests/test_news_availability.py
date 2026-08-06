"""Guards for the distinction between "no news" and "no news source".

TYL, 2026-08-04: the window held 48 articles including an acquisition, the
run received 0, and every downstream consumer treated that as a quiet week.
The research report then stated there was "no company-specific news sentiment
to confirm or conflict with the technical picture" -- a hole presented as a
finding, on a prediction that had been stored and scored.

The cause was a funnel. Alpha Vantage signals throttling with HTTP 200 and an
explanatory body, so raise_for_status() sees nothing wrong; the client mapped
both that and every exception to an empty list; the caller mapped its own
exceptions to an empty list too. Four distinct conditions arrived downstream
as the same value.

These tests pin each condition to a distinguishable outcome.
"""

from unittest.mock import patch

import pytest

import services.news_service as ns
from services.news_service import NewsUnavailable


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        """Throttled responses are HTTP 200: this must stay a no-op."""

    def json(self):
        return self._payload


def av(payload):
    return patch.object(ns.requests, "get", return_value=FakeResponse(payload))


class TestSourceFailureIsNotEmptyNews:
    @pytest.mark.parametrize("payload", [
        {"Information": "Our standard API rate limit is 25 requests per day."},
        {"Note": "Thank you for using Alpha Vantage!"},
        {"Error Message": "Invalid API call."},
    ])
    def test_two_hundred_with_an_explanation_is_a_failure(self, payload):
        with av(payload), pytest.raises(NewsUnavailable):
            ns.fetch_alpha_vantage_news("TYL")

    def test_the_vendor_message_survives_into_the_error(self):
        """Whoever reads the log needs to know it was the rate limit."""
        with av({"Information": "rate limit is 25 requests per day"}):
            with pytest.raises(NewsUnavailable, match="rate limit"):
                ns.fetch_alpha_vantage_news("TYL")

    def test_transport_failure_is_a_failure(self):
        with patch.object(ns.requests, "get", side_effect=TimeoutError("timed out")):
            with pytest.raises(NewsUnavailable):
                ns.fetch_alpha_vantage_news("TYL")

    def test_missing_feed_key_is_a_failure(self):
        with av({"unexpected": "shape"}), pytest.raises(NewsUnavailable):
            ns.fetch_alpha_vantage_news("TYL")


class TestGenuineEmptinessStaysEmpty:
    def test_empty_feed_returns_no_articles_without_raising(self):
        """A quiet week is a real answer and must not look like an outage."""
        with av({"feed": []}):
            assert ns.fetch_alpha_vantage_news("TYL") == []

    def test_articles_below_the_relevance_floor_are_not_an_outage(self):
        """Filtering everything out is still a successful fetch."""
        item = {
            "title": "Unrelated market wrap", "url": "http://x", "summary": "",
            "time_published": "20260804T120000", "source": "Test",
            "overall_sentiment_score": 0.0, "overall_sentiment_label": "Neutral",
            "ticker_sentiment": [{"ticker": "OTHER", "relevance_score": "0.9",
                                  "ticker_sentiment_score": "0.1"}],
        }
        with av({"feed": [item]}):
            assert ns.fetch_alpha_vantage_news("TYL", relevance_threshold=0.5) == []


class TestFallbackBehaviour:
    def test_yfinance_stays_lenient(self):
        """The fallback returns empty rather than raising; its caller decides.

        If this raised, a yfinance hiccup would mask the primary source's
        error, which is the more informative one.
        """
        import services.stock_data as sd
        with patch.object(sd, "get_ticker", side_effect=RuntimeError("boom")):
            assert ns.fetch_yfinance_news("TYL") == []

    def test_missing_api_key_is_reported_not_silently_empty(self):
        """API is a frozen dataclass, so swap the whole object."""
        import dataclasses
        keyless = dataclasses.replace(ns.API, ALPHA_VANTAGE_API_KEY="")
        with patch.object(ns, "API", keyless):
            with pytest.raises(NewsUnavailable, match="API key"):
                ns.fetch_alpha_vantage_news("TYL")
