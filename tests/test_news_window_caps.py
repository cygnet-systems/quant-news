"""The news window a user asks for is the news window the models read.

2026-09-01: "30 days" was selected in the Run dialog, the fetch honoured it,
and the models still saw the last week. Three newest-first truncations
(fetch cap, research prompt ``[:20]``, synthesis ``[:15]``) each dropped the
OLDEST articles, and the models stage read the browser's rolling-7-day news
store regardless of the dialog. These tests pin the replacements:

* prompts sample ACROSS the window (``select_spread``), never the newest N;
* the fetch cap is explicit (0 = all) and REPORTS what it dropped;
* every run path goes through one fetcher whose stats say ok/empty/
  unavailable per symbol.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import services.news_window as nw
from services.news_service import NewsUnavailable


def _art(days_ago: int, rel: float = 0.5, symbol: str = "TYL", hour: int = 12):
    ts = datetime(2026, 8, 31, hour, tzinfo=timezone.utc) - timedelta(days=days_ago)
    return {"symbol": symbol, "title": f"t-{days_ago}", "published_at": ts.isoformat(),
            "ticker_relevance_score": rel, "summary": ""}


class TestSelectSpread:
    def test_covers_the_whole_window_not_just_the_newest_days(self):
        """30 daily articles, budget 5: the pick must reach the oldest week."""
        arts = [_art(d) for d in range(30)]
        chosen = nw.select_spread(arts, 5)
        ages = sorted(int(a["title"].split("-")[1]) for a in chosen)
        assert len(chosen) == 5
        assert ages[0] <= 5 and ages[-1] >= 24, ages

    def test_prefers_relevance_within_a_stratum(self):
        arts = [_art(1, rel=0.2), _art(1, rel=0.9, hour=9), _art(20, rel=0.3)]
        chosen = nw.select_spread(arts, 2)
        assert {a["ticker_relevance_score"] for a in chosen} == {0.9, 0.3}

    def test_no_cap_and_small_lists_pass_through_newest_first(self):
        arts = [_art(3), _art(1), _art(2)]
        assert [a["title"] for a in nw.select_spread(arts, 0)] == ["t-1", "t-2", "t-3"]
        assert [a["title"] for a in nw.select_spread(arts, 10)] == ["t-1", "t-2", "t-3"]

    def test_fills_empty_strata_from_leftovers(self):
        """All articles on two days, budget 4: still returns 4."""
        arts = [_art(0, rel=r) for r in (0.1, 0.2, 0.3)] + [_art(29, rel=0.9)]
        assert len(nw.select_spread(arts, 4)) == 4

    def test_works_on_dataclass_style_articles(self):
        arts = [SimpleNamespace(published_at=datetime(2026, 8, 1 + d),
                                ticker_relevance_score=0.5) for d in range(20)]
        assert len(nw.select_spread(arts, 4)) == 4


class TestCapNewest:
    def test_zero_means_all(self):
        kept, stats = nw.cap_newest([_art(d) for d in range(40)], 0, "2026-08-31")
        assert len(kept) == 40 and stats["capped"] is False and stats["cap"] == 0

    def test_cap_keeps_newest_and_reports_the_effective_window(self):
        kept, stats = nw.cap_newest([_art(d) for d in range(40)], 10, "2026-08-31")
        assert [a["title"] for a in kept][:3] == ["t-0", "t-1", "t-2"]
        assert stats == {
            "fetched": 40, "kept": 10, "cap": 10, "capped": True,
            "oldest": "2026-08-22", "newest": "2026-08-31", "effective_days": 10,
        }

    def test_empty(self):
        kept, stats = nw.cap_newest([], 5, "2026-08-31")
        assert kept == [] and stats["oldest"] is None and stats["effective_days"] is None


class TestNormalizers:
    """The window and cap come from the frontend or not at all."""

    @pytest.mark.parametrize("raw, expected", [
        ("30", (False, 30)), (365, (False, 365)), ("overnight", (True, 1)),
    ])
    def test_lookback(self, raw, expected):
        assert nw.normalize_lookback(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "seven", "0", -3])
    def test_missing_or_invalid_lookback_raises_instead_of_defaulting(self, raw):
        with pytest.raises(nw.RunParameterMissing):
            nw.normalize_lookback(raw)

    def test_article_cap_zero_is_all_and_missing_raises(self):
        assert nw.normalize_article_cap(0) == 0
        assert nw.normalize_article_cap("100") == 100
        for raw in (None, "", "lots", -5):
            with pytest.raises(nw.RunParameterMissing):
                nw.normalize_article_cap(raw)

    def test_no_library_default_exists(self):
        assert not hasattr(nw, "DEFAULT_NEWS_LOOKBACK_DAYS")


class TestFetchRunNews:
    def setup_method(self):
        nw.clear_pit_news_cache()

    def test_status_per_symbol_and_dict_shape(self):
        def fake(symbol, as_of, lookback_days, max_articles, relevance_threshold=0.5):
            if symbol == "DOWN":
                raise NewsUnavailable("rate limit")
            if symbol == "QUIET":
                return nw.cap_newest([], max_articles, as_of)
            arts = [SimpleNamespace(id="1", symbol=symbol, title="x", source="s",
                                    url="u", published_at=datetime(2026, 8, 30),
                                    summary=None, sentiment=None, sentiment_score=None,
                                    impact=None, price_change_percent=None,
                                    ticker_relevance_score=0.7, topics=None,
                                    overall_sentiment_score=None,
                                    overall_sentiment_label=None)]
            return nw.cap_newest(arts, max_articles, as_of)

        slept = []
        with patch.object(nw, "fetch_point_in_time_news_with_stats", side_effect=fake):
            by_sym, stats = nw.fetch_run_news(
                ["OK", "QUIET", "DOWN"], "2026-08-31", "2026-09-01",
                overnight=False, lookback_days=30, max_articles=100,
                retries=3, sleep=slept.append)

        assert stats["OK"]["status"] == "ok" and by_sym["OK"][0]["published_at"] == "2026-08-30T00:00:00"
        assert stats["QUIET"]["status"] == "empty" and by_sym["QUIET"] == []
        assert stats["DOWN"]["status"] == "unavailable" and "rate limit" in stats["DOWN"]["error"]
        assert slept == [2.0, 4.0], "NewsUnavailable is retried with backoff"

    def test_payload_and_description_surface_the_cap(self):
        stats = {
            "NVDA": {**nw.cap_newest([_art(d, symbol="NVDA") for d in range(30)], 5,
                                     "2026-08-31")[1], "status": "ok"},
            "TYL": {**nw.cap_newest([_art(20, symbol="TYL")], 5, "2026-08-31")[1],
                    "status": "ok"},
        }
        payload = nw.news_window_payload(
            overnight=False, lookback_days=30, max_articles=5, as_of="2026-08-31",
            target="2026-09-01", stats_by_symbol=stats)
        assert payload["articles"] == 6
        assert payload["by_symbol"]["NVDA"]["capped"] is True
        assert payload["by_symbol"]["NVDA"]["effective_days"] == 5
        assert payload["by_symbol"]["TYL"]["capped"] is False
        line = nw.describe_news_window(payload)
        assert "CAP HIT" in line and "NVDA 30→5 (effective 5d)" in line
        assert "TYL" not in line.split("CAP HIT")[1]


class TestFetchCapIsExplicit:
    """The vendor client no longer silently keeps the newest 50."""

    def test_alpha_vantage_default_is_uncapped(self):
        import inspect
        import services.news_service as ns
        assert inspect.signature(ns.fetch_alpha_vantage_news).parameters["max_articles"].default == 0

    def test_point_in_time_fetch_reports_and_caches_stats(self):
        import services.news_service as ns
        nw.clear_pit_news_cache()
        arts = [SimpleNamespace(published_at=datetime(2026, 8, 31 - d), ticker_relevance_score=0.6)
                for d in range(12)]
        with patch.object(ns, "fetch_alpha_vantage_news", return_value=arts) as av:
            kept, stats = nw.fetch_point_in_time_news_with_stats(
                "TYL", "2026-08-31", lookback_days=30, max_articles=4)
            again = nw.fetch_point_in_time_news("TYL", "2026-08-31", lookback_days=30,
                                                max_articles=4)
        assert av.call_count == 1, "second call is served from the PIT cache"
        assert "max_articles" not in av.call_args.kwargs, "no hidden vendor-side cap"
        assert len(kept) == 4 and again == kept
        assert stats["fetched"] == 12 and stats["capped"] and stats["effective_days"] == 4
