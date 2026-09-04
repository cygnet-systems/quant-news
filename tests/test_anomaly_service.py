"""What a report is allowed to be about.

The detector decides which sections a run writes, so its failure modes are
about claiming a story exists:

* a thin chain, two insiders, four failing checks and a quiet news window are
  all NORMAL, and each has to come back silent rather than as a low-severity
  section a researcher then spends a web search on;
* a missing block is not a neutral block. Nothing here may invent an anomaly
  from an argument that was never passed;
* a zero-priced Form 4 row (grant, gift, option exercise) carries no dollar
  value, so no dollar floor may ever be cleared by one;
* ranking is what the cap acts on, and the cross-source reading
  (positioning_vs_price) has to outrank the single-source ones or the section
  that says something about the future gets dropped for one that recites the
  past;
* detection is arithmetic over blocks the run already has: no vendor call, no
  LLM call, no database read. The stubs below fail the test if any is made.

Input shapes are built through their real producers wherever one exists
(``options_service._aggregate`` from vendor contracts, ``summarize_insiders``
over reader rows, ``bad_apples_service.summarize`` over a screen result), so a
change to those shapes breaks these tests rather than silently starving the
detectors.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from services import anomaly_service as anom
from services import av_store, bad_apples_service, insider_service, options_service
from services.news_service import NewsArticle

AS_OF = "2026-09-02"


@pytest.fixture(autouse=True)
def no_outside_world(monkeypatch):
    """Every door out of the process, wired to fail the test."""
    def forbidden(*args, **kwargs):
        raise AssertionError("anomaly detection reached outside the process")

    import requests
    import db.session as dbs
    from services import alpha_vantage, llm_service

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(alpha_vantage, "fetch", forbidden)
    monkeypatch.setattr(av_store, "fetch", forbidden)
    monkeypatch.setattr(llm_service, "get_llm", forbidden)
    monkeypatch.setattr(dbs, "get_session", forbidden)


# --- seeds ---------------------------------------------------------------

def chain(put_vol, call_vol, put_oi=0, call_oi=0, when=AS_OF):
    """Metrics through the vendor path, so the keys are the vendor's."""
    contracts = ([{"type": "put", "volume": put_vol, "open_interest": put_oi}]
                 + [{"type": "call", "volume": call_vol, "open_interest": call_oi}])
    return options_service._aggregate(contracts, when)


def insider_row(executive, side, shares=1000, price=100.0, days_ago=10,
                title="EVP"):
    """One row in ``av_store.insider_transactions_for`` shape. A price of 0 is
    how the vendor sends grants and exercises; value_usd is NULL for those."""
    when = (datetime.fromisoformat(AS_OF) - timedelta(days=days_ago)).date()
    return {"symbol": "NVDA", "executive": executive, "title": title,
            "security_type": "Common Stock", "side": side,
            "transaction_date": when.isoformat(),
            "visible_from": when.isoformat(), "shares": float(shares),
            "share_price": float(price),
            "value_usd": (shares * price) if price else None}


def summary(rows, monkeypatch):
    """The same rows through ``summarize_insiders``, which is the other shape
    ``detect`` accepts. No DB: the reader is stubbed."""
    monkeypatch.setattr(av_store, "insider_transactions_for",
                        lambda *a, **k: list(rows))
    return insider_service.summarize_insiders("NVDA", AS_OF)


def congress_trade(politician, kind="PURCHASE", bioguide="P000001",
                   lo=15001, hi=50000, days_ago=20):
    when = (datetime.fromisoformat(AS_OF) - timedelta(days=days_ago)).date()
    return {"symbol": "NVDA", "bioguide_id": bioguide, "politician": politician,
            "party": "D", "chamber": "House", "state": "CA-11",
            "type": kind, "owner": "self", "asset_name": "NVIDIA Corp",
            "transaction_date": when.isoformat(),
            "filed_date": (when + timedelta(days=14)).isoformat(),
            "amount_min": float(lo), "amount_max": float(hi)}


def quality(fails, checks=20):
    """A screen result put through ``bad_apples_service.summarize``."""
    rows = [{"category": "valuation", "check": f"check {i}", "status": "fail",
             "value": "1.0", "threshold": "0.5", "note": ""}
            for i in range(fails)]
    rows += [{"category": "growth", "check": f"ok {i}", "status": "pass",
              "value": "", "threshold": "", "note": ""}
             for i in range(checks - fails)]
    scores = {}
    for c in rows:
        s = scores.setdefault(c["category"], {"fail": 0, "pass": 0, "n/a": 0})
        s[c["status"]] += 1
    return bad_apples_service.summarize({
        "as_of": AS_OF, "flag": "caution", "total_fails": fails,
        "total_checks": checks, "scores": scores, "checks": rows,
        "red_flags": [], "red_flag_scope": "run news window (0 articles)"})


def article(day, n=0):
    return NewsArticle(
        id=f"{day}-{n}", symbol="NVDA", title=f"NVDA headline {day} #{n}",
        source="wire", url=f"https://example.invalid/{day}/{n}",
        published_at=datetime.fromisoformat(day).replace(tzinfo=timezone.utc),
        summary="")


def bars(*closes):
    idx = pd.date_range(end=AS_OF, periods=len(closes), freq="D")
    return pd.DataFrame({"Close": list(closes)}, index=idx)


def ramp(start, end, n=25):
    step = (end - start) / (n - 1)
    return bars(*[start + step * i for i in range(n)])


def only(result, key):
    hits = [a for a in result if a["key"] == key]
    return hits[0] if hits else None


# --- detectors -----------------------------------------------------------

class TestOptionsSkew:
    def test_a_liquid_put_tilted_chain_fires_with_its_own_numbers(self):
        a = only(anom.detect("NVDA", AS_OF,
                             options=chain(9000, 3000, put_oi=40000, call_oi=20000)),
                 "options_skew")
        assert a is not None
        assert a["evidence_key"] == "options"
        assert "3.00 puts traded per call" in a["facts"][0]
        assert "12,000" in a["facts"][0]
        assert a["question"].endswith("?") and "NVDA" in a["question"]

    def test_a_balanced_chain_is_not_an_anomaly(self):
        assert anom.detect("NVDA", AS_OF, options=chain(3000, 4000)) == []

    def test_a_chain_too_thin_to_read_is_silent_however_lopsided(self):
        # 20 puts to 1 call is 20.0, and meaningless: one 20-lot set it.
        thin = chain(20, 1)
        assert thin["read"] == "low-liquidity"
        assert anom.detect("NVDA", AS_OF, options=thin) == []

    def test_no_chain_at_all_produces_nothing(self):
        assert anom.detect("NVDA", AS_OF, options=None) == []

    def test_severity_grows_with_distance_from_the_neutral_band(self):
        mild = only(anom.detect("NVDA", AS_OF, options=chain(5500, 5000)),
                    "options_skew")
        extreme = only(anom.detect("NVDA", AS_OF, options=chain(11000, 1000)),
                       "options_skew")
        assert mild["severity"] < extreme["severity"]

    def test_a_call_tilted_chain_fires_on_the_other_side(self):
        a = only(anom.detect("NVDA", AS_OF, options=chain(1000, 9000)),
                 "options_skew")
        assert "call-tilted" in a["title"]


class TestTermDivergence:
    def near(self, ratio, full=1.0):
        return {"as_of": AS_OF, "full_chain": full,
                "by_expiry": [("2026-09-18", ratio), ("2026-10-16", full),
                              ("2026-12-18", full)]}

    def test_a_front_month_far_off_the_chain_fires(self):
        a = only(anom.detect("NVDA", AS_OF, by_expiry=self.near(3.0)),
                 "options_term_divergence")
        assert a is not None
        assert "2026-09-18" in a["facts"][0] and "3.00" in a["facts"][0]
        assert "bracing for" in a["question"]

    def test_a_front_month_within_half_the_chain_is_silent(self):
        assert anom.detect("NVDA", AS_OF, by_expiry=self.near(1.4)) == []

    def test_the_earliest_expiry_is_the_one_read(self):
        d = {"as_of": AS_OF, "full_chain": 1.0,
             "by_expiry": [("2026-12-18", 1.0), ("2026-09-18", 4.0)]}
        a = only(anom.detect("NVDA", AS_OF, by_expiry=d),
                 "options_term_divergence")
        assert "2026-09-18" in a["facts"][0]

    def test_a_missing_full_chain_or_empty_curve_produces_nothing(self):
        assert anom.detect("NVDA", AS_OF, by_expiry={
            "as_of": AS_OF, "full_chain": None,
            "by_expiry": [("2026-09-18", 3.0)]}) == []
        assert anom.detect("NVDA", AS_OF, by_expiry={
            "as_of": AS_OF, "full_chain": 1.0, "by_expiry": []}) == []
        assert anom.detect("NVDA", AS_OF, by_expiry=None) == []


class TestInsiderCluster:
    def sellers(self, n):
        return [insider_row(f"Exec {i}", "D", shares=500) for i in range(n)]

    def test_three_one_way_executives_are_a_cluster(self):
        a = only(anom.detect("NVDA", AS_OF, insiders=self.sellers(3)),
                 "insider_cluster")
        assert a is not None and a["evidence_key"] == "insiders"
        assert "3 distinct executives" in a["facts"][0]

    def test_two_are_not(self):
        assert anom.detect("NVDA", AS_OF, insiders=self.sellers(2)) == []

    def test_an_executive_who_traded_both_ways_does_not_count_toward_a_side(self):
        rows = self.sellers(2) + [insider_row("Exec 2", "D"),
                                  insider_row("Exec 2", "A")]
        assert anom.detect("NVDA", AS_OF, insiders=rows) == []

    def test_the_summary_shape_reads_the_same_as_the_rows(self, monkeypatch):
        rows = self.sellers(3)
        from_rows = only(anom.detect("NVDA", AS_OF, insiders=rows),
                         "insider_cluster")
        from_summary = only(
            anom.detect("NVDA", AS_OF, insiders=summary(rows, monkeypatch)),
            "insider_cluster")
        assert from_summary is not None
        assert from_summary["severity"] == from_rows["severity"]

    def test_one_disposal_over_the_floor_is_its_own_anomaly(self):
        rows = [insider_row("Jensen Huang", "D", shares=100_000, price=180.0,
                            title="CEO")]
        a = only(anom.detect("NVDA", AS_OF, insiders=rows), "insider_cluster")
        assert a is not None
        assert "$18.0M" in a["facts"][0] and "Jensen Huang" in a["facts"][0]
        assert "10b5-1" in a["question"]

    def test_one_disposal_under_the_floor_is_routine(self):
        rows = [insider_row("Jensen Huang", "D", shares=1000, price=180.0)]
        assert anom.detect("NVDA", AS_OF, insiders=rows) == []

    def test_a_zero_priced_row_can_never_clear_a_dollar_floor(self):
        # A grant of a million shares at the vendor's 0.0 is worth nothing
        # nameable; treating it as a purchase or a sale is the whole trap.
        rows = [insider_row("Jensen Huang", "D", shares=1_000_000, price=0.0)]
        assert rows[0]["value_usd"] is None
        assert anom.detect("NVDA", AS_OF, insiders=rows) == []

    def test_no_insider_input_produces_nothing(self):
        assert anom.detect("NVDA", AS_OF, insiders=None) == []
        assert anom.detect("NVDA", AS_OF, insiders=[]) == []


class TestCongressActivity:
    def test_a_single_disclosed_trade_is_already_the_exception(self):
        a = only(anom.detect("NVDA", AS_OF,
                             congress=[congress_trade("Rep. A")]),
                 "congress_activity")
        assert a is not None and a["evidence_key"] == "politicians"
        assert "Rep. A" in a["facts"][0]
        assert any("filing date" in f for f in a["facts"])

    def test_two_members_outrank_one(self):
        one = only(anom.detect("NVDA", AS_OF,
                               congress=[congress_trade("Rep. A")]),
                   "congress_activity")
        two = only(anom.detect("NVDA", AS_OF, congress=[
            congress_trade("Rep. A"),
            congress_trade("Rep. B", bioguide="B000002")]),
            "congress_activity")
        assert two["severity"] > one["severity"]

    def test_a_purchase_into_a_decline_outranks_the_same_purchase_into_a_rally(self):
        buy = [congress_trade("Rep. A", "PURCHASE")]
        falling = only(anom.detect("NVDA", AS_OF, congress=buy,
                                   ohlcv=ramp(200.0, 150.0)), "congress_activity")
        rising = only(anom.detect("NVDA", AS_OF, congress=buy,
                                  ohlcv=ramp(150.0, 200.0)), "congress_activity")
        assert falling["severity"] > rising["severity"]
        assert any("buying into a decline" in f for f in falling["facts"])

    def test_the_dossier_names_the_member_when_it_has_a_profile(self):
        dossier = {"symbol": "NVDA", "entries": [
            {"bioguide_id": "P000001",
             "identity": "Nancy Example (D, House, CA-11)"}]}
        a = only(anom.detect("NVDA", AS_OF, congress=[congress_trade("Rep. A")],
                             dossier=dossier), "congress_activity")
        assert "Nancy Example (D, House, CA-11)" in a["facts"][0]

    def test_an_empty_window_produces_nothing(self):
        assert anom.detect("NVDA", AS_OF, congress=[]) == []
        assert anom.detect("NVDA", AS_OF, congress=None) == []


class TestQualityFailures:
    def test_the_screens_own_caution_boundary_is_the_floor(self):
        assert anom.detect("NVDA", AS_OF, quality=quality(4)) == []
        a = only(anom.detect("NVDA", AS_OF, quality=quality(5)),
                 "quality_failures")
        assert a is not None and a["evidence_key"] == "quality"
        assert "5 of 20" in a["facts"][0] and "CAUTION" in a["facts"][0]

    def test_a_bad_apple_outranks_a_caution_and_says_so(self):
        caution = only(anom.detect("NVDA", AS_OF, quality=quality(5)),
                       "quality_failures")
        bad = only(anom.detect("NVDA", AS_OF, quality=quality(11)),
                   "quality_failures")
        assert bad["severity"] > caution["severity"]
        assert "BAD APPLE" in bad["facts"][0]

    def test_it_refuses_to_read_as_direction(self):
        a = only(anom.detect("NVDA", AS_OF, quality=quality(9)),
                 "quality_failures")
        assert any("not a timing signal" in f for f in a["facts"])

    def test_no_screen_produces_nothing(self):
        assert anom.detect("NVDA", AS_OF, quality=None) == []
        assert anom.detect("NVDA", AS_OF, quality={}) == []


class TestNewsSpike:
    def quiet(self):
        return [article("2026-08-29"), article("2026-08-30"),
                article("2026-08-31"), article("2026-09-01")]

    def test_a_day_far_above_the_symbols_own_norm_fires(self):
        news = self.quiet() + [article("2026-09-02", n) for n in range(6)]
        a = only(anom.detect("NVDA", AS_OF, news=news), "news_spike")
        assert a is not None and a["evidence_key"] == "news_source"
        assert "6 articles on 2026-09-02" in a["facts"][0]
        assert any("NVDA headline" in f for f in a["facts"])

    def test_a_flat_window_is_not_a_spike(self):
        news = self.quiet() + [article("2026-09-02"), article("2026-09-02", 1)]
        assert anom.detect("NVDA", AS_OF, news=news) == []

    def test_a_window_too_short_to_have_a_norm_produces_nothing(self):
        # The overnight close-to-open window is one or two days wide; a spike
        # against a baseline of one day is not a measurement.
        news = [article("2026-09-01")] + [article("2026-09-02", n)
                                          for n in range(9)]
        assert anom.detect("NVDA", AS_OF, news=news) == []

    def test_no_news_produces_nothing(self):
        assert anom.detect("NVDA", AS_OF, news=None) == []
        assert anom.detect("NVDA", AS_OF, news=[]) == []


class TestPositioningVsPrice:
    def selling(self):
        return [insider_row(f"Exec {i}", "D") for i in range(3)]

    def test_insiders_and_the_chain_leaning_against_a_rally_fire(self):
        a = only(anom.detect("NVDA", AS_OF, ohlcv=ramp(150.0, 200.0),
                             options=chain(9000, 3000),
                             insiders=self.selling()), "positioning_vs_price")
        assert a is not None and a["evidence_key"] == "ohlcv"
        assert "+26.3% over 20 sessions" in a["facts"][0]
        assert any("how the symbol may behave" in f for f in a["facts"])

    def test_it_outranks_the_single_source_readings_it_is_built_from(self):
        found = anom.detect("NVDA", AS_OF, ohlcv=ramp(150.0, 200.0),
                            options=chain(9000, 3000), insiders=self.selling())
        assert found[0]["key"] == "positioning_vs_price"
        assert found[0]["severity"] > max(a["severity"] for a in found[1:])

    def test_positioning_that_agrees_with_the_tape_is_not_an_anomaly(self):
        found = anom.detect("NVDA", AS_OF, ohlcv=ramp(200.0, 150.0),
                            options=chain(9000, 3000), insiders=self.selling())
        assert only(found, "positioning_vs_price") is None

    def test_one_party_alone_is_not_two(self):
        # Chain leaning bearish, insiders not leaning at all.
        found = anom.detect("NVDA", AS_OF, ohlcv=ramp(150.0, 200.0),
                            options=chain(9000, 3000),
                            insiders=[insider_row("Exec 0", "D")])
        assert only(found, "positioning_vs_price") is None

    def test_a_flat_tape_has_nothing_to_lean_against(self):
        found = anom.detect("NVDA", AS_OF, ohlcv=ramp(180.0, 181.0),
                            options=chain(9000, 3000), insiders=self.selling())
        assert only(found, "positioning_vs_price") is None

    def test_without_price_bars_it_produces_nothing(self):
        found = anom.detect("NVDA", AS_OF, options=chain(9000, 3000),
                            insiders=self.selling())
        assert only(found, "positioning_vs_price") is None
        assert anom.price_trend(None) is None
        assert anom.price_trend(bars(1.0, 2.0, 3.0)) is None


class TestRankingAndCap:
    def everything(self):
        return dict(
            options=chain(11000, 1000, put_oi=40000, call_oi=10000),
            by_expiry={"as_of": AS_OF, "full_chain": 1.0,
                       "by_expiry": [("2026-09-18", 4.0)]},
            insiders=[insider_row(f"Exec {i}", "D") for i in range(4)],
            congress=[congress_trade("Rep. A"),
                      congress_trade("Rep. B", bioguide="B000002")],
            quality=quality(11),
            news=([article("2026-08-29"), article("2026-08-30"),
                   article("2026-08-31"), article("2026-09-01")]
                  + [article("2026-09-02", n) for n in range(8)]),
            ohlcv=ramp(150.0, 200.0),
        )

    def test_the_list_is_capped_and_ordered_by_severity(self, caplog):
        with caplog.at_level("INFO", logger="services.anomaly_service"):
            found = anom.detect("NVDA", AS_OF, **self.everything())
        assert len(found) == anom.MAX_ANOMALIES
        assert [a["severity"] for a in found] == sorted(
            (a["severity"] for a in found), reverse=True)
        assert len({a["key"] for a in found}) == len(found)

    def test_what_was_dropped_is_logged_by_name(self, caplog):
        with caplog.at_level("INFO", logger="services.anomaly_service"):
            anom.detect("NVDA", AS_OF, **self.everything())
        dropped = [r for r in caplog.records if "over the cap" in r.getMessage()]
        assert dropped, "anomalies past the cap must not vanish silently"
        assert "NVDA" in dropped[0].getMessage()

    def test_ties_break_the_same_way_on_every_run(self):
        seeds = self.everything()
        assert ([a["key"] for a in anom.detect("NVDA", AS_OF, **seeds)]
                == [a["key"] for a in anom.detect("NVDA", AS_OF, **seeds)])

    def test_every_anomaly_carries_a_question_and_its_own_numbers(self):
        for a in anom.detect("NVDA", AS_OF, **self.everything()):
            assert set(a) == {"key", "title", "severity", "facts", "question",
                              "evidence_key"}
            assert 0.0 <= a["severity"] <= 1.0
            assert a["question"].endswith("?")
            assert a["facts"] and all(isinstance(f, str) and f for f in a["facts"])
            assert any(ch.isdigit() for ch in a["facts"][0])

    def test_a_quiet_symbol_gets_no_sections_at_all(self):
        assert anom.detect("NVDA", AS_OF) == []
        assert anom.detect("NVDA", AS_OF, options=chain(3000, 4000),
                           insiders=[], congress=[], quality=quality(1),
                           news=[], ohlcv=ramp(180.0, 181.0)) == []


class TestNoOutsideCalls:
    def test_detection_is_arithmetic_over_blocks_the_run_already_has(self):
        # Every detector firing at once, with the autouse fixture holding the
        # process shut. Detection runs on the report path and on a retry, so a
        # vendor call hidden in here would be spent per symbol per run.
        assert anom.detect("NVDA", AS_OF, **TestRankingAndCap().everything())

    def test_the_guard_above_is_not_vacuous(self):
        import requests
        from services import alpha_vantage

        with pytest.raises(AssertionError):
            requests.get("https://example.invalid/")
        with pytest.raises(AssertionError):
            alpha_vantage.fetch("NEWS_SENTIMENT", symbol="NVDA")
