"""What a report is allowed to be about.

The detector decides which sections a run writes, so its failure modes are
about claiming a story exists:

* a thin chain, two insiders, four failing checks and a quiet news window are
  all NORMAL, and each has to come back silent rather than as a low-severity
  section a researcher then spends a web search on;
* a missing block is not a neutral block. Nothing here may invent an anomaly
  from an argument that was never passed;
* a zero-priced Form 4 row (grant, gift, option exercise) carries no dollar
  value, so no dollar floor may ever be cleared by one, a set of them filed on
  one day is an annual grant rather than a cluster, and one landing this week
  does not re-date a priced sale from four months ago;
* insider activity is judged against the symbol's own recent rate over a
  cluster-length window, not by a flat count over the store's whole read
  window, or every large cap trips the detector every day;
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


def bars(*closes, volume=None, opens=None):
    """A Close-only frame by default (what the trend tests use); Volume and
    Open columns only when a test hands them over."""
    idx = pd.date_range(end=AS_OF, periods=len(closes), freq="D")
    cols = {"Close": list(closes)}
    if volume is not None:
        cols["Volume"] = list(volume)
    if opens is not None:
        cols["Open"] = list(opens)
    return pd.DataFrame(cols, index=idx)


def calm(n=61, level=100.0, wobble=0.5):
    """n closes that alternate +/- wobble percent around level: a symbol
    with a small, steady daily move and a known sigma."""
    out, sign = [], 1
    for i in range(n):
        out.append(level * (1 + sign * wobble / 100.0))
        sign = -sign
    return out


def ramp(start, end, n=25):
    step = (end - start) / (n - 1)
    return bars(*[start + step * i for i in range(n)])


def only(result, key):
    hits = [a for a in result if a["key"] == key]
    return hits[0] if hits else None


# --- detectors -----------------------------------------------------------

class TestPriceShock:
    def test_a_move_far_outside_the_symbols_own_range_fires(self):
        closes = calm() + [100.0 * 1.08]        # +8% on a +/-0.5% name
        hit = only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes)), "price_shock")
        assert hit is not None
        assert hit["evidence_key"] == "ohlcv"
        assert "closed up" in hit["facts"][0] and "8." in hit["facts"][0]
        assert "standard deviations" in hit["facts"][1]
        assert AS_OF in hit["question"]

    def test_the_same_move_on_a_volatile_name_is_its_norm(self):
        # +/-6% every day: an 8% day is barely over one sigma.
        closes = [100.0 * (1 + (0.06 if i % 2 else -0.06)) for i in range(61)]
        closes.append(closes[-1] * 1.08)
        assert only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes)), "price_shock") is None

    def test_a_dead_calm_name_still_needs_the_absolute_floor(self):
        # +/-0.05%: a 1% day is 20 sigma but under SHOCK_MIN_PCT.
        closes = calm(wobble=0.05) + [100.0 * 1.01]
        assert only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes)), "price_shock") is None

    def test_the_gap_fact_reads_the_open_when_there_is_one(self):
        closes = calm() + [108.0]
        opens = closes[:-1] + [107.5]           # opened up 7.5%: overnight
        hit = only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes, opens=opens)),
                   "price_shock")
        assert any("overnight" in f for f in hit["facts"])
        opens = closes[:-1] + [closes[-2]]      # opened flat: built in-session
        hit = only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes, opens=opens)),
                   "price_shock")
        assert any("during the session" in f for f in hit["facts"])

    def test_too_few_bars_produce_nothing(self):
        closes = calm(n=15) + [108.0]
        assert only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes)), "price_shock") is None

    def test_severity_grows_with_the_z_score(self):
        # calm() ends on its +0.5% bar (100.5), so 104.0 is a +3.5% day:
        # over the absolute floor and about three sigma on a +/-1% tape.
        low = only(anom.detect("NVDA", AS_OF, ohlcv=bars(*(calm() + [104.0]))),
                   "price_shock")
        high = only(anom.detect("NVDA", AS_OF, ohlcv=bars(*(calm() + [112.0]))),
                    "price_shock")
        assert low is not None and high["severity"] > low["severity"]


class TestVolumeShock:
    def test_a_session_far_above_the_median_fires(self):
        closes = calm(n=30)
        vol = [1_000_000] * 29 + [4_000_000]
        hit = only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes, volume=vol)),
                   "volume_shock")
        assert hit is not None and hit["evidence_key"] == "ohlcv"
        assert "4,000,000 shares" in hit["facts"][0]
        assert "4.0x" in hit["facts"][1]

    def test_the_median_ignores_one_earlier_blow_off_day(self):
        closes = calm(n=30)
        vol = [1_000_000] * 20 + [30_000_000] + [1_000_000] * 8 + [4_000_000]
        assert only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes, volume=vol)),
                    "volume_shock") is not None

    def test_a_thin_name_does_not_fire_on_one_block(self):
        closes = calm(n=30)
        vol = [10_000] * 29 + [50_000]          # 5x, but under the share floor
        assert only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes, volume=vol)),
                    "volume_shock") is None

    def test_a_close_only_frame_is_silent_not_an_error(self):
        assert only(anom.detect("NVDA", AS_OF, ohlcv=bars(*calm(n=30))),
                    "volume_shock") is None

    def test_absorbed_flow_is_named_as_a_hand_off(self):
        closes = calm(n=30)
        vol = [1_000_000] * 29 + [5_000_000]
        hit = only(anom.detect("NVDA", AS_OF, ohlcv=bars(*closes, volume=vol)),
                   "volume_shock")
        assert any("hand-off" in f for f in hit["facts"])


class TestOptionsFlow:
    def test_a_session_that_turned_over_the_open_interest_fires(self):
        # 3,000 traded against 2,000 standing: turnover 1.5, calls heavier.
        hit = only(anom.detect("NVDA", AS_OF,
                               options=chain(500, 2500, put_oi=1000, call_oi=1000)),
                   "options_flow")
        assert hit is not None and hit["evidence_key"] == "options"
        assert "1.50" in hit["facts"][0] and "calls" in hit["title"]

    def test_a_balanced_chain_can_still_carry_heavy_orders(self):
        # Skew is silent (pc 0.8, inside the neutral band); flow is not.
        found = anom.detect("NVDA", AS_OF,
                            options=chain(800, 1000, put_oi=600, call_oi=600))
        assert only(found, "options_skew") is None
        assert only(found, "options_flow")["title"].endswith("both sides")

    def test_ordinary_turnover_is_silent(self):
        assert only(anom.detect("NVDA", AS_OF,
                                options=chain(500, 500, put_oi=20_000, call_oi=20_000)),
                    "options_flow") is None

    def test_no_open_interest_or_a_thin_chain_produces_nothing(self):
        assert only(anom.detect("NVDA", AS_OF, options=chain(300, 100)),
                    "options_flow") is None
        assert only(anom.detect("NVDA", AS_OF,
                                options=chain(100, 100, put_oi=10, call_oi=10)),
                    "options_flow") is None


class TestTapeScreen:
    def test_enough_bars_count_as_a_screen_of_the_tape(self):
        assert "the price and volume tape" in anom.screened({}, ohlcv=bars(*calm()))
        assert "the price and volume tape" not in anom.screened({}, ohlcv=bars(*calm(n=5)))
        assert "the price and volume tape" not in anom.screened({}, ohlcv=None)


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
        # Including the per-side dates the cluster window is read from: the
        # summary aggregates the whole window, so without them the summary
        # shape would silently stop firing where the row shape fires.
        spread = [insider_row(f"Exec {i}", "D", shares=500, days_ago=d)
                  for i, d in enumerate((60, 100, 150))]
        assert anom.detect("NVDA", AS_OF, insiders=spread) == []
        assert anom.detect("NVDA", AS_OF,
                           insiders=summary(spread, monkeypatch)) == []

    def test_a_cluster_spread_over_the_whole_window_is_a_calendar(self):
        """Three people who each filed once, months apart, are not acting
        together. Counting them over the store's 180-day read window fired
        on 88 of 104 symbol-days in the local store, so no symbol was ever
        quiet and every one claimed a research question."""
        rows = [insider_row(f"Exec {i}", "D", shares=500, days_ago=d)
                for i, d in enumerate((60, 100, 150))]
        assert anom.detect("NVDA", AS_OF, insiders=rows) == []

    def test_an_unpriced_cluster_is_a_grant_day_not_a_decision(self):
        """The most common shape in the store: three directors taking the
        same annual award on the same day. The vendor prices those at 0 and
        av_store stores no value, which is what says they are paperwork."""
        rows = [insider_row(f"Director {i}", "A", shares=1000, price=0.0,
                            days_ago=5, title="Director") for i in range(3)]
        assert all(r["value_usd"] is None for r in rows)
        assert anom.detect("NVDA", AS_OF, insiders=rows) == []
        # The same three, paid for, are an open-market purchase and do fire.
        bought = [insider_row(f"Director {i}", "A", shares=1000, price=90.0,
                              days_ago=5, title="Director") for i in range(3)]
        a = only(anom.detect("NVDA", AS_OF, insiders=bought), "insider_cluster")
        assert a is not None and "priced acquisitions" in a["facts"][0]

    def test_a_grant_last_week_does_not_date_a_sale_from_may(self, monkeypatch):
        """The window test and the price test have to be asked of the SAME
        filing. Three executives whose only priced disposal is months old,
        each with an unpriced row inside the window, have sold nothing in the
        last 30 days; reading the recency off all rows and the money off the
        priced ones reported the May sales as a cluster today, and the $9M
        row would have cleared the single-disposal floor the same way."""
        rows = []
        for i in range(3):
            rows.append(insider_row(f"Exec {i}", "D", shares=50_000,
                                    price=180.0, days_ago=100))
            rows.append(insider_row(f"Exec {i}", "D", shares=1000, price=0.0,
                                    days_ago=5, title="EVP"))
        assert anom.detect("NVDA", AS_OF, insiders=rows) == []
        assert anom.detect("NVDA", AS_OF,
                           insiders=summary(rows, monkeypatch)) == []

    def test_the_filings_line_counts_only_the_priced_rows(self, monkeypatch):
        """The count sits beside a dollar total that excludes unpriced rows;
        counting them here made the two halves of one sentence disagree."""
        rows = []
        for i in range(3):
            rows.append(insider_row(f"Exec {i}", "D", shares=500, price=100.0,
                                    days_ago=5))
            rows.append(insider_row(f"Exec {i}", "D", shares=900, price=0.0,
                                    days_ago=6))
        for shape in (rows, summary(rows, monkeypatch)):
            a = only(anom.detect("NVDA", AS_OF, insiders=shape),
                     "insider_cluster")
            assert a is not None
            assert "3 priced filings from them" in a["facts"][3]
            assert "$150K" in a["facts"][3]

    def test_a_cluster_must_beat_the_symbols_own_rate(self):
        """Three officers selling in a month is not news at a symbol where
        eight sold over the preceding five. The comparison is to the symbol's
        own trailing rate, never to a flat count."""
        recent = [insider_row(f"Now {i}", "D", shares=500, days_ago=5)
                  for i in range(3)]
        busy = [insider_row(f"Then {i}", "D", shares=500, days_ago=60 + i)
                for i in range(8)]
        assert anom.detect("NVDA", AS_OF, insiders=recent + busy) == []
        # The same three against a quieter history clear it.
        quiet = busy[:4]
        a = only(anom.detect("NVDA", AS_OF, insiders=recent + quiet),
                 "insider_cluster")
        assert a is not None
        assert any("its own recent rate" in f for f in a["facts"])

    def test_a_large_disposal_months_ago_is_not_todays_story(self):
        """It stays in the 180-day window for six months; reading the largest
        row anywhere in that window kept the section alive the whole time."""
        rows = [insider_row("Jensen Huang", "D", shares=100_000, price=180.0,
                            days_ago=150, title="CEO")]
        assert anom.detect("NVDA", AS_OF, insiders=rows) == []

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
