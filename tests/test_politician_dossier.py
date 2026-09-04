"""What the congressional dossier must get right about people.

The block names individuals and describes their offices, which is a licence
to be confidently wrong in ways a count of buys and sells never could be.
The rules pinned here:

* a name matched in an article is matched longest-alias-first, on word
  boundaries, and an alias two members share identifies neither;
* a name with no title beside it and no disclosed trade in the symbol is a
  namesake until proved otherwise, and is not named at all;
* tenure is the current unbroken stint in the current chamber, so a gap in
  service is never counted as time in office;
* a member whose term had not begun on the as-of date is not described as
  sitting, and the roster does carry future terms;
* the cap is three members, and what the cap dropped is named in the block
  and logged rather than silently disappearing;
* point-in-time survives the block layer: a filing dated after the as-of is
  not in the dossier, however recent the underlying trade;
* a vendor throttle produces a recorded gap, not a crash, and a
  half-failure that still renders a dossier says so inside the block.

In-memory SQLite, HTTP stubbed, no network.
"""

import logging
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import (
    AvFetchLog,
    Base,
    CongressTrade,
    Politician,
    PoliticianAlias,
)
from services import av_store, politician_dossier as pd_
from services.alpha_vantage import AlphaVantageUnavailable
from services.evidence_contract import EvidenceLedger

pytestmark = pytest.mark.filterwarnings(
    r"ignore:.*does \*not\* support Decimal.*")

TABLES = [CongressTrade.__table__, Politician.__table__,
          PoliticianAlias.__table__, AvFetchLog.__table__]

AS_OF = "2026-08-20"


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=TABLES)
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


def trade(bioguide, name, ttype, tx, filed, symbol="NVDA", party="R",
          chamber="HOUSE", state="WA04", amount=(15001, 50000)):
    return {"bioguide_id": bioguide, "politician_canonical": name,
            "politician": name, "symbol": symbol, "party": party,
            "chamber": chamber, "state": state[:2], "state_district": state,
            "asset_name": f"{symbol} common", "asset_type_code": "ST",
            "transaction_type": ttype, "transaction_date": tx,
            "filed_date": filed, "amount_min": f"{amount[0]}.00",
            "amount_max": f"{amount[1]}.00", "owner_code": "SELF",
            "filing_status": "NEW"}


NVDA_TRADES = {"symbol": "NVDA", "trades": [
    trade("N000189", "Dan Newhouse", "BUY", "2026-07-10", "2026-07-17"),
    trade("N000189", "Dan Newhouse", "SELL", "2026-06-01", "2026-06-10"),
    # Filed after the as-of: traded before it, public after it.
    trade("L000601", "Sam T. Liccardo", "SELL", "2026-07-21", "2026-08-27",
          party="D", state="CA16"),
    trade("W000802", "Sheldon Whitehouse", "BUY", "2026-06-30", "2026-07-08",
          party="D", chamber="SENATE", state="RI"),
    trade("M000100", "Mary Miller", "BUY", "2026-05-20", "2026-06-01",
          state="IL15"),
    trade("M000200", "Mary Miller Meeks", "SELL", "2026-05-10", "2026-05-20",
          state="IA01"),
]}

# The same member's other tape, which is what "what else are they trading"
# reads. Seven rows so the per-member cap has something to drop.
MSFT_TRADES = {"symbol": "MSFT", "trades": [
    trade("N000189", "Dan Newhouse", "BUY", f"2026-0{m}-05",
          f"2026-0{m}-15", symbol="MSFT") for m in range(1, 8)
]}

ROSTER = {"politicians": [
    {"bioguide_id": "N000189", "display_name": "Dan Newhouse",
     "aliases": ["dan newhouse", "newhouse dan"], "chamber": "HOUSE",
     "state": "WA", "district": "04", "party": "R",
     "terms": [{"chamber": "HOUSE", "state": "WA", "district": "04",
                "start_date": "2015-01-06", "end_date": "2027-01-03"}]},
    {"bioguide_id": "W000802", "display_name": "Sheldon Whitehouse",
     "aliases": ["sheldon whitehouse"], "chamber": "SENATE", "state": "RI",
     "district": None, "party": "D",
     "terms": [{"chamber": "SENATE", "state": "RI", "district": None,
                "start_date": "2007-01-04", "end_date": "2031-01-03"}]},
    {"bioguide_id": "L000601", "display_name": "Sam T. Liccardo",
     "aliases": ["sam liccardo"], "chamber": "HOUSE", "state": "CA",
     "district": "16", "party": "D",
     "terms": [{"chamber": "HOUSE", "state": "CA", "district": "16",
                "start_date": "2025-01-03", "end_date": "2027-01-03"}]},
    # A constructed overlap: one member's whole alias is the opening of the
    # other's. Longest-first is what keeps the shorter one from claiming it.
    {"bioguide_id": "M000100", "display_name": "Mary Miller",
     "aliases": ["mary miller"], "chamber": "HOUSE", "state": "IL",
     "district": "15", "party": "R",
     "terms": [{"chamber": "HOUSE", "state": "IL", "district": "15",
                "start_date": "2021-01-03", "end_date": "2027-01-03"}]},
    {"bioguide_id": "M000200", "display_name": "Mary Miller Meeks",
     "aliases": ["mary miller meeks"], "chamber": "HOUSE", "state": "IA",
     "district": "01", "party": "R",
     "terms": [{"chamber": "HOUSE", "state": "IA", "district": "01",
                "start_date": "2021-01-03", "end_date": "2027-01-03"}]},
    # Elected, sworn in next January: the roster knows about the term now.
    {"bioguide_id": "E000999", "display_name": "Elect Newcomer",
     "aliases": ["elect newcomer"], "chamber": "HOUSE", "state": "TX",
     "district": "05", "party": "D",
     "terms": [{"chamber": "HOUSE", "state": "TX", "district": "05",
                "start_date": "2027-01-03", "end_date": "2029-01-03"}]},
    # Service with a real break in it, and a chamber change across it: the
    # roster shape that turned "in the Senate since 1993" into a fact.
    {"bioguide_id": "C000127", "display_name": "Gap Cantwell",
     "aliases": ["gap cantwell"], "chamber": "SENATE", "state": "WA",
     "district": None, "party": "D",
     "terms": [{"chamber": "HOUSE", "state": "WA", "district": "01",
                "start_date": "1993-01-05", "end_date": "1995-01-03"},
               {"chamber": "SENATE", "state": "WA", "district": None,
                "start_date": "2001-01-03", "end_date": "2007-01-03"},
               {"chamber": "SENATE", "state": "WA", "district": None,
                "start_date": "2007-01-04", "end_date": "2031-01-03"}]},
    # Two spells in the same chamber four years apart.
    {"bioguide_id": "C001123", "display_name": "Gap Cisneros",
     "aliases": ["gap cisneros"], "chamber": "HOUSE", "state": "CA",
     "district": "31", "party": "D",
     "terms": [{"chamber": "HOUSE", "state": "CA", "district": "39",
                "start_date": "2019-01-03", "end_date": "2021-01-03"},
               {"chamber": "HOUSE", "state": "CA", "district": "31",
                "start_date": "2025-01-03", "end_date": "2027-01-03"}]},
    # A sitting member who shares a full name with a hedge fund manager.
    {"bioguide_id": "C001068", "display_name": "Steve Cohen",
     "aliases": ["steve cohen"], "chamber": "HOUSE", "state": "TN",
     "district": "09", "party": "D",
     "terms": [{"chamber": "HOUSE", "state": "TN", "district": "09",
                "start_date": "2007-01-04", "end_date": "2027-01-03"}]},
    # Two members whose short variant is identical.
    {"bioguide_id": "S000001", "display_name": "John Smith",
     "aliases": ["john smith", "j. smith"], "chamber": "HOUSE",
     "state": "TX", "district": "01", "party": "R", "terms": []},
    {"bioguide_id": "S000002", "display_name": "Jane Smith",
     "aliases": ["jane smith", "j. smith"], "chamber": "HOUSE",
     "state": "NY", "district": "02", "party": "D", "terms": []},
]}


def article(title, summary="", published="2026-08-18"):
    return {"title": title, "summary": summary, "published_at": published}


def stub_fetch(monkeypatch, error=None, payloads=None, congress=None):
    payloads = payloads or {av_store.ROSTER_FUNCTION: ROSTER,
                            av_store.CONGRESS_FUNCTION: NVDA_TRADES}
    congress = congress or {"NVDA": NVDA_TRADES, "MSFT": MSFT_TRADES}
    calls: list[tuple] = []

    def _fetch(function, **params):
        calls.append((function, params.get("symbol")))
        if error:
            raise error
        if function == av_store.CONGRESS_FUNCTION:
            return congress[params["symbol"]]
        return payloads[function]

    monkeypatch.setattr(av_store, "fetch", _fetch)
    return calls


@pytest.fixture
def seeded(db, monkeypatch):
    stub_fetch(monkeypatch)
    av_store.sync_politicians()
    av_store.sync_congress_trades("NVDA")
    av_store.sync_congress_trades("MSFT")
    return db


class TestNewsMatching:
    def test_a_member_named_only_in_the_news_gets_a_dossier(self, seeded):
        news = [article("Rep.-elect Elect Newcomer calls for a review of "
                        "NVIDIA export licences",
                        "The incoming member wants hearings.")]
        d = pd_.build_dossier("ZZZZ", AS_OF, news=news)
        block = pd_.format_dossier_block(d)

        assert d["trades_in_window"] == 0
        assert [e["bioguide_id"] for e in d["entries"]] == ["E000999"]
        assert "Elect Newcomer" in block
        assert "no disclosed trade in the window" in block
        assert "Named in this run's news" in block

    def test_the_longest_alias_claims_the_name(self, seeded):
        hits = pd_.politicians_in_news(
            [article("Rep. Mary Miller Meeks introduces the chips measure")])
        assert [h["bioguide_id"] for h in hits] == ["M000200"]

    def test_an_alias_two_members_share_identifies_neither(self, seeded):
        assert pd_.politicians_in_news(
            [article("Rep. J. Smith says the bill is dead")]) == []

    def test_a_bare_surname_is_not_an_identification(self, seeded):
        assert pd_.politicians_in_news(
            [article("Rep. Newhouse pressed the agency on subsidies")]) == []
        # The full name in the summary still matches.
        hits = pd_.politicians_in_news(
            [article("Chip subsidies questioned",
                     "Rep. Dan Newhouse pressed the agency.")])
        assert [h["bioguide_id"] for h in hits] == ["N000189"]

    def test_a_name_inside_a_longer_word_is_not_a_match(self, seeded):
        assert pd_.politicians_in_news(
            [article("Rep. Newhousedan Corp files for an IPO")]) == []

    def test_ordinary_prose_does_not_fabricate_a_member(self, db, monkeypatch):
        # The vendor stores initial forms, so the live roster carries "a
        # green" and "a gray". Matched against article text they turn "gave
        # the deal a green light" into a sitting member of Congress named in
        # this run's news, which the report then states as fact.
        stub_fetch(monkeypatch, payloads={av_store.ROSTER_FUNCTION: {
            "politicians": [
                {"bioguide_id": "G000553", "display_name": "Al Green",
                 "aliases": ["al green", "a green"], "chamber": "HOUSE",
                 "state": "TX", "district": "09", "party": "D", "terms": []},
                {"bioguide_id": "G000600", "display_name": "Adam Gray",
                 "aliases": ["adam gray", "a gray"], "chamber": "HOUSE",
                 "state": "CA", "district": "13", "party": "D", "terms": []},
            ]}})
        av_store.sync_politicians()

        prose = [article("Regulators give the HPE-Juniper deal a green light",
                         "The stock is in a gray area on valuation.")]
        assert pd_.politicians_in_news(prose) == []
        block = pd_.format_dossier_block(
            pd_.build_dossier("HPE", AS_OF, news=prose))
        assert "Al Green" not in block and "Adam Gray" not in block
        # The real name still matches when an article actually uses it.
        hits = pd_.politicians_in_news(
            [article("Rep. Al Green questions the deal")])
        assert [h["bioguide_id"] for h in hits] == ["G000553"]


class TestNamesakes:
    """A full name is not an identification, and this block asserts identity
    as fact. Steve Cohen sits for TN-9 and also runs Point72."""

    NAMESAKE = [article(
        "Smart money is buying Netflix",
        "Steve Cohen and Chris Bloomstran initiated new positions.")]

    def test_a_name_with_no_title_and_no_trade_is_not_attributed(self, seeded):
        assert pd_.politicians_in_news(self.NAMESAKE) == []
        block = pd_.format_dossier_block(
            pd_.build_dossier("NFLX", AS_OF, news=self.NAMESAKE))
        assert "Steve Cohen" not in block
        assert "TN-9" not in block

    def test_a_title_next_to_the_name_is_the_corroboration(self, seeded):
        hits = pd_.politicians_in_news([article(
            "Netflix hearing", "Rep. Steve Cohen asked about the merger.")])
        assert [h["bioguide_id"] for h in hits] == ["C001068"]
        assert hits[0]["titled"] == "rep"

    def test_a_title_cannot_vouch_for_a_name_in_another_article(self, seeded):
        news = [article("Rep. Mary Miller Meeks opens the hearing"),
                article("Smart money", "Steve Cohen took a stake.")]
        assert [h["bioguide_id"] for h in pd_.politicians_in_news(news)] \
            == ["M000200"]

    def test_the_members_own_filing_corroborates_the_name(self, seeded):
        # Newhouse discloses NVDA trades, so an article naming him without a
        # title is still him; the block says that is why.
        news = [article("Chips hearing set", "Dan Newhouse will chair it.")]
        d = pd_.build_dossier("NVDA", AS_OF, news=news)
        block = pd_.format_dossier_block(d)
        assert d["entries"][0]["bioguide_id"] == "N000189"
        assert ("with no title next to the name, matched to them because "
                "they filed a trade in it") in block


class TestTermDates:
    def test_a_member_elect_is_never_described_as_sitting(self, seeded):
        profile = av_store.politician("E000999")
        sentence = pd_.describe_tenure(profile, AS_OF)
        assert "not yet in office" in sentence
        assert "sitting" not in sentence

        block = pd_.format_dossier_block(pd_.build_dossier(
            "ZZZZ", AS_OF,
            news=[article("Rep.-elect Elect Newcomer on chip policy")]))
        assert "not yet in office as of 2026-08-20" in block
        assert "sitting" not in block

        # After the swearing-in the same row reads as current.
        assert "sitting HOUSE member" in pd_.describe_tenure(profile,
                                                             "2027-06-01")

    def test_a_sitting_member_says_since_and_how_long(self, seeded):
        sentence = pd_.describe_tenure(av_store.politician("N000189"), AS_OF)
        assert sentence.startswith("sitting HOUSE member, in office since "
                                   "2015-01-06")
        assert "(11 yr)" in sentence

    def test_a_member_who_has_left_is_past_tense(self, seeded):
        sentence = pd_.describe_tenure(av_store.politician("N000189"),
                                       "2028-06-01")
        assert sentence.startswith("former member: served 2015-01-06 to "
                                   "2027-01-03")

    def test_a_break_in_service_is_not_counted_as_tenure(self, seeded):
        # Two spells in the House, four years apart. The first start is not
        # when this member has been in office since.
        sentence = pd_.describe_tenure(av_store.politician("C001123"), AS_OF)
        assert sentence == ("sitting HOUSE member, in office since "
                            "2025-01-03 (1 yr)")

    def test_a_chamber_change_is_dated_from_the_chamber(self, seeded):
        # House 1993-1995, then the Senate from 2001: neither "senator since
        # 1993" nor "in office since 1993" is true.
        sentence = pd_.describe_tenure(av_store.politician("C000127"), AS_OF)
        assert sentence == ("sitting SENATE member, in office since "
                            "2001-01-03 (25 yr)")

    def test_consecutive_terms_are_one_continuous_stint(self, seeded):
        # The 2007-01-03 / 2007-01-04 seam between two Senate terms is a
        # hand-over, not a break, and must not restart the clock.
        profile = av_store.politician("C000127")
        assert "2001-01-03" in pd_.describe_tenure(profile, "2010-06-01")

    def test_a_former_member_reports_only_the_last_stint(self, seeded):
        sentence = pd_.describe_tenure(av_store.politician("C001123"),
                                       "2028-06-01")
        assert sentence.startswith("former member: served 2025-01-03 to "
                                   "2027-01-03")
        assert "earlier, non-consecutive service from 2019-01-03" in sentence

    def test_service_that_began_in_the_other_chamber_is_a_footnote(self, seeded):
        profile = dict(av_store.politician("C000127"))
        # Same member, but without the 1990s gap: House then Senate, no break.
        profile["terms"] = [
            {"chamber": "HOUSE", "start_date": "1999-01-06",
             "end_date": "2013-01-03"},
            {"chamber": "SENATE", "start_date": "2013-01-03",
             "end_date": "2031-01-03"}]
        sentence = pd_.describe_tenure(profile, AS_OF)
        assert sentence.startswith("sitting SENATE member, in office since "
                                   "2013-01-03 (13 yr)")
        assert "in Congress without a break since 1999-01-06" in sentence

    def test_a_roster_with_no_terms_says_so(self, seeded):
        sentence = pd_.describe_tenure(av_store.politician("S000001"), AS_OF)
        assert sentence == "HOUSE member; term dates unavailable"


class TestDossier:
    NEWS = [article("Rep.-elect Elect Newcomer calls for an NVIDIA export "
                    "review"),
            article("Chips hearing set", "Dan Newhouse will chair it."),
            article("Mary Miller Meeks introduces the measure")]

    def test_it_ranks_news_plus_trades_first_and_caps_at_three(
            self, seeded, caplog):
        with caplog.at_level(logging.INFO, logger="services.politician_dossier"):
            d = pd_.build_dossier("NVDA", AS_OF, news=self.NEWS)
        block = pd_.format_dossier_block(d)

        # Named AND trading, newest trade first; then the untrumpeted trader.
        assert [e["bioguide_id"] for e in d["entries"]] == [
            "N000189", "M000200", "W000802"]
        assert len(d["dropped"]) == 2
        assert "Mary Miller (1 trade(s) in NVDA)" in d["dropped"]
        assert "Elect Newcomer (named in the news, no disclosed trade here)" \
               in d["dropped"]
        assert "2 further member(s) over the cap of 3 are not shown" in block
        assert "not shown" in caplog.text and "Mary Miller" in caplog.text

    def test_the_order_sentence_matches_the_order_of_the_entries(
            self, seeded):
        """A member named only in the news sorts LAST, behind everyone who
        traded the symbol. The header sentence claimed the opposite, and the
        header is what tells the model how to weigh the list."""
        news = [article("Rep.-elect Elect Newcomer calls for a review of "
                        "NVIDIA export licences",
                        "The incoming member wants hearings.")]
        d = pd_.build_dossier("NVDA", AS_OF, news=news, days=60)
        block = pd_.format_dossier_block(d)

        assert [e["bioguide_id"] for e in d["entries"]] == [
            "N000189", "W000802", "E000999"]
        assert not d["dropped"]
        assert "ranked by news mention first" not in block
        assert "then members named only in the news." in block
        assert block.index("Dan Newhouse") < block.index("Elect Newcomer")

    def test_it_shows_the_trade_lines_and_the_bands(self, seeded):
        block = pd_.format_dossier_block(
            pd_.build_dossier("NVDA", AS_OF, news=self.NEWS))
        assert "Dan Newhouse (R, HOUSE, WA-04)" in block
        assert "In NVDA: 2 disclosed trade(s) in the window, 1 buy / 1 sell" \
               in block
        assert "2026-07-10 BUY $15K–$50K, public from 2026-07-17" in block

    def test_other_trades_exclude_this_symbol_and_are_capped(self, seeded):
        d = pd_.build_dossier("NVDA", AS_OF, news=self.NEWS)
        newhouse = d["entries"][0]
        assert {t["symbol"] for t in newhouse["elsewhere"]} == {"MSFT"}
        assert len(newhouse["elsewhere"]) == pd_.MAX_OTHER_TRADES
        assert newhouse["elsewhere_hidden"] >= 1
        block = pd_.format_dossier_block(d)
        assert "Also disclosed lately, among the other symbols this system " \
               "syncs" in block
        assert "further trade(s) in the fetched range not shown" in block

    def test_an_empty_cross_symbol_view_is_stated_as_coverage(self, seeded):
        # Whitehouse traded only NVDA in the seed, so the store holds nothing
        # else for him. That is what this system synced, not what he traded,
        # and the block must not let the report conclude the second.
        d = pd_.build_dossier("NVDA", "2026-08-28", news=[])
        block = pd_.format_dossier_block(d)
        assert "No other trade by this member is in the store." in block
        assert "not evidence they traded nothing else" in block
        assert "bounded by the symbols this system syncs" in block

    def test_point_in_time_holds_through_the_block_layer(self, seeded):
        assert "Liccardo" not in pd_.format_dossier_block(
            pd_.build_dossier("NVDA", AS_OF))
        assert "Liccardo" in pd_.format_dossier_block(
            pd_.build_dossier("NVDA", "2026-08-28"))

    def test_an_empty_dossier_states_the_absence(self, seeded):
        block = pd_.format_dossier_block(pd_.build_dossier("ZZZZ", AS_OF))
        assert "No member of Congress disclosed a trade in ZZZZ" in block
        assert "named a member of the roster" in block


class TestBlockSize:
    """The caller slices every context block to a fixed width. Anything the
    block puts after its caveats is what a slice takes off first, so the
    over-cap list is bounded and the caveats come before it."""

    def test_a_crowded_window_stays_inside_the_callers_block_cap(
            self, db, monkeypatch):
        from models.trading_agents_model import PER_BLOCK_CHARS

        many = {"symbol": "NVDA", "trades": [
            trade(f"X{i:06d}", f"Member Number {i}", "BUY",
                  f"2026-0{1 + i % 8}-1{i % 9}",
                  f"2026-0{1 + i % 8}-2{i % 8}")
            for i in range(30)]}
        stub_fetch(monkeypatch, payloads={av_store.ROSTER_FUNCTION: ROSTER},
                   congress={"NVDA": many})
        av_store.sync_politicians()
        av_store.sync_congress_trades("NVDA")

        d = pd_.build_dossier("NVDA", "2026-09-01")
        block = pd_.format_dossier_block(d)
        assert len(d["dropped"]) > pd_.MAX_DROPPED_NAMED
        assert len(block) < PER_BLOCK_CHARS
        # The caveats survive because they are written before the over-cap
        # list, which is the line a truncation would eat.
        assert (block.index("not evidence of foreknowledge")
                < block.index("over the cap of"))
        assert f"and {len(d['dropped']) - pd_.MAX_DROPPED_NAMED} more" in block


class TestVendorFailure:
    def test_an_unavailable_vendor_is_a_gap_not_a_crash(self, db, monkeypatch):
        stub_fetch(monkeypatch, error=AlphaVantageUnavailable("rate limit"))
        block, problems = pd_.politician_block("NVDA", AS_OF,
                                               news=[article("Anything")])
        assert block == ""
        assert len(problems) == 2
        assert all("rate limit" in p for p in problems)

        ledger = EvidenceLedger("NVDA")
        ledger.missing("politicians", "; ".join(problems))
        assert [g.block for g in ledger.gaps] == ["politicians"]
        assert ledger.gaps[0].severity == "optional"
        assert not ledger.degraded

    def test_stored_rows_still_answer_when_the_top_up_fails(self, seeded,
                                                            monkeypatch):
        stub_fetch(monkeypatch, error=AlphaVantageUnavailable("throttled"))
        block, problems = pd_.politician_block("NVDA", AS_OF)
        assert problems == []
        assert "Dan Newhouse" in block

    def test_the_block_spends_no_call_when_the_store_is_fresh(self, seeded,
                                                              monkeypatch):
        calls = stub_fetch(monkeypatch)
        pd_.politician_block("NVDA", AS_OF)
        assert calls == []

    def test_a_stale_source_is_stated_in_the_block_it_rendered(
            self, seeded, monkeypatch):
        """The roster top-up failing while the trades still answer is a
        PARTIAL failure: the dossier is written and goes into the prompt, so
        the caveat has to be in the text the model reads. The caller records
        a rendered block as present, and nothing else states this."""
        with seeded.get_session() as session:
            for row in session.execute(select(AvFetchLog)).scalars():
                row.last_fetched_at = (datetime.now(timezone.utc)
                                       - timedelta(days=60))
        stub_fetch(monkeypatch, error=AlphaVantageUnavailable("throttled"))

        block, problems = pd_.politician_block("NVDA", AS_OF)
        assert [p for p in problems if p.startswith("roster:")]
        assert "Dan Newhouse" in block
        assert "Not every source behind this block refreshed on this run" \
               in block
        assert "throttled" in block.rsplit("\n", 1)[-1]

    def test_a_transport_error_leaves_the_run_on_the_stored_rows(
            self, seeded, monkeypatch):
        # The rate limiter raises RateLimitTimeout and requests raises
        # RequestException; neither is an AlphaVantageUnavailable, and the
        # top-up sits above rows that can still answer.
        with seeded.get_session() as session:
            for row in session.execute(select(AvFetchLog)).scalars():
                row.last_fetched_at = (datetime.now(timezone.utc)
                                       - timedelta(days=60))
        stub_fetch(monkeypatch, error=RuntimeError("transport blew up"))

        block, problems = pd_.politician_block(
            "NVDA", AS_OF, news=[article("Dan Newhouse on chip subsidies")])
        assert problems == []
        assert "Dan Newhouse" in block
