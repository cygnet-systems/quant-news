"""What the stored Alpha Vantage filings must never do.

The one that matters is point-in-time. Both sources describe trades that
happened weeks before anyone could read about them, so a row served on the
wrong date is lookahead dressed as evidence, and this repo has shipped that
bug before. The rules pinned here:

* a congressional trade is invisible until it is FILED, and a row the vendor
  gave neither a filing nor a notification date is never served at all;
* a Form 4 is invisible until its filing deadline, the second trading day
  after the transaction, because the payload carries no filing date;
* re-fetching a symbol writes nothing new (both endpoints return full
  history on every call, so this happens every week);
* a zero share price yields NULL, not a fabricated dollar value, and a
  value that is stored is stored at the scale of its column;
* a fetch inside its max age spends no call.

Everything runs on in-memory SQLite with the HTTP layer stubbed. No test
here touches the network.
"""

import importlib.util
import threading
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import (
    AvFetchLog,
    Base,
    CongressTrade,
    InsiderTransaction,
    JobRun,
    Politician,
    PoliticianAlias,
    ScheduledJob,
)
from services import av_store

# SQLite has no native Decimal; the conversion warning is expected here and
# says nothing about the code under test.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:.*does \*not\* support Decimal.*")

TABLES = [Politician.__table__, PoliticianAlias.__table__,
          CongressTrade.__table__, InsiderTransaction.__table__,
          AvFetchLog.__table__, ScheduledJob.__table__, JobRun.__table__]

MIGRATION = (Path(__file__).resolve().parents[1]
             / "db" / "migrations" / "versions" / "017_av_intelligence.py")


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    # Test-only: lets the Postgres models build on in-memory SQLite.
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


@pytest.fixture
def threaded_db(tmp_path, monkeypatch):
    """A file-backed database for the concurrency test. An in-memory SQLite
    engine hands every thread its own empty database, which would hide the
    race rather than run it."""
    import db.session as dbs

    eng = create_engine(f"sqlite:///{tmp_path / 'av.db'}")
    Base.metadata.create_all(eng, tables=TABLES)
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


def count(db, model) -> int:
    with db.get_session() as session:
        return session.execute(select(func.count()).select_from(model)).scalar()


def _unavailable(function, **params):
    raise av_store.AlphaVantageUnavailable("Note: rate limit")


def stub_fetch(monkeypatch, payloads: dict):
    """Replace the HTTP client with a canned payload per function, and count
    the calls so a test can prove one was not spent."""
    calls: list[str] = []

    def _fetch(function, **params):
        calls.append(function)
        return payloads[function]

    monkeypatch.setattr(av_store, "fetch", _fetch)
    return calls


# 2026: Aug 31 Mon, Sep 1 Tue, Sep 2 Wed are all NYSE sessions, so the
# 2-trading-day Form 4 deadline for an Aug-31 transaction is Sep 2.
CONGRESS_PAYLOAD = {"symbol": "NVDA", "trades": [
    {"chamber": "HOUSE", "politician": "Hon. Dan Newhouse",
     "politician_canonical": "Dan Newhouse", "bioguide_id": "N000189",
     "party": "R", "state": "WA", "state_district": "WA04", "symbol": "NVDA",
     "asset_name": "NVIDIA Corporation", "asset_type_code": "ST",
     "transaction_type": "SELL", "transaction_date": "2026-07-10",
     "notification_date": "2026-07-16", "filed_date": "2026-07-17",
     "amount_min": "1001.00", "amount_max": "15000.00",
     "owner_code": "SPOUSE", "filing_status": "NEW"},
    # Filed after the as_of below: traded before it, public after it.
    {"chamber": "HOUSE", "politician": "Sam T. Liccardo",
     "politician_canonical": "Sam T. Liccardo", "bioguide_id": "L000601",
     "party": "D", "state": "CA", "state_district": "CA16", "symbol": "NVDA",
     "transaction_type": "SELL", "transaction_date": "2026-07-21",
     "filed_date": "2026-07-27", "amount_min": "15001.00",
     "amount_max": "50000.00", "owner_code": "SELF"},
    # No filed_date: the notification date is the visibility.
    {"chamber": "SENATE", "politician": "Sheldon Whitehouse",
     "politician_canonical": "Sheldon Whitehouse", "bioguide_id": "W000802",
     "party": "D", "state": "RI", "symbol": "NVDA",
     "transaction_type": "BUY", "transaction_date": "2026-06-30",
     "notification_date": "2026-07-08", "amount_min": "15001.00",
     "amount_max": "50000.00", "owner_code": "SELF"},
    # Neither date: cannot be placed in time.
    {"chamber": "HOUSE", "politician": "Undated Filer", "bioguide_id": "U000001",
     "party": "R", "symbol": "NVDA", "transaction_type": "BUY",
     "transaction_date": "2026-07-01", "amount_min": "1001.00",
     "amount_max": "15000.00", "owner_code": "SELF"},
]}

INSIDER_PAYLOAD = {"data": [
    {"transaction_date": "2026-08-31", "ticker": "NVDA",
     "executive": "Jensen Huang", "executive_title": "CEO",
     "security_type": "Common Stock", "acquisition_or_disposal": "D",
     "shares": "100000", "share_price": "180.50"},
    # A grant: price 0, so no dollar value exists to store.
    {"transaction_date": "2026-08-31", "ticker": "NVDA",
     "executive": "Colette Kress", "executive_title": "CFO",
     "security_type": "Restricted Stock Units",
     "acquisition_or_disposal": "A", "shares": "25000", "share_price": "0.0"},
    # A 10b5-1 sale of a fractional lot: shares x price runs to six decimals,
    # which is more than the value column holds. The two rows above are both
    # exact at two decimals and cannot catch a writer that stores the
    # unrounded product.
    {"transaction_date": "2026-08-31", "ticker": "NVDA",
     "executive": "Debora Shoquist", "executive_title": "EVP Operations",
     "security_type": "Common Stock", "acquisition_or_disposal": "D",
     "shares": "1350.7315", "share_price": "182.4433"},
]}

ROSTER_PAYLOAD = {"politicians": [
    {"bioguide_id": "N000189", "display_name": "Dan Newhouse",
     "aliases": ["dan newhouse", "daniel newhouse", "newhouse, dan"],
     "chamber": "HOUSE", "state": "WA", "district": "04", "party": "R",
     "terms": [{"chamber": "HOUSE", "state": "WA", "district": "04",
                "start_date": "2015-01-06", "end_date": "2027-01-03"}]},
    {"bioguide_id": "W000802", "display_name": "Sheldon Whitehouse",
     "aliases": ["sheldon whitehouse"], "chamber": "SENATE", "state": "RI",
     "district": None, "party": "D", "terms": []},
    # Two members sharing a variant: the alias cannot identify either.
    {"bioguide_id": "S000001", "display_name": "John Smith",
     "aliases": ["john smith", "j. smith"], "chamber": "HOUSE",
     "state": "TX", "district": "01", "party": "R", "terms": []},
    {"bioguide_id": "S000002", "display_name": "Jane Smith",
     "aliases": ["jane smith", "j. smith"], "chamber": "HOUSE",
     "state": "NY", "district": "02", "party": "D", "terms": []},
]}


class TestCongressPointInTime:
    def test_a_trade_is_invisible_until_it_is_filed(self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        av_store.sync_congress_trades("NVDA")

        visible = av_store.congress_trades_for("NVDA", "2026-07-20")
        assert [t["politician"] for t in visible] == ["Dan Newhouse",
                                                      "Sheldon Whitehouse"]
        # Same rows, a week later: the 07-27 filing is public now.
        later = av_store.congress_trades_for("NVDA", "2026-07-28")
        assert "Sam T. Liccardo" in [t["politician"] for t in later]

    def test_a_row_with_no_filing_or_notification_is_never_served(
            self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        av_store.sync_congress_trades("NVDA")
        with db.get_session() as session:
            stored = session.execute(
                select(CongressTrade)
                .where(CongressTrade.bioguide_id == "U000001")).scalar_one()
            assert stored.visible_from is None
        for as_of in ("2026-07-02", "2026-07-28", "2030-01-01"):
            served = av_store.congress_trades_for("NVDA", as_of, days=3650)
            assert "Undated Filer" not in [t["politician"] for t in served]

    def test_the_window_is_measured_on_the_transaction_date(self, db,
                                                            monkeypatch):
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        av_store.sync_congress_trades("NVDA")
        # 20 days back from 2026-07-28 excludes the 06-30 purchase, which is
        # visible (notified 07-08) but outside the transaction window.
        recent = av_store.congress_trades_for("NVDA", "2026-07-28", days=20)
        assert "Sheldon Whitehouse" not in [t["politician"] for t in recent]

    def test_visibility_precedence_matches_the_live_endpoint(self):
        # political_service applies filed_date, then notification_date, to
        # the same payload. The two must not drift apart.
        assert av_store.congress_visible_from(
            date(2026, 7, 17), date(2026, 7, 16)) == date(2026, 7, 17)
        assert av_store.congress_visible_from(
            None, date(2026, 7, 8)) == date(2026, 7, 8)
        assert av_store.congress_visible_from(None, None) is None

    def test_by_politician_crosses_symbols(self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        av_store.sync_congress_trades("NVDA")
        other = {"trades": [dict(CONGRESS_PAYLOAD["trades"][0],
                                 symbol="AMD", transaction_date="2026-07-12",
                                 filed_date="2026-07-18")]}
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: other})
        av_store.sync_congress_trades("AMD")

        rows = av_store.congress_trades_by_politician("N000189", "2026-07-20")
        assert [r["symbol"] for r in rows] == ["AMD", "NVDA"]
        # Still point-in-time: nothing filed after as_of.
        assert av_store.congress_trades_by_politician(
            "N000189", "2026-07-17") == [
            r for r in rows if r["symbol"] == "NVDA"]


class TestInsiderPointInTime:
    def test_a_form4_is_invisible_inside_its_two_day_deadline(self, db,
                                                              monkeypatch):
        stub_fetch(monkeypatch, {av_store.INSIDER_FUNCTION: INSIDER_PAYLOAD})
        av_store.sync_insider_transactions("NVDA")

        # Transacted 2026-08-31 (Mon); deadline is 2026-09-02 (Wed).
        assert av_store.insider_transactions_for("NVDA", "2026-08-31") == []
        assert av_store.insider_transactions_for("NVDA", "2026-09-01") == []
        served = av_store.insider_transactions_for("NVDA", "2026-09-02")
        assert {r["executive"] for r in served} == {"Jensen Huang",
                                                    "Colette Kress",
                                                    "Debora Shoquist"}

    def test_the_deadline_is_two_trading_days_not_two_calendar_days(self):
        # Friday 2026-09-04 -> Monday 09-07 is Labor Day -> Tue 08, Wed 09.
        assert av_store.insider_visible_from(date(2026, 9, 4)) == date(2026, 9, 9)

    def test_a_zero_share_price_stores_no_dollar_value(self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.INSIDER_FUNCTION: INSIDER_PAYLOAD})
        av_store.sync_insider_transactions("NVDA")
        served = {r["executive"]: r
                  for r in av_store.insider_transactions_for("NVDA", "2026-09-02")}
        assert served["Colette Kress"]["share_price"] == 0.0
        assert served["Colette Kress"]["value_usd"] is None
        assert served["Jensen Huang"]["value_usd"] == 100000 * 180.50

    def test_a_dollar_value_is_stored_at_the_scale_of_its_column(
            self, db, monkeypatch):
        """1350.7315 x 182.4433 = 246,431.91227395, which Numeric(24, 2)
        cannot hold. Rounding has to happen where the row is built, or the
        value read back never equals the value computed."""
        stub_fetch(monkeypatch, {av_store.INSIDER_FUNCTION: INSIDER_PAYLOAD})
        av_store.sync_insider_transactions("NVDA")
        served = {r["executive"]: r
                  for r in av_store.insider_transactions_for("NVDA", "2026-09-02")}
        assert served["Debora Shoquist"]["value_usd"] == pytest.approx(
            round(1350.7315 * 182.4433, 2))

        # Asserted on the row the writer builds, not on the row read back:
        # every database rounds on the way in, so a value read back is
        # always at the column's scale and cannot show what was sent.
        built = av_store._insider_rows("NVDA", INSIDER_PAYLOAD["data"])
        values = [v["value_usd"] for v in built.values()
                  if v["executive"] == "Debora Shoquist"]
        assert values == [Decimal("246431.91")]

    def test_a_row_without_a_transaction_date_is_dropped(self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.INSIDER_FUNCTION: {"data": [
            {"ticker": "NVDA", "executive": "No Date", "shares": "1",
             "share_price": "1"}]}})
        assert av_store.sync_insider_transactions("NVDA") == 0
        assert count(db, InsiderTransaction) == 0


class TestUpsertIdempotence:
    def test_resyncing_the_same_congress_payload_writes_nothing(
            self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        assert av_store.sync_congress_trades("NVDA") == 4
        assert count(db, CongressTrade) == 4
        assert av_store.sync_congress_trades("NVDA") == 0
        assert count(db, CongressTrade) == 4

    def test_resyncing_the_same_insider_payload_writes_nothing(
            self, db, monkeypatch):
        """Both endpoints return full history on every call, so this runs
        weekly per symbol. A value the writer computes at more precision
        than the column holds is rewritten on every one of those syncs."""
        stub_fetch(monkeypatch, {av_store.INSIDER_FUNCTION: INSIDER_PAYLOAD})
        assert av_store.sync_insider_transactions("NVDA") == 3
        assert av_store.sync_insider_transactions("NVDA") == 0
        assert count(db, InsiderTransaction) == 3

    def test_a_late_filing_date_updates_the_row_in_place(self, db, monkeypatch):
        """The vendor fills filed_date in later on some Senate rows; that is
        the one revision that changes when a stored trade becomes visible."""
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        av_store.sync_congress_trades("NVDA")
        patched = {"trades": [
            dict(t, filing_status="AMENDED") if t["bioguide_id"] == "W000802"
            else t for t in CONGRESS_PAYLOAD["trades"]]}
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: patched})
        assert av_store.sync_congress_trades("NVDA") == 1
        assert count(db, CongressTrade) == 4

    def test_a_repeated_line_in_one_payload_is_one_row(self, db, monkeypatch):
        doubled = {"trades": CONGRESS_PAYLOAD["trades"]
                   + [CONGRESS_PAYLOAD["trades"][0]]}
        stub_fetch(monkeypatch, {av_store.CONGRESS_FUNCTION: doubled})
        assert av_store.sync_congress_trades("NVDA") == 4
        assert count(db, CongressTrade) == 4


class TestEnsureFresh:
    def _log(self, db, function, subject, age_days, ok=True):
        with db.get_session() as session:
            session.add(AvFetchLog(
                function=function, subject=subject, ok=ok, rows_seen=1,
                last_fetched_at=datetime.now(timezone.utc)
                - timedelta(days=age_days)))

    def test_a_recent_fetch_spends_no_call(self, db, monkeypatch):
        calls = stub_fetch(monkeypatch,
                           {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        self._log(db, av_store.CONGRESS_FUNCTION, "NVDA", age_days=1)
        result = av_store.ensure_fresh(av_store.CONGRESS_FUNCTION, "NVDA")
        assert result["fetched"] is False and calls == []

    def test_a_stale_fetch_is_refreshed(self, db, monkeypatch):
        calls = stub_fetch(monkeypatch,
                           {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        self._log(db, av_store.CONGRESS_FUNCTION, "NVDA", age_days=30)
        result = av_store.ensure_fresh(av_store.CONGRESS_FUNCTION, "NVDA")
        assert result["fetched"] is True and result["rows"] == 4
        assert calls == [av_store.CONGRESS_FUNCTION]

    def test_a_failed_attempt_is_not_freshness(self, db, monkeypatch):
        calls = stub_fetch(monkeypatch,
                           {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        self._log(db, av_store.CONGRESS_FUNCTION, "NVDA", age_days=0, ok=False)
        assert av_store.ensure_fresh(
            av_store.CONGRESS_FUNCTION, "NVDA")["fetched"] is True
        assert calls == [av_store.CONGRESS_FUNCTION]

    def test_a_throttle_is_reported_not_raised_and_is_retried_next_time(
            self, db, monkeypatch):
        def _throttled(function, **params):
            raise av_store.AlphaVantageUnavailable("Note: rate limit")

        monkeypatch.setattr(av_store, "fetch", _throttled)
        result = av_store.ensure_fresh(av_store.CONGRESS_FUNCTION, "NVDA")
        assert result["fetched"] is False
        assert "unavailable" in result["reason"]
        stored = av_store.last_fetch(av_store.CONGRESS_FUNCTION, "NVDA")
        assert stored["ok"] is False
        calls = stub_fetch(monkeypatch,
                           {av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD})
        assert av_store.ensure_fresh(
            av_store.CONGRESS_FUNCTION, "NVDA")["fetched"] is True
        assert calls == [av_store.CONGRESS_FUNCTION]

    def test_a_transport_error_degrades_the_same_way_a_throttle_does(
            self, db, monkeypatch):
        # The rate limiter raises RateLimitTimeout and requests raises
        # RequestException; neither is an AlphaVantageUnavailable, and a
        # caller reading stored rows must not be taken down by either.
        def _blew_up(function, **params):
            raise RuntimeError("transport blew up")

        monkeypatch.setattr(av_store, "fetch", _blew_up)
        result = av_store.ensure_fresh(av_store.INSIDER_FUNCTION, "NVDA")
        assert result["fetched"] is False
        assert result["reason"] == "unavailable: transport blew up"
        assert av_store.last_fetch(av_store.INSIDER_FUNCTION,
                                   "NVDA")["ok"] is False

    def test_only_a_success_moves_the_success_stamp(self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.INSIDER_FUNCTION: INSIDER_PAYLOAD})
        av_store.ensure_fresh(av_store.INSIDER_FUNCTION, "NVDA")
        synced = av_store.last_synced_date(av_store.INSIDER_FUNCTION, "NVDA")
        assert synced

        # Age the attempt past the max so the next call tries and fails.
        with db.get_session() as session:
            session.get(AvFetchLog, (av_store.INSIDER_FUNCTION, "NVDA")) \
                .last_fetched_at = datetime.now(timezone.utc) - timedelta(days=30)
        monkeypatch.setattr(av_store, "fetch", _unavailable)
        av_store.ensure_fresh(av_store.INSIDER_FUNCTION, "NVDA")

        record = av_store.last_fetch(av_store.INSIDER_FUNCTION, "NVDA")
        assert record["ok"] is False
        assert record["last_success_at"] < record["last_fetched_at"]
        assert av_store.last_synced_date(
            av_store.INSIDER_FUNCTION, "NVDA") == synced

    def test_concurrent_symbols_spend_one_roster_call_between_them(
            self, threaded_db, monkeypatch):
        # Research runs a thread per symbol and every one of them wants the
        # roster on a cold database. Before the lock each spent its own call
        # and the losers died on the keys the winner had just inserted.
        calls: list[str] = []
        barrier = threading.Barrier(4)

        def _slow_fetch(function, **params):
            calls.append(function)
            time.sleep(0.2)
            return ROSTER_PAYLOAD

        monkeypatch.setattr(av_store, "fetch", _slow_fetch)
        results, errors = [], []

        def _worker():
            barrier.wait()
            try:
                results.append(av_store.ensure_fresh(av_store.ROSTER_FUNCTION,
                                                     av_store.ALL_SUBJECTS))
            except Exception as e:                    # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert [type(e).__name__ for e in errors] == []
        assert calls == [av_store.ROSTER_FUNCTION]
        assert sum(1 for r in results if r["fetched"]) == 1
        assert count(threaded_db, Politician) == 4


class TestRoster:
    def test_a_payload_with_no_members_is_a_failure_not_an_empty_roster(
            self, db, monkeypatch):
        # The envelope key is the vendor's to rename. Reported as a success,
        # an unparseable roster would leave alias_index() empty week after
        # week while the refresh job mailed "0 members, all good".
        monkeypatch.setattr(av_store, "fetch",
                            lambda function, **params: {"roster": []})
        with pytest.raises(av_store.AlphaVantageUnavailable):
            av_store.sync_politicians()

        result = av_store.ensure_fresh(av_store.ROSTER_FUNCTION,
                                       av_store.ALL_SUBJECTS)
        assert result["reason"].startswith("unavailable")
        assert av_store.last_fetch(av_store.ROSTER_FUNCTION,
                                   av_store.ALL_SUBJECTS)["ok"] is False
        summary = av_store.refresh_all([])
        assert summary["failed"] == 1
        assert summary["problems"] == [f"roster: {result['reason']}"]

    def test_an_initial_and_a_surname_is_not_a_matchable_name(self, db,
                                                              monkeypatch):
        # The vendor stores initial forms and the live roster holds 69 that
        # start with the bare article "a". Scanned over prose, "a green" in
        # "gave the deal a green light" names a sitting member of Congress.
        stub_fetch(monkeypatch, {av_store.ROSTER_FUNCTION: {"politicians": [
            {"bioguide_id": "G000553", "display_name": "Al Green",
             "aliases": ["al green", "a green", "green"], "chamber": "HOUSE",
             "state": "TX", "district": "09", "party": "D", "terms": []}]}})
        av_store.sync_politicians()

        index = av_store.alias_index()
        assert index == {"al green": "G000553"}
        assert av_store.matchable_alias("a green") is False
        assert av_store.matchable_alias("green") is False
        assert av_store.matchable_alias("al green") is True
        # Dropped from prose matching, still resolvable when a filing row
        # hands the name over directly.
        assert av_store.politician_by_alias("A. Green")["bioguide_id"] \
               == "G000553"


    def test_alias_matching_finds_the_member(self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.ROSTER_FUNCTION: ROSTER_PAYLOAD})
        assert av_store.sync_politicians() == 4

        for spelling in ("Dan Newhouse", "Hon. Dan Newhouse", "DANIEL NEWHOUSE",
                         "  newhouse, dan  "):
            found = av_store.politician_by_alias(spelling)
            assert found and found["bioguide_id"] == "N000189", spelling
        assert av_store.politician_by_alias("Nobody Here") is None
        assert av_store.politician("W000802")["chamber"] == "SENATE"

    def test_an_alias_two_members_share_identifies_neither(self, db,
                                                           monkeypatch):
        stub_fetch(monkeypatch, {av_store.ROSTER_FUNCTION: ROSTER_PAYLOAD})
        av_store.sync_politicians()
        assert av_store.politician_by_alias("J. Smith") is None
        assert av_store.politician_by_alias("Jane Smith")["bioguide_id"] == "S000002"

    def test_resyncing_the_roster_leaves_the_alias_table_alone(self, db,
                                                               monkeypatch):
        stub_fetch(monkeypatch, {av_store.ROSTER_FUNCTION: ROSTER_PAYLOAD})
        av_store.sync_politicians()
        before = count(db, PoliticianAlias)
        av_store.sync_politicians()
        assert count(db, PoliticianAlias) == before == 8

    def test_a_dropped_alias_is_removed(self, db, monkeypatch):
        stub_fetch(monkeypatch, {av_store.ROSTER_FUNCTION: ROSTER_PAYLOAD})
        av_store.sync_politicians()
        trimmed = {"politicians": [dict(m, aliases=[m["display_name"].lower()])
                                   for m in ROSTER_PAYLOAD["politicians"]]}
        stub_fetch(monkeypatch, {av_store.ROSTER_FUNCTION: trimmed})
        av_store.sync_politicians()
        assert count(db, PoliticianAlias) == 4
        assert av_store.politician_by_alias("J. Smith") is None


class TestRefreshAll:
    def test_one_symbol_failing_does_not_stop_the_rest(self, db, monkeypatch):
        def _fetch(function, **params):
            if params.get("symbol") == "BAD":
                raise av_store.AlphaVantageUnavailable("Information: invalid")
            return {av_store.ROSTER_FUNCTION: ROSTER_PAYLOAD,
                    av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD,
                    av_store.INSIDER_FUNCTION: INSIDER_PAYLOAD}[function]

        monkeypatch.setattr(av_store, "fetch", _fetch)
        summary = av_store.refresh_all(["NVDA", "BAD"])
        assert summary["politicians"] == 4 and summary["symbols"] == 2
        assert summary["congress_rows"] == 4 and summary["insider_rows"] == 3
        assert summary["failed"] == 2 and len(summary["problems"]) == 2

    def test_a_second_refresh_spends_no_calls(self, db, monkeypatch):
        calls = stub_fetch(monkeypatch, {
            av_store.ROSTER_FUNCTION: ROSTER_PAYLOAD,
            av_store.CONGRESS_FUNCTION: CONGRESS_PAYLOAD,
            av_store.INSIDER_FUNCTION: INSIDER_PAYLOAD})
        av_store.refresh_all(["NVDA"])
        assert len(calls) == 3
        again = av_store.refresh_all(["NVDA"])
        assert len(calls) == 3 and again["skipped"] == 2

    def test_watchlist_symbols_come_from_the_analysis_jobs(self, db):
        with db.get_session() as session:
            session.add(ScheduledJob(id="daily_analysis", kind="analysis",
                                     hour=7, minute=0, symbols_csv="nvda, amd",
                                     params_json={}))
            session.add(ScheduledJob(id="daily_evaluation", kind="evaluation",
                                     hour=18, minute=0, params_json={}))
        assert av_store.watchlist_symbols() == ["NVDA", "AMD"]


class TestSchedulerRegistration:
    """The weekly job reaches both a fresh database (seed) and one that
    already has a schedule (migration 017), and the two specs agree."""

    def _run_migration(self, db, direction="upgrade"):
        spec = importlib.util.spec_from_file_location("migration_017", MIGRATION)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with db._engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                getattr(module, direction)()

    @pytest.fixture
    def bare_db(self, monkeypatch):
        """Only the scheduler tables: the migration creates the rest."""
        import db.session as dbs

        eng = create_engine("sqlite://")
        Base.metadata.create_all(
            eng, tables=[ScheduledJob.__table__, JobRun.__table__])
        monkeypatch.setattr(dbs, "_engine", eng)
        monkeypatch.setattr(
            dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
        return dbs

    def test_the_kind_is_declared_and_builds_a_command(self):
        from services import scheduler_service as ss

        jt = ss.JOB_TYPES[ss.AV_REFRESH_JOB]
        assert jt.verb == "av-refresh" and jt.needs_symbols is False
        cmd = ss._build_command({"kind": ss.AV_REFRESH_JOB, "params": {}})
        assert cmd[-2:] == ["av-refresh", "--json"]
        spec = next(j for j in ss.DEFAULT_JOBS if j["id"] == ss.AV_REFRESH_JOB)
        assert ss._weekday_set(spec["days_of_week"]) == {"sun"}

    def test_the_migration_installs_it_on_a_populated_schedule(self, bare_db):
        from services import scheduler_service as ss

        with bare_db.get_session() as session:
            session.add(ScheduledJob(id="daily_analysis", kind="analysis",
                                     hour=7, minute=0, symbols_csv="AAPL",
                                     params_json={}))
        self._run_migration(bare_db)
        with bare_db.get_session() as session:
            row = session.get(ScheduledJob, ss.AV_REFRESH_JOB)
        spec = next(j for j in ss.DEFAULT_JOBS if j["id"] == ss.AV_REFRESH_JOB)
        # The migration carries literals; they must not drift from the seed.
        for key in ("kind", "description", "hour", "minute", "days_of_week",
                    "symbols_csv", "params_json"):
            assert getattr(row, key) == spec[key], key

    def test_the_migration_leaves_an_empty_schedule_to_the_seed(self, bare_db):
        from services import scheduler_service as ss

        self._run_migration(bare_db)
        with bare_db.get_session() as session:
            assert session.get(ScheduledJob, ss.AV_REFRESH_JOB) is None

    def test_the_migration_never_resurrects_a_deleted_job(self, bare_db):
        from services import scheduler_service as ss

        with bare_db.get_session() as session:
            session.add(ScheduledJob(id="daily_analysis", kind="analysis",
                                     hour=7, minute=0, params_json={}))
            session.add(JobRun(job_id=ss.AV_REFRESH_JOB, trigger="schedule",
                               status="success"))
        self._run_migration(bare_db)
        with bare_db.get_session() as session:
            assert session.get(ScheduledJob, ss.AV_REFRESH_JOB) is None
