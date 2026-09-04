"""What the insider (Form 4) evidence block may and may not say.

The block exists because a reader cannot get "who inside the company is
selling" off a price chart. The failure modes it has to avoid are all about
claiming more than the filing supports:

* a grant, gift or option exercise arrives priced at 0.0, and calling that an
  acquisition worth $0 tells a reader the executive bought for nothing;
* dollar totals must be summed over priced rows only, and must say how many
  rows they cover;
* visibility is the SEC filing DEADLINE, a proxy, and the block has to name
  it as one or the model reads the transaction-to-visible gap as timing;
* a vendor throttle is a recorded gap, never a crash and never a silent
  "no insider activity".

In-memory SQLite, HTTP stubbed, no network.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import AvFetchLog, Base, InsiderTransaction
from services import av_store, insider_service
from services.alpha_vantage import AlphaVantageUnavailable
from services.evidence_contract import EvidenceLedger

pytestmark = pytest.mark.filterwarnings(
    r"ignore:.*does \*not\* support Decimal.*")

TABLES = [InsiderTransaction.__table__, AvFetchLog.__table__]


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


# 2026: Aug 10-12 and Aug 31-Sep 2 are all NYSE sessions, so the two-trading-
# day Form 4 deadline lands on Aug 12 and Sep 2 respectively.
PAYLOAD = {"data": [
    {"transaction_date": "2026-08-10", "ticker": "NVDA",
     "executive": "Jensen Huang", "executive_title": "President and CEO",
     "security_type": "Common Stock", "acquisition_or_disposal": "D",
     "shares": "100000", "share_price": "180.50"},
    # A vesting grant: the vendor prices it at zero.
    {"transaction_date": "2026-08-10", "ticker": "NVDA",
     "executive": "Colette Kress", "executive_title": "EVP and CFO",
     "security_type": "Restricted Stock Units",
     "acquisition_or_disposal": "A", "shares": "25000", "share_price": "0.0"},
    {"transaction_date": "2026-07-15", "ticker": "NVDA",
     "executive": "Colette Kress", "executive_title": "EVP and CFO",
     "security_type": "Common Stock", "acquisition_or_disposal": "D",
     "shares": "5000", "share_price": "170.00"},
    # Transacted late in the window: not visible until its Sep 2 deadline.
    {"transaction_date": "2026-08-31", "ticker": "NVDA",
     "executive": "Debora Shoquist", "executive_title": "EVP Operations",
     "security_type": "Common Stock", "acquisition_or_disposal": "D",
     "shares": "8000", "share_price": "175.00"},
]}


def stub_fetch(monkeypatch, payload=PAYLOAD, error=None):
    calls: list[str] = []

    def _fetch(function, **params):
        calls.append(function)
        if error:
            raise error
        return payload

    monkeypatch.setattr(av_store, "fetch", _fetch)
    return calls


def seeded(db, monkeypatch):
    stub_fetch(monkeypatch)
    av_store.sync_insider_transactions("NVDA")


class TestFormatter:
    def test_it_names_executives_titles_shares_and_real_dollars(
            self, db, monkeypatch):
        seeded(db, monkeypatch)
        s = insider_service.summarize_insiders("NVDA", "2026-08-20")
        block = insider_service.format_insider_block("NVDA", s)

        assert s["n"] == 3 and s["executives"] == 2
        assert "3 filings by 2 named executives" in block
        assert "1 acquisition, 2 disposals" in block
        # 100,000 @ 180.50 plus 5,000 @ 170.00 = $18.90M, over 2 priced rows,
        # while the share counts cover all three.
        assert "Shares moved, all 3 rows: 25,000 sh acquired vs 105,000 sh " \
               "disposed." in block
        assert "over the 2 row(s) carrying a real share price: $0 acquired, " \
               "$18.90M disposed." in block
        assert "Jensen Huang (President and CEO)" in block
        assert "Colette Kress (EVP and CFO)" in block

    def test_a_zero_price_row_is_described_as_what_it_is(self, db, monkeypatch):
        seeded(db, monkeypatch)
        block = insider_service.format_insider_block(
            "NVDA", insider_service.summarize_insiders("NVDA", "2026-08-20"))

        assert "1 of 3 rows report a share price of 0.00" in block
        assert "grants, awards, gifts or option exercises" in block
        # The acquisition moved 25,000 shares and no money anyone can name.
        assert "$0 acquired" in block
        assert "share price reported as 0.00" in block
        assert "$0.00" not in block

    def test_the_visibility_proxy_is_stated(self, db, monkeypatch):
        seeded(db, monkeypatch)
        block = insider_service.format_insider_block(
            "NVDA", insider_service.summarize_insiders("NVDA", "2026-08-20"))
        assert "PROXY" in block
        assert "second trading day after the transaction" in block

    def test_an_empty_window_says_so_rather_than_vanishing(self, db,
                                                           monkeypatch):
        seeded(db, monkeypatch)
        s = insider_service.summarize_insiders("NVDA", "2026-08-20", days=3)
        block = insider_service.format_insider_block("NVDA", s)
        assert s["n"] == 0
        assert "No Form 4 lines were visible in this window" in block
        assert "PROXY" in block


class TestPointInTime:
    def test_the_filing_deadline_gates_the_block(self, db, monkeypatch):
        seeded(db, monkeypatch)
        early = insider_service.summarize_insiders("NVDA", "2026-08-11")
        assert [r["executive"] for r in early["recent"]] == ["Colette Kress"]

        # Aug 12 is the deadline for the Aug 10 transactions.
        on_deadline = insider_service.summarize_insiders("NVDA", "2026-08-12")
        assert on_deadline["n"] == 3

        # The Aug 31 sale is invisible until Sep 2 whatever the block asks.
        assert "Debora Shoquist" not in insider_service.format_insider_block(
            "NVDA", insider_service.summarize_insiders("NVDA", "2026-09-01"))
        assert "Debora Shoquist" in insider_service.format_insider_block(
            "NVDA", insider_service.summarize_insiders("NVDA", "2026-09-02"))

    def test_the_block_layer_requires_an_as_of(self, db, monkeypatch):
        seeded(db, monkeypatch)
        with pytest.raises(ValueError):
            insider_service.summarize_insiders("NVDA", None)


class TestVendorFailure:
    def test_an_unavailable_vendor_is_a_gap_not_a_crash(self, db, monkeypatch):
        calls = stub_fetch(monkeypatch,
                           error=AlphaVantageUnavailable("rate limit"))
        block, problems = insider_service.insider_block("NVDA", "2026-08-20")

        assert calls == [av_store.INSIDER_FUNCTION]
        assert block == ""
        assert problems and "rate limit" in problems[0]

        ledger = EvidenceLedger("NVDA")
        ledger.missing("insiders", "; ".join(problems))
        assert [g.block for g in ledger.gaps] == ["insiders"]
        assert ledger.gaps[0].severity == "optional"
        assert not ledger.degraded

    def test_stored_rows_still_answer_when_the_top_up_fails(self, db,
                                                            monkeypatch):
        seeded(db, monkeypatch)
        # Age the fetch log past the max so the next block tries to top up.
        with db.get_session() as session:
            row = session.get(AvFetchLog,
                              (av_store.INSIDER_FUNCTION, "NVDA"))
            row.last_fetched_at = datetime.now(timezone.utc) - timedelta(days=30)
        stub_fetch(monkeypatch, error=AlphaVantageUnavailable("throttled"))

        block, problems = insider_service.insider_block("NVDA", "2026-08-20")
        assert problems == []
        assert "Jensen Huang" in block
        assert "the rows above are as stored" in block

    def test_a_failed_top_up_does_not_date_the_store_to_today(self, db,
                                                              monkeypatch):
        # The header is the first thing the model reads and section 5 tells
        # the report to quote dates from it. Stamped from the last ATTEMPT it
        # would read "store synced <today>" directly above a line saying the
        # top-up failed and the rows are last month's.
        seeded(db, monkeypatch)
        synced = av_store.last_synced_date(av_store.INSIDER_FUNCTION, "NVDA")
        with db.get_session() as session:
            row = session.get(AvFetchLog,
                              (av_store.INSIDER_FUNCTION, "NVDA"))
            row.last_fetched_at = datetime.now(timezone.utc) - timedelta(days=30)
            row.last_success_at = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
        stub_fetch(monkeypatch, error=AlphaVantageUnavailable("throttled"))

        block, _ = insider_service.insider_block("NVDA", "2026-08-20")
        assert "store synced 2026-08-01" in block
        assert f"store synced {synced}" not in block

    def test_a_transport_error_leaves_the_run_on_the_stored_rows(
            self, db, monkeypatch):
        # Not every failure below this block is an AlphaVantageUnavailable:
        # the rate limiter raises RateLimitTimeout and requests raises
        # RequestException. Neither may cost the run a block it can write.
        seeded(db, monkeypatch)
        with db.get_session() as session:
            session.get(AvFetchLog, (av_store.INSIDER_FUNCTION, "NVDA")) \
                .last_fetched_at = datetime.now(timezone.utc) - timedelta(days=30)
        stub_fetch(monkeypatch, error=RuntimeError("transport blew up"))

        block, problems = insider_service.insider_block("NVDA", "2026-08-20")
        assert problems == []
        assert "Jensen Huang" in block
        assert "transport blew up" in block


class TestCallCost:
    def test_a_second_block_inside_the_max_age_spends_no_call(self, db,
                                                              monkeypatch):
        calls = stub_fetch(monkeypatch)
        first, _ = insider_service.insider_block("NVDA", "2026-08-20")
        second, _ = insider_service.insider_block("NVDA", "2026-08-20")
        assert len(calls) == 1
        assert "Jensen Huang" in first and first == second
