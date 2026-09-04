"""Political/institutional blocks are point-in-time on the FILING date and
say what they are (lagged positioning), never a signal.

The congressional half reads ``services.av_store`` now, not the endpoint: the
vendor hands back the symbol's whole history on every call, so a run tops the
symbol up at most weekly and filters locally. These tests therefore run
against an in-memory SQLite store with the fetch stubbed, and one of them
pins the thing that change bought: a second run spends no call.
"""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import AvFetchLog, Base, CongressTrade
from services import av_store
from services import political_service as ps

pytestmark = pytest.mark.filterwarnings(
    r"ignore:.*does \*not\* support Decimal.*")

TABLES = [CongressTrade.__table__, AvFetchLog.__table__]


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


@pytest.fixture(autouse=True)
def _clear():
    ps._CACHE.clear()
    yield
    ps._CACHE.clear()


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=TABLES)
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


CONGRESS = {"trades": [
    # Filed AFTER as_of: invisible on 2026-07-20 even though traded before.
    {"chamber": "HOUSE", "politician": "Hon. Sam T. Liccardo",
     "politician_canonical": "Sam T. Liccardo", "bioguide_id": "L000601",
     "party": "D", "state_district": "CA16", "transaction_type": "SELL",
     "transaction_date": "2026-07-21", "filed_date": "2026-07-27",
     "amount_min": "15001.00", "amount_max": "50000.00", "owner_code": "SELF"},
    {"chamber": "HOUSE", "politician": "Hon. Dan Newhouse",
     "politician_canonical": "Dan Newhouse", "bioguide_id": "N000189",
     "party": "R", "state_district": "WA04", "transaction_type": "SELL",
     "transaction_date": "2026-07-10", "filed_date": "2026-07-17",
     "amount_min": "1001.00", "amount_max": "15000.00", "owner_code": "SPOUSE"},
    # Senate row without filed_date: notification_date is the visibility.
    {"chamber": "SENATE", "politician": "Sheldon Whitehouse",
     "politician_canonical": "Sheldon Whitehouse", "bioguide_id": "W000802",
     "party": "D", "state": "RI", "transaction_type": "BUY",
     "transaction_date": "2026-06-30", "notification_date": "2026-07-08",
     "amount_min": "15001.00", "amount_max": "50000.00", "owner_code": "SELF"},
    # Too old for the 180-day window.
    {"chamber": "HOUSE", "politician": "Old Trade", "bioguide_id": "O000001",
     "politician_canonical": "Old Trade", "party": "R",
     "transaction_type": "BUY", "transaction_date": "2025-01-01",
     "filed_date": "2025-01-10", "amount_min": "1001.00",
     "amount_max": "15000.00"},
]}


def stub_fetch(monkeypatch, payload=CONGRESS, error=None):
    """Every vendor call this module can make, counted."""
    calls: list[tuple] = []

    def _fetch(function, **params):
        calls.append((function, params.get("symbol")))
        if error:
            raise error
        return payload

    monkeypatch.setattr(av_store, "fetch", _fetch)
    return calls


def test_congress_trades_are_visible_only_from_filing_date(db, monkeypatch):
    stub_fetch(monkeypatch)
    c = ps.get_congress_trades("NVDA", "2026-07-20")
    names = [t["politician"] for t in c["trades"]]
    assert "Sam T. Liccardo" not in " ".join(names)      # filed 07-27
    assert "Dan Newhouse" in " ".join(names)              # filed 07-17
    assert "Sheldon Whitehouse" in " ".join(names)        # notified 07-08
    assert "Old Trade" not in " ".join(names)
    assert c["n"] == 2 and c["buys"] == 1 and c["sells"] == 1
    assert c["by_party"] == {"D": {"buys": 1, "sells": 0}, "R": {"buys": 0, "sells": 1}}
    # Everything ever filed and public by the as-of date, window or not.
    assert c["total_disclosed_visible"] == 3
    block = ps.format_congress_block("NVDA", c)
    assert "visible through 2026-07-20" in block
    assert "2026-07-10 SELL $1K–$15K: Dan Newhouse (R-WA04, HOUSE), spouse; filed 2026-07-17" in block
    assert "never as a timing signal" in block


def test_a_second_run_reads_the_store_and_spends_no_call(db, monkeypatch):
    calls = stub_fetch(monkeypatch)
    ps.get_congress_trades("NVDA", "2026-07-20")
    ps._CACHE.clear()                       # a different process, same store
    again = ps.get_congress_trades("NVDA", "2026-07-21")
    assert len(calls) == 1
    assert again["n"] == 2


def test_no_trades_is_stated_not_silent(db, monkeypatch):
    stub_fetch(monkeypatch, payload={"trades": []})
    c = ps.get_congress_trades("BHF", "2026-09-01")
    block = ps.format_congress_block("BHF", c)
    assert "None disclosed in the window" in block
    assert "0 on record and public by 2026-09-01" in block
    assert "Absence is uninformative" in block


def test_the_dossier_supersedes_the_congressional_half(db, monkeypatch):
    stub_fetch(monkeypatch)
    with patch.object(ps, "get_institutional_holdings",
                      return_value=None):
        blocks, problems = ps.political_blocks("NVDA", "2026-07-20",
                                               include_congress=False)
    assert blocks == [] and problems == []
    with patch.object(ps, "get_institutional_holdings", return_value=None):
        blocks, _ = ps.political_blocks("NVDA", "2026-07-20")
    assert len(blocks) == 1 and "congressional trades" in blocks[0]


def test_an_empty_store_and_a_throttled_top_up_is_a_gap_not_no_trades(
        db, monkeypatch):
    stub_fetch(monkeypatch, error=ps.AlphaVantageUnavailable("rate limit"))
    with pytest.raises(ps.AlphaVantageUnavailable, match="rate limit"):
        ps.get_congress_trades("BHF", "2026-09-01")

    with patch.object(ps, "get_institutional_holdings",
                      side_effect=ps.AlphaVantageUnavailable("throttled")):
        blocks, problems = ps.political_blocks("BHF", "2026-09-01")
    assert blocks == []
    assert len(problems) == 2 and "rate limit" in problems[0]


def test_stored_rows_survive_a_throttled_top_up(db, monkeypatch):
    stub_fetch(monkeypatch)
    ps.get_congress_trades("NVDA", "2026-07-20")
    ps._CACHE.clear()
    # A week later the top-up fails; the block is still written from what
    # the store holds rather than reporting no disclosures.
    monkeypatch.setattr(av_store, "MAX_AGE_DAYS",
                        {**av_store.MAX_AGE_DAYS, av_store.CONGRESS_FUNCTION: 0})
    stub_fetch(monkeypatch, error=ps.AlphaVantageUnavailable("rate limit"))
    c = ps.get_congress_trades("NVDA", "2026-07-20")
    assert c["n"] == 2


def test_the_shared_client_raises_on_an_http_200_throttle_body():
    # The HTTP-200-with-a-message body lives in one place; the 13F block,
    # which still fetches per run, reaches it through alpha_vantage.fetch.
    from types import SimpleNamespace

    from services import alpha_vantage as av

    fake_api = SimpleNamespace(ALPHA_VANTAGE_API_KEY="k", ALPHA_VANTAGE_BASE_URL="u",
                               DEFAULT_TIMEOUT=1)
    with patch.object(av.requests, "get") as get, \
         patch.object(av, "API", fake_api), \
         patch.object(av, "alpha_vantage_bucket"):
        get.return_value.json.return_value = {"Note": "Thank you for using Alpha Vantage! rate limit"}
        get.return_value.raise_for_status.return_value = None
        with pytest.raises(ps.AlphaVantageUnavailable, match="rate limit"):
            ps.get_institutional_holdings("BHF", "2026-09-01")


INST = {
    "total_institutional_holders": "541", "holders_with_increased_holdings": "194",
    "holders_with_decreased_holdings": "180", "total_institutional_ownership_percentage": "101%",
    "holdings": [
        {"holder_name": "VANGUARD GROUP INC", "shares_held": "5410582", "shares_changed": "-126853",
         "shares_changed_percentage": "-2.29%", "last_reported": "2025-12-31"},
        {"holder_name": "BLACKROCK", "shares_held": "5393709", "shares_changed": "120769",
         "shares_changed_percentage": "2.29%", "last_reported": "2026-06-30"},
        # Reported after as_of: not visible.
        {"holder_name": "FUTURE FUND", "shares_held": "9999999", "shares_changed": "9999999",
         "shares_changed_percentage": "99%", "last_reported": "2026-09-30"},
    ],
}


def test_institutional_rows_are_filtered_to_reports_on_or_before_as_of():
    with patch.object(ps, "_av_get", return_value=INST):
        h = ps.get_institutional_holdings("BHF", "2026-09-01")
    assert h["holders_visible"] == 2
    assert h["latest_report"] == "2026-06-30"
    assert [r["holder"] for r in h["top_holders"]] == ["VANGUARD GROUP INC", "BLACKROCK"]
    block = ps.format_institutional_block("BHF", h)
    assert "FUTURE FUND" not in block
    assert "Biggest adds: BLACKROCK +121K (2.29%)" in block
    assert "Biggest cuts: VANGUARD GROUP INC -127K (-2.29%)" in block
    assert "not a signal about the next session" in block
