"""Guards for the symbol lookup cache behind the Run dialog typeahead.

The rules worth pinning: a prefix hit on the symbol outranks a substring
hit on the name, an upsert never lowers a row's source (the weekly index
refresh must survive the next run that mentions the symbol), membership
tags merge rather than replace, and validate_symbol fetches only for a
name the cache does not know, then remembers the answer.
"""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import Base, ModelPrediction, StockInfo, Ticker, TradingAgentReport
from services import ticker_service as ts


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    # Test-only: lets the Postgres models build on in-memory SQLite.
    return "JSON"


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=[
        Ticker.__table__, StockInfo.__table__, ModelPrediction.__table__,
        TradingAgentReport.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


def rows():
    with ts.get_session() as session:
        return {r.symbol: r for r in session.execute(select(Ticker)).scalars()}


class TestNormalize:
    def test_vendor_spelling_and_case(self):
        assert ts.normalize_symbol(" brk.b ") == "BRK-B"
        assert ts.normalize_symbol("nvda") == "NVDA"

    def test_rejects_non_tickers(self):
        assert ts.normalize_symbol("") is None
        assert ts.normalize_symbol("NOT A TICKER") is None
        assert ts.normalize_symbol("A" * 17) is None


class TestSearch:
    @pytest.fixture(autouse=True)
    def seed(self, db):
        ts.upsert([
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
            {"symbol": "A", "name": "Agilent Technologies"},
            {"symbol": "AMD", "name": "Advanced Micro Devices"},
            {"symbol": "APLE", "name": "Apple Hospitality REIT"},
            {"symbol": "NVDA", "name": "NVIDIA"},
            {"symbol": "SNAP", "name": "Snap Inc."},
        ], "index")

    def test_symbol_prefix_outranks_name_substring(self):
        got = [r["symbol"] for r in ts.search("ap")]
        # APLE is a prefix hit, AAPL only a name hit, SNAP only a name hit.
        assert got[0] == "APLE"
        assert set(got) == {"APLE", "AAPL", "SNAP"}

    def test_shortest_symbol_first_among_prefix_hits(self):
        assert [r["symbol"] for r in ts.search("a")][:3] == ["A", "AMD", "AAPL"]

    def test_name_prefix_outranks_name_substring(self):
        ts.upsert([{"symbol": "CECO", "name": "CECO Environmental"},
                   {"symbol": "NVRI", "name": "Enviri Corp"}], "index")
        assert [r["symbol"] for r in ts.search("nvi")][0] == "NVDA"
        assert [r["symbol"] for r in ts.search("nvi", limit=2)] == ["NVDA", "CECO"]

    def test_case_insensitive_and_shape(self):
        assert ts.search("nvda") == [
            {"symbol": "NVDA", "name": "NVIDIA", "exchange": None}]
        assert ts.search("NVIDIA")[0]["symbol"] == "NVDA"

    def test_wildcards_are_literal(self):
        assert ts.search("%") == []
        assert ts.search("_") == []

    def test_empty_and_limit(self):
        assert ts.search("") == []
        assert len(ts.search("a", limit=2)) == 2


class TestUpsert:
    def test_idempotent(self, db):
        item = [{"symbol": "AAPL", "name": "Apple", "membership": ["sp500"]}]
        assert ts.upsert(item, "index") == 1
        assert ts.upsert(item, "index") == 1
        r = rows()
        assert list(r) == ["AAPL"]
        assert r["AAPL"].membership == ["sp500"]

    def test_run_never_downgrades_index(self, db):
        ts.upsert([{"symbol": "AAPL", "name": "Apple Inc.",
                    "exchange": "NASDAQ", "membership": ["sp500"]}], "index")
        ts.upsert([{"symbol": "AAPL", "name": "apple from a run"}], "run")
        r = rows()["AAPL"]
        assert r.source == "index"
        assert r.name == "Apple Inc."
        assert r.exchange == "NASDAQ"

    def test_index_upgrades_run_and_fills_name(self, db):
        ts.ensure_symbols(["aapl"])
        assert rows()["AAPL"].source == "run"
        ts.upsert([{"symbol": "AAPL", "name": "Apple Inc.",
                    "membership": ["sp500"]}], "index")
        r = rows()["AAPL"]
        assert (r.source, r.name, r.membership) == ("index", "Apple Inc.", ["sp500"])

    def test_weaker_source_fills_a_blank_name_only(self, db):
        ts.upsert([{"symbol": "X"}], "validated")
        ts.upsert([{"symbol": "X", "name": "from history"}], "run")
        assert rows()["X"].name == "from history"
        assert rows()["X"].source == "validated"

    def test_membership_merges_across_indexes(self, db):
        ts.upsert([{"symbol": "IOVA", "membership": ["r2000"]}], "index")
        ts.upsert([{"symbol": "IOVA", "membership": ["sp500"]}], "index")
        assert rows()["IOVA"].membership == ["r2000", "sp500"]

    def test_duplicates_and_junk_collapse(self, db):
        assert ts.upsert([{"symbol": "nvda"}, {"symbol": "NVDA "},
                          {"symbol": ""}, {"symbol": "bad ticker"}], "run") == 1

    def test_unknown_source_rejected(self, db):
        with pytest.raises(ValueError):
            ts.upsert([{"symbol": "AAPL"}], "wikipedia")


class TestEnsureSymbols:
    def test_bare_rows_with_run_source(self, db):
        assert ts.ensure_symbols(["nvda", "AMD", "nvda"]) == 2
        r = rows()
        assert {s: (r[s].source, r[s].name) for s in r} == {
            "NVDA": ("run", None), "AMD": ("run", None)}
        assert ts.get("AMD") == {"symbol": "AMD", "name": None, "exchange": None,
                                 "membership": [], "source": "run"}


class Info:
    def __init__(self, name):
        self.name = name


class TestValidate:
    def test_known_symbol_answers_without_fetch(self, db, monkeypatch):
        ts.upsert([{"symbol": "AAPL", "name": "Apple"}], "index")
        calls = []
        monkeypatch.setattr(ts, "_fetch_info", lambda s: calls.append(s))
        assert ts.validate_symbol("aapl") == {"ok": True, "symbol": "AAPL",
                                              "name": "Apple"}
        assert calls == []

    def test_unknown_symbol_fetches_once_then_caches(self, db, monkeypatch):
        calls = []
        monkeypatch.setattr(
            ts, "_fetch_info", lambda s: calls.append(s) or Info("Newco Inc"))
        assert ts.validate_symbol("newc") == {"ok": True, "symbol": "NEWC",
                                              "name": "Newco Inc"}
        assert rows()["NEWC"].source == "validated"
        assert ts.validate_symbol("NEWC")["ok"] is True
        assert calls == ["NEWC"]

    def test_fetch_failure_is_not_ok_and_not_cached(self, db, monkeypatch):
        def boom(s):
            raise ValueError(f"Failed to get info for {s}")
        monkeypatch.setattr(ts, "_fetch_info", boom)
        out = ts.validate_symbol("ZZZZ")
        assert out["ok"] is False and out["symbol"] == "ZZZZ"
        assert rows() == {}

    def test_garbage_never_fetches(self, db, monkeypatch):
        monkeypatch.setattr(ts, "_fetch_info", lambda s: pytest.fail("fetched"))
        assert ts.validate_symbol("not a ticker")["ok"] is False


IWM_SAMPLE = """iShares Russell 2000 ETF
Fund Holdings as of,"Sep 01, 2026"
Shares Outstanding,"272,100,000.00"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Price,Location,Exchange,Currency,FX Rate,Market Currency,Accrual Date
"XTSLA","BLK CSH FND TREASURY SL AGENCY","Cash and/or Derivatives","Money Market","1","0.34","1","1","1.00","United States","-","USD","1.00","USD","-"
"MOGA","MOOG INC CLASS A","Industrials","Equity","1","0.34","1","1","365.03","United States","NYSE","USD","1.00","USD","-"
"UMBF","UMB FINANCIAL","Financials","Equity","1","0.33","1","1","137.95","United States","NASDAQ","USD","1.00","USD","-"
"-","ARCELLX INC CVR","Health Care","Equity","1","0.00","1","1","0.07","United States","NO MARKET (E.G. UNLISTED)","USD","1.00","USD","-"
"CAD","CAD CASH","Cash and/or Derivatives","Cash","0.01","0.00","0.01","0.00","71.93","Canada","-","USD","1.39","CAD","-"
"RTYU6","RUSSELL 2000 EMINI CME SEP 26","Cash and/or Derivatives","Futures","0.00","0.00","1","1","2924.70","-","Chicago Mercantile Exchange","USD","1.00","USD","-"
"""


class TestSeeders:
    def test_iwm_parser_keeps_listed_equities_only(self):
        got = ts.parse_iwm_holdings(IWM_SAMPLE)
        assert [r["symbol"] for r in got] == ["MOGA", "UMBF"]
        assert got[0] == {"symbol": "MOGA", "name": "MOOG INC CLASS A",
                          "exchange": "NYSE", "membership": ["r2000"]}

    def test_iwm_parser_rejects_a_page_instead_of_a_csv(self):
        with pytest.raises(ValueError):
            ts.parse_iwm_holdings("<!DOCTYPE html><html></html>")

    def test_seed_from_indexes_survives_a_dead_source(self, db, monkeypatch):
        def down():
            raise ConnectionError("offline")
        monkeypatch.setattr(ts, "fetch_sp500", down)
        monkeypatch.setattr(ts, "fetch_r2000",
                            lambda: ts.parse_iwm_holdings(IWM_SAMPLE))
        assert ts.seed_from_indexes() == 2
        assert rows()["UMBF"].source == "index"
        monkeypatch.setattr(ts, "fetch_r2000", down)
        assert ts.seed_from_indexes() == 0

    def test_seed_from_history_unions_tables_and_takes_names(self, db):
        from datetime import date
        with ts.get_session() as session:
            session.add(StockInfo(symbol="AAPL", name="Apple Inc."))
            session.add(ModelPrediction(
                id="AMD_kronos_mini_20260901", symbol="AMD",
                model_name="kronos_mini", prediction_date=date(2026, 9, 1),
                target_date=date(2026, 9, 2), decision="BUY"))
            session.add(TradingAgentReport(
                id="r1", symbol="NVDA", trade_date=date(2026, 9, 1),
                decision="HOLD"))
        assert ts.seed_from_history() == 3
        r = rows()
        assert {s: (r[s].source, r[s].name) for s in r} == {
            "AAPL": ("run", "Apple Inc."), "AMD": ("run", None),
            "NVDA": ("run", None)}


class TestSchedulerRegistration:
    def test_weekly_kind_is_declared_and_builds_a_command(self):
        from services import scheduler_service as ss

        jt = ss.JOB_TYPES["ticker_refresh"]
        assert jt.verb == "ticker-refresh" and jt.needs_symbols is False
        cmd = ss._build_command({"kind": "ticker_refresh", "params": {}})
        assert cmd[-2:] == ["ticker-refresh", "--json"]
        weekly = [j for j in ss.DEFAULT_JOBS if j["kind"] == "ticker_refresh"]
        assert weekly and ss._weekday_set(weekly[0]["days_of_week"]) == {"sun"}
