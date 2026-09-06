"""One adjustment basis per stored series, and per scored row.

Adjusted bars are rebased backwards at every dividend and split. The price
cache replaces only the dates a fetch covers, so without a check a fetch
made after an event writes post-event bars beside pre-event ones (measured
2026-09-06: seams up to 1.8% on off-watchlist names). And a prediction's
entry close is copied when the row is stored, so scoring it against an
exit fetched after an ex-date counted the dividend as a price drop (72 of
4,126 scored rows, mean -0.36%).
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import (Base, CacheMetadata, ModelPrediction, StockPrice,
                       StrategyEvaluation)
from services import price_basis as pb
from services.cache_service import CacheService


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


UTC = timezone.utc


def _bars(start, n, close0=100.0, step=0.0, dividends=None, factor=1.0):
    """n daily bars from start; ``dividends`` maps date -> amount; every
    close is scaled by ``factor`` (a rebased fetch)."""
    dividends = dividends or {}
    rows = []
    for i in range(n):
        d = start + timedelta(days=i)
        c = (close0 + step * i) * factor
        rows.append({"date": pd.Timestamp(d), "open": c, "high": c, "low": c,
                     "close": c, "volume": 1000, "dividends": dividends.get(d, 0.0),
                     "stock_splits": 0.0})
    return pd.DataFrame(rows)


D0 = date(2026, 1, 5)


# --- stale_history: pure decision -------------------------------------------

def test_same_basis_short_refresh_is_not_stale():
    stored = _bars(D0, 40).assign(fetched_at=datetime(2026, 2, 20, tzinfo=UTC))
    new = _bars(D0 + timedelta(days=30), 12)
    assert pb.stale_history(new, stored) is None


def test_dividend_after_stored_fetch_marks_older_rows_stale():
    stored = _bars(D0, 40).assign(fetched_at=datetime(2026, 2, 1, tzinfo=UTC))
    ex = D0 + timedelta(days=35)
    new = _bars(D0 + timedelta(days=30), 12, dividends={ex: 0.5}, factor=0.995)
    reason = pb.stale_history(new, stored)
    assert reason and str(ex) in reason


def test_dividend_recorded_in_stored_rows_counts_too():
    """The event fell in the gap between the stored newest bar and the fetch
    window: only a stored row knows about it."""
    ex = D0 + timedelta(days=20)
    stored = _bars(D0, 25, dividends={ex: 0.5})
    stored["fetched_at"] = [datetime(2026, 1, 10, tzinfo=UTC)] * 15 + \
        [datetime(2026, 2, 5, tzinfo=UTC)] * 10
    new = _bars(D0 + timedelta(days=40), 5)
    assert pb.stale_history(new, stored)


def test_event_older_than_the_fetch_is_not_stale():
    ex = D0 + timedelta(days=3)
    stored = _bars(D0, 40, dividends={ex: 0.5}).assign(
        fetched_at=datetime(2026, 3, 1, tzinfo=UTC))
    new = _bars(D0 + timedelta(days=30), 12)
    assert pb.stale_history(new, stored) is None


def test_overlap_mismatch_marks_stale_without_an_event_row():
    """A rebased fetch whose window holds no event row (Yahoo revised the
    history, or the event predates the window): the overlap disagrees."""
    stored = _bars(D0, 40).assign(fetched_at=datetime(2026, 3, 1, tzinfo=UTC))
    new = _bars(D0 + timedelta(days=30), 12, factor=0.99)
    reason = pb.stale_history(new, stored)
    assert reason and "stored close" in reason


def test_partial_newest_bar_does_not_count_as_a_basis_change():
    stored = _bars(D0, 40).assign(fetched_at=datetime(2026, 3, 1, tzinfo=UTC))
    new = _bars(D0 + timedelta(days=30), 10)
    new.loc[new.index[-1], "close"] *= 1.03   # today, mid-session
    assert pb.stale_history(new, stored) is None


def test_nothing_older_than_the_fetch_is_never_stale():
    stored = _bars(D0 + timedelta(days=30), 5).assign(
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC))
    new = _bars(D0, 40, dividends={D0 + timedelta(days=33): 1.0}, factor=0.9)
    assert pb.stale_history(new, stored) is None


def test_unstamped_rows_are_treated_as_old():
    stored = _bars(D0, 40).assign(fetched_at=None)
    new = _bars(D0 + timedelta(days=30), 12,
                dividends={D0 + timedelta(days=35): 0.5})
    assert pb.stale_history(new, stored)


# --- adjustment_after / explained_close: which bar a recorded close came from

def _history(dividend_on=None, amount=0.5, split_on=None, ratio=2.0):
    rows = _bars(D0, 10, close0=100.0, step=1.0)
    if dividend_on is not None:
        rows.loc[rows["date"] == pd.Timestamp(dividend_on), "dividends"] = amount
    if split_on is not None:
        rows.loc[rows["date"] == pd.Timestamp(split_on), "stock_splits"] = ratio
    return rows


def test_adjustment_after_is_one_without_events():
    f = pb.adjustment_after(_history())
    assert (f == 1.0).all()


def test_adjustment_after_compounds_yahoo_dividend_factor():
    ex = D0 + timedelta(days=5)
    bars = _history(dividend_on=ex)              # prev close 104 (adjusted)
    f = pb.adjustment_after(bars)
    # Bars before the ex-date carry 1 - 0.5/raw_prev with raw_prev = 104/f.
    expected = 1 / (1 + 0.5 / 104.0)
    assert f[pd.Timestamp(ex - timedelta(days=1))] == pytest.approx(expected)
    assert f[pd.Timestamp(D0)] == pytest.approx(expected)
    assert f[pd.Timestamp(ex)] == 1.0


def test_adjustment_after_split():
    sp = D0 + timedelta(days=5)
    f = pb.adjustment_after(_history(split_on=sp, ratio=4.0))
    assert f[pd.Timestamp(D0)] == pytest.approx(0.25)
    assert f[pd.Timestamp(sp)] == 1.0


def test_explained_close_prefers_the_bar_that_matches_not_the_newest():
    """The 7am run records the prior close; the prediction date's own bar
    arrives later and must not be mistaken for the entry."""
    bars = _history()
    pred_date = D0 + timedelta(days=4)            # close 104; prior 103
    assert pb.explained_close(bars, 103.0, pred_date) == (pred_date - timedelta(days=1), 103.0, 1.0)
    assert pb.explained_close(bars, 104.0, pred_date) == (pred_date, 104.0, 1.0)
    assert pb.explained_close(bars, 103.7, pred_date) is None


def test_explained_close_sees_through_a_later_dividend():
    ex = D0 + timedelta(days=5)
    bars = _history(dividend_on=ex)
    f = 1 - 0.5 / 104.0                          # Yahoo: 1 - d / raw prev close
    # Rebase the stored history as the cache would hold it today.
    bars.loc[bars["date"] < pd.Timestamp(ex), "close"] *= f
    recorded_raw = 104.0                          # copied before the ex-date
    got = pb.explained_close(bars, recorded_raw, ex - timedelta(days=1))
    assert got == (ex - timedelta(days=1), pytest.approx(104.0 * f), pytest.approx(f))


def test_explained_close_exact_wants_that_date():
    bars = _history()
    day = D0 + timedelta(days=4)
    assert pb.explained_close(bars, 104.0, day, exact=True) == (day, 104.0, 1.0)
    assert pb.explained_close(bars, 103.0, day, exact=True) is None


def test_price_moved_tolerance():
    assert not pb.price_moved(100.0, 100.0)
    assert not pb.price_moved(100.0, 100.000001)
    assert pb.price_moved(100.0, 99.5)
    assert not pb.price_moved(None, 99.5)
    assert not pb.price_moved(float("nan"), 99.5)
    assert not pb.price_moved(100.0, 0.0)


# --- the cache writer and the evaluator, on sqlite ---------------------------

@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://", poolclass=StaticPool,
                        connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng, tables=[
        StockPrice.__table__, CacheMetadata.__table__,
        ModelPrediction.__table__, StrategyEvaluation.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


def _stored_closes(db, symbol):
    with db.get_session() as s:
        rows = s.execute(select(StockPrice.date, StockPrice.close)
                         .where(StockPrice.symbol == symbol)
                         .order_by(StockPrice.date)).all()
    return {pd.Timestamp(d).date(): c for d, c in rows}


def _indexed(df):
    return df.set_index("date").rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "dividends": "Dividends",
        "stock_splits": "Stock Splits"})


def test_short_refresh_after_dividend_refetches_the_whole_span(db, monkeypatch):
    svc = CacheService.__new__(CacheService)
    svc._cache_prices("ACME", _indexed(_bars(D0, 60)), "1y")
    assert len(_stored_closes(db, "ACME")) == 60

    ex = D0 + timedelta(days=50)
    rebased_full = _bars(D0, 63, dividends={ex: 0.5}, factor=0.995)
    calls = []

    def fake_fetch(symbol, period="1y", start=None):
        calls.append((symbol, period, start))
        assert start is not None
        return _indexed(rebased_full[rebased_full["date"] >= pd.Timestamp(start)])

    import services.stock_data as sd
    monkeypatch.setattr(sd, "fetch_stock_data", fake_fetch)

    short = rebased_full[rebased_full["date"] >= pd.Timestamp(D0 + timedelta(days=45))]
    svc._cache_prices("ACME", _indexed(short), "3mo")

    closes = _stored_closes(db, "ACME")
    assert len(closes) == 63
    assert len(calls) == 1 and pd.Timestamp(calls[0][2]).date() == D0
    # One basis end to end: every stored close is the rebased one.
    for _, row in rebased_full.iterrows():
        assert closes[row["date"].date()] == pytest.approx(row["close"])


def test_short_refresh_on_the_same_basis_touches_only_its_dates(db, monkeypatch):
    svc = CacheService.__new__(CacheService)
    svc._cache_prices("ACME", _indexed(_bars(D0, 60)), "1y")
    import services.stock_data as sd
    monkeypatch.setattr(sd, "fetch_stock_data",
                        lambda *a, **k: pytest.fail("no refetch expected"))
    svc._cache_prices("ACME", _indexed(_bars(D0 + timedelta(days=55), 8)), "5d")
    assert len(_stored_closes(db, "ACME")) == 63


def test_failed_basis_refetch_still_writes_the_short_frame(db, monkeypatch):
    svc = CacheService.__new__(CacheService)
    svc._cache_prices("ACME", _indexed(_bars(D0, 60)), "1y")
    import services.stock_data as sd

    def boom(*a, **k):
        raise ValueError("rate limited")
    monkeypatch.setattr(sd, "fetch_stock_data", boom)
    ex = D0 + timedelta(days=50)
    short = _bars(D0 + timedelta(days=45), 18, dividends={ex: 0.5}, factor=0.995)
    svc._cache_prices("ACME", _indexed(short), "3mo")
    assert len(_stored_closes(db, "ACME")) == 63


def _pred(symbol, pred_date, prev, model="xgboost_shap", decision="BUY",
          actual=None, predicted=None):
    return ModelPrediction(
        id=f"{symbol}_{model}_{pred_date:%Y%m%d}", symbol=symbol,
        model_name=model, prediction_date=pred_date,
        target_date=pred_date + timedelta(days=1), decision=decision,
        confidence=0.6, previous_close=prev, predicted_close=predicted,
        actual_close=actual, is_public=True)


def _seed_prices(db, symbol, rows):
    with db.get_session() as s:
        for d, c in rows:
            s.add(StockPrice(symbol=symbol, date=d, open=c, high=c, low=c,
                             close=c, volume=1, dividends=0.0, stock_splits=0.0,
                             fetched_at=datetime.now(UTC)))


def _seed_dividend(db, symbol, day, amount):
    with db.get_session() as s:
        row = s.execute(select(StockPrice).where(
            StockPrice.symbol == symbol, StockPrice.date == day)).scalar_one()
        row.dividends = amount


def test_evaluator_scores_entry_on_the_cache_basis(db, monkeypatch):
    """Prediction stored with the pre-dividend close (100); by evaluation
    time the cache holds the rebased bar (99.5). Exit 99.8: a raw-basis
    score calls the BUY wrong (-0.2%); on one basis it is right (+0.3%)."""
    pred_date = date(2026, 8, 3)
    ex = pred_date + timedelta(days=1)
    f = 1 - 0.5 / 100.0
    _seed_prices(db, "MET", [(pred_date - timedelta(days=1), 99.0 * f),
                             (pred_date, 100.0 * f), (ex, 99.8)])
    _seed_dividend(db, "MET", ex, 0.5)
    with db.get_session() as s:
        s.add(_pred("MET", pred_date, prev=100.0, predicted=101.0))

    svc = CacheService.__new__(CacheService)
    monkeypatch.setattr(svc, "get_stock_prices", lambda *a, **k: (None, {}))
    monkeypatch.setattr("utils.trading_calendar.is_market_open_today", lambda: False)
    monkeypatch.setattr(svc, "_hold_band", lambda *a, **k: 0.01)
    n = svc.evaluate_predictions()
    assert n == 1
    with db.get_session() as s:
        row = s.get(ModelPrediction, f"MET_xgboost_shap_{pred_date:%Y%m%d}")
    assert row.previous_close == pytest.approx(100.0 * f)
    assert row.predicted_close == pytest.approx(101.0 * f)
    assert row.actual_close == pytest.approx(99.8)
    assert row.was_correct is True
    assert row.details_json["entry_rebased"]["from"] == 100.0


def test_evaluator_keeps_a_prior_close_entry_when_the_days_bar_arrives(db, monkeypatch):
    """The 7am run recorded yesterday's close (100); today's bar (98) has
    since been written. No dividend: the entry stays 100."""
    pred_date = date(2026, 8, 3)
    _seed_prices(db, "EPAM", [(pred_date - timedelta(days=1), 100.0),
                              (pred_date, 98.0), (pred_date + timedelta(days=1), 99.0)])
    with db.get_session() as s:
        s.add(_pred("EPAM", pred_date, prev=100.0))
    svc = CacheService.__new__(CacheService)
    monkeypatch.setattr(svc, "get_stock_prices", lambda *a, **k: (None, {}))
    monkeypatch.setattr("utils.trading_calendar.is_market_open_today", lambda: False)
    assert svc.evaluate_predictions() == 1
    with db.get_session() as s:
        row = s.get(ModelPrediction, f"EPAM_xgboost_shap_{pred_date:%Y%m%d}")
    assert row.previous_close == 100.0
    assert row.was_correct is False
    assert "entry_rebased" not in (row.details_json or {})


def test_rebase_pass_rescoring_and_dry_run(db, monkeypatch):
    pred_date = date(2026, 8, 12)
    ex = pred_date + timedelta(days=1)
    f = 1 - 0.64 / 107.85
    _seed_prices(db, "ETR", [(pred_date - timedelta(days=1), 107.0 * f),
                             (pred_date, 107.85 * f), (ex, 106.91)])
    _seed_dividend(db, "ETR", ex, 0.64)
    with db.get_session() as s:
        # Scored across the ex-date on the raw basis: entry 107.85 vs exit
        # 106.91 read as a SELL win; rebased entry 107.21 says it was still
        # a win, but by 0.28% instead of 0.87%.
        s.add(_pred("ETR", pred_date, prev=107.85, decision="SELL",
                    actual=106.91))
        # Untouched row: both closes match the cache.
        s.add(_pred("ETR", pred_date, prev=107.85 * f, model="kronos_mini",
                    actual=106.91))
        s.add(StrategyEvaluation(id="x", prediction_id=f"ETR_xgboost_shap_{pred_date:%Y%m%d}",
                                 strategy_name="directional", action="SELL",
                                 entry_price=107.85, exit_price=106.91))
    with db.get_session() as s:
        s.get(ModelPrediction, f"ETR_xgboost_shap_{pred_date:%Y%m%d}").was_correct = True
        s.get(ModelPrediction, f"ETR_xgboost_shap_{pred_date:%Y%m%d}").pnl_dollars = 8.7

    svc = CacheService.__new__(CacheService)
    monkeypatch.setattr(svc, "_hold_band", lambda *a, **k: 0.01)

    preview = svc.rebase_scored_predictions(since_days=None, dry_run=True)
    assert [m["model"] for m in preview] == ["xgboost_shap"]
    with db.get_session() as s:
        assert s.get(ModelPrediction, f"ETR_xgboost_shap_{pred_date:%Y%m%d}") \
            .previous_close == 107.85

    moved = svc.rebase_scored_predictions(since_days=None)
    assert len(moved) == 1
    with db.get_session() as s:
        row = s.get(ModelPrediction, f"ETR_xgboost_shap_{pred_date:%Y%m%d}")
    assert row.previous_close == pytest.approx(107.85 * f)
    assert row.was_correct is True
    assert row.pnl_dollars == pytest.approx(
        1000 * (107.85 * f - 106.91) / (107.85 * f), rel=1e-3)
    assert svc.delete_strategy_evaluations_for([m["id"] for m in moved]) == 1
    # A second pass finds nothing: the row now matches the cache.
    assert svc.rebase_scored_predictions(since_days=None) == []


def test_rebase_pass_leaves_unexplained_entries_alone(db):
    """A recorded entry matching no stored bar (a partial print, a bar
    Yahoo later revised) is not re-picked."""
    pred_date = date(2026, 8, 12)
    _seed_prices(db, "XYZ", [(pred_date - timedelta(days=1), 50.0),
                             (pred_date, 51.0), (pred_date + timedelta(days=1), 52.0)])
    with db.get_session() as s:
        s.add(_pred("XYZ", pred_date, prev=50.7, actual=52.0))
    svc = CacheService.__new__(CacheService)
    assert svc.rebase_scored_predictions(since_days=None, dry_run=True) == []


def test_rebase_pass_ignores_a_rerounded_bar(db):
    """Yahoo re-rounded a close by a hundredth of a percent, no event: the
    recorded row is explained, and left as it is."""
    pred_date = date(2026, 7, 15)
    _seed_prices(db, "CPRX", [(pred_date, 31.48), (pred_date + timedelta(days=1), 31.49)])
    with db.get_session() as s:
        s.add(_pred("CPRX", pred_date, prev=31.475, actual=31.49))
    svc = CacheService.__new__(CacheService)
    assert svc.rebase_scored_predictions(since_days=None, dry_run=True) == []


def test_rebase_pass_ignores_an_event_after_both_bars(db):
    """A dividend after the exit rebases entry and exit alike: nothing to
    re-score."""
    pred_date = date(2026, 8, 12)
    f = 0.99
    _seed_prices(db, "VZ", [(pred_date, 40.0 * f), (pred_date + timedelta(days=1), 40.4 * f),
                            (pred_date + timedelta(days=2), 40.5)])
    _seed_dividend(db, "VZ", pred_date + timedelta(days=2), 40.4 * (1 / f - 1))
    with db.get_session() as s:
        s.add(_pred("VZ", pred_date, prev=40.0, actual=40.4))
    svc = CacheService.__new__(CacheService)
    assert svc.rebase_scored_predictions(since_days=None, dry_run=True) == []


def test_rebase_pass_respects_the_recency_window(db):
    old = date(2026, 1, 5)
    ex = old + timedelta(days=1)
    f = 1 - 0.5 / 55.0
    _seed_prices(db, "OLD", [(old, 55.0 * f), (ex, 51.0)])
    _seed_dividend(db, "OLD", ex, 0.5)
    with db.get_session() as s:
        s.add(_pred("OLD", old, prev=55.0, actual=51.0))
    svc = CacheService.__new__(CacheService)
    assert svc.rebase_scored_predictions(since_days=30, dry_run=True) == []
    assert len(svc.rebase_scored_predictions(since_days=None, dry_run=True)) == 1
