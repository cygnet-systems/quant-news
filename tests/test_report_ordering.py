"""The "latest report" queries rank by trading day, then write time.

Ordering by created_at alone let a backfill that regenerated an old day's
reports (2026-09-04) hide the current day's from the watchlist rail and the
Home reading pane: the rail said 08-28 while 09-03 reports existed.
"""
from contextlib import contextmanager

from sqlalchemy.dialects import postgresql

from services import cache_service


class _Result:
    def scalars(self):
        return self

    def all(self):
        return []

    def __iter__(self):
        return iter(())


class _Session:
    def __init__(self, seen):
        self.seen = seen

    def execute(self, stmt):
        self.seen.append(stmt)
        return _Result()


def _capture(monkeypatch):
    seen = []

    @contextmanager
    def fake_session():
        yield _Session(seen)

    monkeypatch.setattr(cache_service, "get_session", fake_session)
    monkeypatch.setattr(cache_service, "_current_uid", lambda: None)
    return seen


def _order_by(stmt) -> str:
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    return sql.split("ORDER BY", 1)[1]


def test_per_symbol_history_is_newest_trading_day_first(monkeypatch):
    seen = _capture(monkeypatch)
    cache_service.CacheService.get_trading_agent_reports(object(), "bac", limit=5)
    tail = _order_by(seen[0])
    assert tail.index("trade_date DESC") < tail.index("created_at DESC")


def test_latest_per_symbol_is_newest_trading_day_first(monkeypatch):
    seen = _capture(monkeypatch)
    cache_service.CacheService.latest_reports_by_symbol(object())
    tail = _order_by(seen[0])
    assert "symbol" in tail.split("trade_date")[0]
    assert tail.index("trade_date DESC") < tail.index("created_at DESC")
