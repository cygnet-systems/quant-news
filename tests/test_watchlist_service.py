"""Guards for durable watchlist history.

Two rules are easy to get backwards. Storage keeps every distinct group, so a
past search stays recoverable; the collapse that hides REX once REX+WGO+IOVA
exists is presentation only. And symbol filtering has to match whole tickers:
a substring match would surface every SPYG search under SPY.
"""

from unittest.mock import patch

from services import watchlist_service as ws


def row(csv, count=1, last="2026-08-05T12:00:00"):
    return {"id": 0, "symbols": csv.split(","), "symbols_csv": csv,
            "first_used_at": last, "last_used_at": last, "use_count": count}


class TestKey:
    def test_normalises_case_order_and_whitespace(self):
        assert ws._key([" msft ", "aapl", "MSFT"]) == "AAPL,MSFT"

    def test_drops_blanks(self):
        assert ws._key(["", "  ", "AAPL"]) == "AAPL"
        assert ws._key([]) == ""


class TestRecentGroupsCollapse:
    def _recent(self, csvs, limit=5):
        with patch.object(ws, "_rows", return_value=[row(c) for c in csvs]):
            return ws.recent_groups(limit=limit)

    def test_superset_hides_its_prefixes(self):
        """Building a watchlist up one ticker at a time leaves one chip."""
        assert self._recent(["IOVA,REX,WGO", "REX,WGO", "REX"]) == [
            ["IOVA", "REX", "WGO"]]

    def test_disjoint_groups_all_survive(self):
        assert self._recent(["AAPL,MSFT", "REX,WGO"]) == [
            ["AAPL", "MSFT"], ["REX", "WGO"]]

    def test_older_superset_does_not_hide_newer_subset(self):
        """Only a group seen *more recently* may absorb another.

        Rows arrive newest first, so a later-listed superset must not swallow
        the group the user actually looked at last.
        """
        assert self._recent(["REX", "IOVA,REX,WGO"]) == [
            ["REX"], ["IOVA", "REX", "WGO"]]

    def test_respects_limit(self):
        out = self._recent(["A", "B", "C", "D", "E", "F"], limit=3)
        assert out == [["A"], ["B"], ["C"]]

    def test_empty_history(self):
        assert self._recent([]) == []


class TestRecordGuards:
    def test_empty_group_is_not_recorded(self):
        assert ws.record_group([]) is False
        assert ws.record_group(["", "  "]) is False

    def test_oversized_group_is_refused(self):
        """A runaway selection should not become a permanent history row."""
        assert ws.record_group([f"S{i}" for i in range(ws.MAX_SYMBOLS_PER_GROUP + 1)]) is False

    def test_db_failure_does_not_propagate(self):
        """History is incidental to fetching data and must never break it."""
        with patch.object(ws, "get_session", side_effect=RuntimeError("db down")):
            assert ws.record_group(["AAPL"]) is False
