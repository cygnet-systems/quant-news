"""The replay must score exactly as the live evaluator does.

scripts/replay_ensemble_methods.py compares combination methods by replaying
them over stored history. That comparison is only meaningful if a replayed
verdict is judged by the same rules as a real one. A second, subtly different
scoring function would produce a ranking that reflects the scorer rather than
the methods.

These pin the replay's scorer against models.base.compute_pnl and the HOLD
band convention the evaluator uses.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from models.base import compute_pnl  # noqa: E402
from replay_ensemble_methods import score  # noqa: E402

BAND = 0.02   # 2% no-trade band for these cases


class TestDirectionalScoring:
    def test_buy_is_right_when_the_close_rises(self):
        correct, pnl = score("BUY", 100.0, 105.0, BAND)
        assert correct is True
        assert pnl == pytest.approx(compute_pnl("BUY", 100.0, 105.0))
        assert pnl > 0

    def test_buy_is_wrong_when_the_close_falls(self):
        correct, pnl = score("BUY", 100.0, 95.0, BAND)
        assert correct is False
        assert pnl < 0

    def test_sell_is_right_when_the_close_falls(self):
        correct, pnl = score("SELL", 100.0, 95.0, BAND)
        assert correct is True
        assert pnl == pytest.approx(compute_pnl("SELL", 100.0, 95.0))
        assert pnl > 0

    def test_sell_is_wrong_when_the_close_rises(self):
        correct, pnl = score("SELL", 100.0, 105.0, BAND)
        assert correct is False
        assert pnl < 0

    def test_pnl_always_matches_the_shared_helper(self):
        """No separate P&L convention may creep into the replay."""
        for decision in ("BUY", "SELL", "HOLD"):
            for actual in (90.0, 99.5, 100.0, 100.5, 110.0):
                _, pnl = score(decision, 100.0, actual, BAND)
                assert pnl == pytest.approx(compute_pnl(decision, 100.0, actual))


class TestHoldScoring:
    def test_hold_is_right_when_the_move_stays_inside_the_band(self):
        correct, pnl = score("HOLD", 100.0, 101.0, BAND)   # +1% vs 2% band
        assert correct is True
        assert pnl == 0.0

    def test_hold_is_wrong_when_the_move_breaks_the_band(self):
        correct, _ = score("HOLD", 100.0, 105.0, BAND)     # +5% vs 2% band
        assert correct is False

    def test_the_band_is_two_sided(self):
        assert score("HOLD", 100.0, 99.0, BAND)[0] is True
        assert score("HOLD", 100.0, 95.0, BAND)[0] is False

    def test_a_hold_never_books_pnl(self):
        for actual in (80.0, 100.0, 130.0):
            assert score("HOLD", 100.0, actual, BAND)[1] == 0.0

    def test_the_band_edge_counts_as_held(self):
        """Inclusive, matching CacheService's `move <= band`."""
        assert score("HOLD", 100.0, 102.0, BAND)[0] is True


class TestEdges:
    def test_an_unchanged_close_is_not_a_buy_win(self):
        """`went_up` is strict, so a flat session does not reward a long."""
        assert score("BUY", 100.0, 100.0, BAND)[0] is False
        assert score("SELL", 100.0, 100.0, BAND)[0] is True

    def test_zero_previous_close_does_not_raise(self):
        correct, pnl = score("HOLD", 0.0, 0.0, BAND)
        assert correct is True and pnl == 0.0
