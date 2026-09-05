"""A run may not spend past its ceiling, whatever the cause.

The 2026-09-02..05 runaway cost roughly $20 through a bug nobody had
anticipated. Fixing that bug was necessary; it bounds nothing else. This is
the cause-independent limit: the run stops buying model calls once it is over
budget, degrades the blocks it could not afford, and says so.

It bounds ONE RUN, not a day. A scheduler that starts runs in a loop can
still spend the ceiling many times over, which is why the backfill sweep has
its own per-date cap. Pinned here so nobody mistakes this for a daily budget.
"""

import pytest

from services import usage_service as us
from services.usage_service import SpendCeilingReached


@pytest.fixture(autouse=True)
def _clean():
    us.reset_run_spend()
    yield
    us.reset_run_spend()


@pytest.fixture
def run(monkeypatch):
    """Pin the run id the ceiling reads, so no database is needed."""
    import services.progress_service as prog

    monkeypatch.setattr(prog, "current_run_id", lambda: "run-1")
    return "run-1"


def _ceiling(monkeypatch, value):
    import config

    monkeypatch.setattr(config, "RUN_SPEND_CEILING_USD", value, raising=False)


class TestAccrual:
    def test_spend_accumulates_per_run(self):
        us._accrue("a", 0.4)
        us._accrue("a", 0.35)
        us._accrue("b", 0.9)
        assert us.spent_on_run("a") == pytest.approx(0.75)
        assert us.spent_on_run("b") == pytest.approx(0.9)

    def test_an_unknown_run_has_spent_nothing(self):
        assert us.spent_on_run("nope") == 0.0
        assert us.spent_on_run(None) == 0.0


class TestTheGate:
    def test_under_budget_passes(self, run, monkeypatch):
        _ceiling(monkeypatch, 1.00)
        us._accrue(run, 0.99)
        us.check_spend_ceiling()          # must not raise

    def test_at_the_ceiling_refuses(self, run, monkeypatch):
        _ceiling(monkeypatch, 1.00)
        us._accrue(run, 1.00)
        with pytest.raises(SpendCeilingReached):
            us.check_spend_ceiling()

    def test_over_the_ceiling_refuses(self, run, monkeypatch):
        _ceiling(monkeypatch, 1.00)
        us._accrue(run, 4.20)
        with pytest.raises(SpendCeilingReached) as exc:
            us.check_spend_ceiling("web_search/gpt-5.6-luna")
        assert "4.20" in str(exc.value) and "1.00" in str(exc.value)

    def test_zero_disables_the_ceiling(self, run, monkeypatch):
        """An unset ceiling must not silently stop a run at $0."""
        _ceiling(monkeypatch, 0)
        us._accrue(run, 99.0)
        us.check_spend_ceiling()

    def test_a_broken_check_never_blocks_a_run(self, run, monkeypatch):
        """Telemetry must not be able to stop work that is within budget."""
        import services.progress_service as prog

        _ceiling(monkeypatch, 1.00)

        def _boom():
            raise RuntimeError("diskcache is gone")

        monkeypatch.setattr(prog, "current_run_id", _boom)
        us.check_spend_ceiling()          # degrades to allowing the call


class TestItIsARunBudgetNotADayBudget:
    def test_a_second_run_starts_with_a_fresh_budget(self, monkeypatch):
        """Documents the limit of this control: 48 runs can each spend the
        ceiling. The backfill sweep's per-date cap is what bounds that."""
        _ceiling(monkeypatch, 1.00)
        import services.progress_service as prog

        us._accrue("run-1", 1.00)
        monkeypatch.setattr(prog, "current_run_id", lambda: "run-2")
        us.check_spend_ceiling()          # run-2 is not run-1's spend
