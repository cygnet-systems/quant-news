"""Point-in-time guards for the Bad Apples screen.

Two handlers here swallowed exceptions in a way that changed the answer rather
than merely losing detail:

  - the earnings and insider ``as_of`` filters caught the failure and carried
    on with the *unfiltered* frame, so the check scored against rows dated
    after as_of and reported a clean "pass". That is lookahead presented as a
    result.
  - a failed lookup produced the note "No earnings history" / "No insider
    data", which reads as a fact about the company. Those notes go into the
    research prompt, so a source outage became an assertion.

Both now refuse the check instead. The status was always "n/a"; what matters
is that the note no longer claims something untrue.
"""

import pandas as pd
import pytest

from services.bad_apples_service import check_earnings_misses, check_insider_selling

AS_OF = "2026-08-04"


class Boom:
    """Attribute access raises, like yfinance under a transport error."""

    def __init__(self, msg="upstream exploded"):
        self._msg = msg

    def __getattr__(self, name):
        raise RuntimeError(self._msg)


class Ticker:
    def __init__(self, earnings=None, insider=None):
        self.earnings_history = earnings
        self.get_earnings_dates = earnings
        self.insider_transactions = insider


def insider_frame(dates, values, kinds=None):
    return pd.DataFrame({
        "Start Date": dates,
        "Value": values,
        "Transaction": kinds or ["Sale"] * len(dates),
    })


class TestFailureIsNotAFinding:
    def test_earnings_lookup_failure_is_not_reported_as_no_history(self):
        status, _, _, note = check_earnings_misses(Boom(), AS_OF)
        assert status == "n/a"
        assert "unavailable" in note.lower()
        assert note != "No earnings history"

    def test_insider_lookup_failure_is_not_reported_as_no_data(self):
        status, _, _, note = check_insider_selling(Boom(), AS_OF)
        assert status == "n/a"
        assert "unavailable" in note.lower()
        assert note != "No insider data"

    def test_a_genuinely_empty_frame_still_reads_as_absence(self):
        """An outage and an empty record must not collapse together."""
        tk = Ticker(earnings=pd.DataFrame())
        status, _, _, note = check_earnings_misses(tk, AS_OF)
        assert status == "n/a"
        assert note == "No earnings history"


class TestNoLookahead:
    def test_unparseable_insider_dates_refuse_rather_than_score_unfiltered(self):
        """The bug: the filter threw, df kept every row, and it scored "pass".

        Rows here are all dated well after as_of and are large sells, so
        scoring the unfiltered frame is clearly distinguishable from refusing.
        """
        tk = Ticker(insider=insider_frame(
            ["not-a-date", "also-not-a-date"], [50_000_000, 40_000_000]))
        status, _, _, note = check_insider_selling(tk, AS_OF)
        assert status == "n/a", f"scored anyway: {status} / {note}"
        assert "point-in-time" in note

    def test_insider_transactions_after_as_of_are_excluded(self):
        tk = Ticker(insider=insider_frame(
            ["2026-09-15", "2026-09-20"], [50_000_000, 40_000_000]))
        status, value, _, _ = check_insider_selling(tk, AS_OF)
        assert status == "pass", "future sells must not count against as_of"
        assert "No txns" in str(value)

    def test_insider_transactions_inside_the_window_do_count(self):
        tk = Ticker(insider=insider_frame(
            ["2026-07-20", "2026-07-25"], [50_000_000, 40_000_000]))
        status, _, _, _ = check_insider_selling(tk, AS_OF)
        assert status == "fail", "large recent sells should trip the check"

    def test_unparseable_earnings_index_refuses(self):
        eh = pd.DataFrame({"epsActual": [1.0], "epsEstimate": [1.2]},
                          index=["definitely not a date"])
        status, _, _, note = check_earnings_misses(Ticker(earnings=eh), AS_OF)
        assert status == "n/a" or "point-in-time" in note


class TestNullTicker:
    @pytest.mark.parametrize("check", [check_earnings_misses,
                                       check_insider_selling])
    def test_missing_ticker_is_reported_not_crashed(self, check):
        status, _, _, note = check(None, AS_OF)
        assert status == "n/a"
        assert "unavailable" in note.lower()
