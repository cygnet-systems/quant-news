"""What the scheduler reads back from a run, and what it mails about it.

The 2026-08-06 failure lived entirely in this seam. The analysis ran, stored
120 predictions and exited zero; the scheduler then parsed the run's summary
out of the 25-line tail it had kept for display, found nothing (a 20-symbol
summary is 53 lines of indented JSON), and mailed "produced no recommendations
: check the History tab" about a run that had produced twenty of them.

So these tests pin the two halves that must not drift back together:

* the summary is parsed from the whole of stdout, at any watchlist size
* the stored log is for humans and is allowed to be truncated
"""

import json

import pytest

from services import scheduler_service as ss

WATCHLIST = ["PANW", "BAC", "VZ", "HWM", "DOC", "HPQ", "LUV", "TPL", "MPWR",
             "MCD", "ROP", "ETR", "CMS", "XYZ", "HIG", "IP", "FLEX", "MET",
             "FIS", "TYL"]


def _summary(symbols):
    """The shape scripts/daily_analysis.py prints with --json."""
    return {
        "symbols": symbols,
        "skipped": [],
        "target_date": "2026-08-06",
        "as_of": "2026-08-05",
        "is_backtest": False,
        "predictions_stored": 6 * len(symbols),
        "evaluated": 0,
        "actions": {s: "BUY" for s in symbols},
        "model_coverage": {"kronos": len(symbols), "ensemble": len(symbols)},
        "report_archived": True,
        "degraded": [],
        "duration_s": 2566.0,
    }


def _stdout(symbols):
    return json.dumps(_summary(symbols), indent=2) + "\n"


@pytest.mark.parametrize("count", [1, 5, 20, 60])
def test_summary_survives_any_watchlist_size(count):
    """The parser must not have a length at which it silently returns {}."""
    symbols = [f"SYM{i}" for i in range(count)]
    parsed = ss._parse_summary(_stdout(symbols))
    assert len(parsed.get("actions") or {}) == count


def test_twenty_symbol_summary_is_longer_than_the_stored_tail():
    """The condition that made the bug reachable, stated outright."""
    assert len(_stdout(WATCHLIST).splitlines()) > 25


def test_summary_ignores_log_lines_printed_before_it():
    noise = ("2026-08-06 08:30:01 INFO services.analysis_runner: starting\n"
             "  {not json}\n")
    parsed = ss._parse_summary(noise + _stdout(WATCHLIST))
    assert parsed["predictions_stored"] == 120


def test_summary_ignores_trailing_output():
    """raw_decode, not loads: a print() after the summary must not lose it."""
    parsed = ss._parse_summary(_stdout(WATCHLIST) + "done in 2566s\n")
    assert len(parsed["actions"]) == 20


def test_no_summary_is_empty_not_an_error():
    assert ss._parse_summary("") == {}
    assert ss._parse_summary("nothing structured here\n") == {}


def test_run_log_keeps_the_summary_and_the_stderr_log():
    """Both streams. Only stdout used to be kept, and stderr is the log."""
    log = ss._run_log(_stdout(WATCHLIST), "INFO: fetched news for PANW\n")
    assert '"predictions_stored": 120' in log
    assert "fetched news for PANW" in log


def test_run_log_trims_the_log_and_keeps_the_summary():
    """Under truncation the summary survives; the oldest log lines go."""
    noisy = "\n".join(f"line {i}" for i in range(5000))
    log = ss._run_log(_stdout(WATCHLIST), noisy)
    assert len(log) <= ss.RUN_LOG_MAX_CHARS
    assert '"target_date": "2026-08-06"' in log
    assert "line 4999" in log
    assert "line 0\n" not in log


def test_run_log_reports_a_timeout_with_the_output_it_got():
    log = ss._run_log("", "INFO: waiting on provider\n",
                      header="TIMED OUT: exceeded 4500s wall clock")
    assert "TIMED OUT" in log
    assert "waiting on provider" in log


def test_abridged_log_keeps_both_ends():
    """The panel polls; the log it ships keeps the summary and the failure."""
    text = _stdout(WATCHLIST) + "\n".join(f"line {i}" for i in range(4000)) + "\nTRACEBACK"
    short = ss._abridge(text, 4000)
    assert len(short) < len(text)
    assert short.startswith("{")
    assert short.endswith("TRACEBACK")
    assert "characters omitted" in short


def test_abridge_leaves_a_short_log_alone():
    assert ss._abridge("two\nlines", 4000) == "two\nlines"
    assert ss._abridge(None, 4000) is None


def test_holiday_no_op_is_not_mailed_as_a_failure(monkeypatch):
    """--only-trading-days prints a no_session summary; nothing is sent."""
    sent = []

    class _Notify:
        @staticmethod
        def enabled():
            return True

        @staticmethod
        def notify_analysis(*a, **k):
            sent.append("analysis")

        @staticmethod
        def notify_job_failure(*a, **k):
            sent.append("failure")

        @staticmethod
        def notify_partial(*a, **k):
            sent.append("partial")

    import sys
    monkeypatch.setitem(sys.modules, "services.notify_service", _Notify)
    ss._notify({"id": "daily_analysis", "kind": "analysis"}, "success",
               {"no_session": True, "date": "2026-07-04"}, "", None, 1200)
    assert sent == []
