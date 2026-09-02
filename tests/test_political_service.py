"""Political/institutional blocks are point-in-time on the FILING date and
say what they are (lagged positioning), never a signal."""

from unittest.mock import patch

import pytest

from services import political_service as ps


@pytest.fixture(autouse=True)
def _clear():
    ps._CACHE.clear()
    yield
    ps._CACHE.clear()


CONGRESS = {"trades": [
    # Filed AFTER as_of: invisible on 2026-07-20 even though traded before.
    {"chamber": "HOUSE", "politician": "Hon. Sam T. Liccardo", "party": "D",
     "state_district": "CA16", "transaction_type": "SELL",
     "transaction_date": "2026-07-21", "filed_date": "2026-07-27",
     "amount_min": "15001.00", "amount_max": "50000.00", "owner_code": "SELF"},
    {"chamber": "HOUSE", "politician": "Hon. Dan Newhouse",
     "politician_canonical": "Dan Newhouse", "party": "R",
     "state_district": "WA04", "transaction_type": "SELL",
     "transaction_date": "2026-07-10", "filed_date": "2026-07-17",
     "amount_min": "1001.00", "amount_max": "15000.00", "owner_code": "SPOUSE"},
    # Senate row without filed_date: notification_date is the visibility.
    {"chamber": "SENATE", "politician": "Sheldon Whitehouse", "party": "D",
     "state": "RI", "transaction_type": "BUY", "transaction_date": "2026-06-30",
     "notification_date": "2026-07-08", "amount_min": "15001.00",
     "amount_max": "50000.00", "owner_code": "SELF"},
    # Too old for the 180-day window.
    {"chamber": "HOUSE", "politician": "Old Trade", "party": "R",
     "transaction_type": "BUY", "transaction_date": "2025-01-01",
     "filed_date": "2025-01-10", "amount_min": "1001.00", "amount_max": "15000.00"},
]}


def test_congress_trades_are_visible_only_from_filing_date():
    with patch.object(ps, "_av_get", return_value=CONGRESS):
        c = ps.get_congress_trades("NVDA", "2026-07-20")
    names = [t["politician"] for t in c["trades"]]
    assert "Sam T. Liccardo" not in " ".join(names)      # filed 07-27
    assert "Dan Newhouse" in " ".join(names)              # filed 07-17
    assert "Sheldon Whitehouse" in " ".join(names)        # notified 07-08
    assert "Old Trade" not in " ".join(names)
    assert c["n"] == 2 and c["buys"] == 1 and c["sells"] == 1
    assert c["by_party"] == {"D": {"buys": 1, "sells": 0}, "R": {"buys": 0, "sells": 1}}
    block = ps.format_congress_block("NVDA", c)
    assert "visible through 2026-07-20" in block
    assert "2026-07-10 SELL $1K–$15K: Dan Newhouse (R-WA04, HOUSE), spouse; filed 2026-07-17" in block
    assert "never as a timing signal" in block


def test_no_trades_is_stated_not_silent():
    with patch.object(ps, "_av_get", return_value={"trades": []}):
        c = ps.get_congress_trades("BHF", "2026-09-01")
    block = ps.format_congress_block("BHF", c)
    assert "None disclosed in the window" in block
    assert "Absence is uninformative" in block


def test_vendor_throttle_raises_so_caller_records_a_gap():
    from types import SimpleNamespace
    fake_api = SimpleNamespace(ALPHA_VANTAGE_API_KEY="k", ALPHA_VANTAGE_BASE_URL="u",
                               DEFAULT_TIMEOUT=1)
    with patch.object(ps.requests, "get") as get, \
         patch.object(ps, "API", fake_api), \
         patch.object(ps, "alpha_vantage_bucket"):
        get.return_value.json.return_value = {"Note": "Thank you for using Alpha Vantage! rate limit"}
        get.return_value.raise_for_status.return_value = None
        with pytest.raises(ps.AlphaVantageUnavailable, match="rate limit"):
            ps.get_congress_trades("BHF", "2026-09-01")
    blocks, problems = None, None
    with patch.object(ps, "get_congress_trades", side_effect=ps.AlphaVantageUnavailable("throttled")), \
         patch.object(ps, "get_institutional_holdings", side_effect=ps.AlphaVantageUnavailable("throttled")):
        blocks, problems = ps.political_blocks("BHF", "2026-09-01")
    assert blocks == []
    assert len(problems) == 2 and "throttled" in problems[0]


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
