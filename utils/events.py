"""Event-calendar gate for short-horizon predictions.

A 1-5 day trading decision must know whether earnings or ex-dividend
dates fall inside the hold window. Works for backtest dates too:
yfinance's earnings history includes past dates, so "next earnings after
as-of" is answerable without lookahead on the decision itself.
"""

import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_events_cache: dict[str, dict] = {}


def get_upcoming_events(symbol: str, as_of: str, window_days: int = 5) -> dict:
    """Find earnings / ex-dividend dates near the as-of date.

    Returns {next_earnings, ex_dividend, in_window: bool, window_end}
    with ISO date strings (or None). Best-effort: failures return {}.
    """
    cache_key = f"{symbol}:{as_of}"
    if cache_key in _events_cache:
        return _events_cache[cache_key]

    try:
        import yfinance as yf

        as_of_d = date.fromisoformat(str(as_of)[:10])
        window_end = as_of_d + timedelta(days=window_days + 2)  # pad for weekends

        t = yf.Ticker(symbol)

        next_earnings: Optional[date] = None
        try:
            ed = t.get_earnings_dates(limit=24)
            if ed is not None and len(ed) > 0:
                dates = sorted(d.date() for d in ed.index.tz_localize(None))
                future = [d for d in dates if d > as_of_d]
                if future:
                    next_earnings = future[0]
        except Exception as e:
            logger.debug(f"Earnings dates lookup failed for {symbol}: {e}")

        ex_div: Optional[date] = None
        try:
            divs = t.dividends
            if divs is not None and len(divs) > 0:
                div_dates = sorted(d.date() for d in divs.index.tz_localize(None))
                future_divs = [d for d in div_dates if d > as_of_d]
                if future_divs:
                    ex_div = future_divs[0]
                elif as_of_d >= date.today() - timedelta(days=2):
                    # Live mode: check the forward calendar for a declared ex-div
                    cal = t.calendar or {}
                    exd = cal.get("Ex-Dividend Date")
                    if exd:
                        ex_div = exd if isinstance(exd, date) else None
        except Exception as e:
            logger.debug(f"Dividend lookup failed for {symbol}: {e}")

        earnings_in_window = bool(next_earnings and as_of_d < next_earnings <= window_end)
        exdiv_in_window = bool(ex_div and as_of_d < ex_div <= window_end)

        result = {
            "next_earnings": next_earnings.isoformat() if next_earnings else None,
            "ex_dividend": ex_div.isoformat() if ex_div else None,
            "in_window": earnings_in_window or exdiv_in_window,
            "earnings_in_window": earnings_in_window,
            "exdiv_in_window": exdiv_in_window,
            "window_end": window_end.isoformat(),
        }
        _events_cache[cache_key] = result
        return result
    except Exception as e:
        logger.warning(f"Event lookup failed for {symbol}: {e}")
        # Not {}: a silent empty dict made the whole block vanish from the
        # prompt, indistinguishable from "no events". The LLM should know
        # the calendar was checked and could not be read.
        return {"unavailable": True}


def format_events_block(symbol: str, events: dict, window_days: int = 5) -> str:
    """Format the event gate as a prompt block. Empty string if unavailable."""
    if not events:
        return ""
    if events.get("unavailable"):
        return (f"[{symbol}: event calendar]\n"
                "Calendar lookup unavailable, event risk (earnings/ex-dividend) "
                "is UNKNOWN for the hold window. Treat as unverified, not as absent.")
    lines = [f"[{symbol}: event calendar for the {window_days}-day hold window "
             f"ending {events.get('window_end', '?')}]"]
    ne = events.get("next_earnings")
    lines.append(f"Next earnings: {ne or 'not found'}"
                 + (": INSIDE HOLD WINDOW" if events.get("earnings_in_window") else ""))
    xd = events.get("ex_dividend")
    lines.append(f"Next ex-dividend: {xd or 'not found'}"
                 + (": INSIDE HOLD WINDOW" if events.get("exdiv_in_window") else ""))
    if events.get("in_window"):
        lines.append("GATE: a scheduled event falls inside the hold window. Cap confidence "
                     "at 0.6 and state the event risk explicitly in the risk assessment.")
    else:
        lines.append("No scheduled earnings/ex-dividend inside the hold window.")
    return "\n".join(lines)
