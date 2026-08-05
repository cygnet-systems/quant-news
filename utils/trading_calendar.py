"""NYSE trading calendar utilities.

Provides functions to determine trading days, next/previous trading days,
and whether the market is currently open.
"""

from datetime import date, datetime, timedelta
from typing import Union

import pandas_market_calendars as mcal

# NYSE calendar instance (cached at module level)
_nyse = mcal.get_calendar("NYSE")


def _to_date(d: Union[str, date, datetime]) -> date:
    """Convert string or datetime to date."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    return d


def is_trading_day(d: Union[str, date, datetime]) -> bool:
    """Check if a given date is a NYSE trading day.

    Args:
        d: Date to check (string "YYYY-MM-DD", date, or datetime).

    Returns:
        True if the date is a trading day.
    """
    d = _to_date(d)
    schedule = _nyse.schedule(start_date=d, end_date=d)
    return len(schedule) > 0


def get_next_trading_day(d: Union[str, date, datetime]) -> date:
    """Get the next NYSE trading day after the given date.

    Args:
        d: Reference date.

    Returns:
        The next trading day (strictly after d).
    """
    d = _to_date(d)
    # Look ahead up to 10 days to handle long weekends/holidays
    start = d + timedelta(days=1)
    end = d + timedelta(days=10)
    schedule = _nyse.schedule(start_date=start, end_date=end)
    if len(schedule) > 0:
        return schedule.index[0].date()
    # Fallback: skip weekends manually
    next_day = d + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def get_previous_trading_day(d: Union[str, date, datetime]) -> date:
    """Get the most recent NYSE trading day before the given date.

    Args:
        d: Reference date.

    Returns:
        The most recent trading day (strictly before d).
    """
    d = _to_date(d)
    # Look back up to 10 days
    end = d - timedelta(days=1)
    start = d - timedelta(days=10)
    schedule = _nyse.schedule(start_date=start, end_date=end)
    if len(schedule) > 0:
        return schedule.index[-1].date()
    # Fallback: skip weekends manually
    prev_day = d - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)
    return prev_day


def get_last_completed_trading_day() -> date:
    """Most recent trading day whose session has already closed.

    Before/during today's session this is the previous trading day; after
    today's close (or on weekends/holidays) it is today / the last trading day.
    Use as the "as-of" date when predicting the upcoming session's close.
    """
    today = date.today()
    if is_trading_day(today):
        schedule = _nyse.schedule(start_date=today, end_date=today)
        market_close = schedule.iloc[0]["market_close"]
        now = datetime.now(tz=market_close.tzinfo)
        if now >= market_close:
            return today
    return get_previous_trading_day(today)


def is_market_open_today() -> bool:
    """Check if NYSE is open today and currently in trading hours.

    Returns:
        True if today is a trading day and current time is within market hours
        (9:30 AM - 4:00 PM ET).
    """
    today = date.today()
    if not is_trading_day(today):
        return False

    schedule = _nyse.schedule(start_date=today, end_date=today)
    if len(schedule) == 0:
        return False

    now = datetime.now(tz=schedule.iloc[0]["market_open"].tzinfo)
    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]

    return market_open <= now <= market_close


def non_trading_days(start: Union[str, date, datetime],
                     end: Union[str, date, datetime]) -> list[date]:
    """All non-trading calendar dates in [start, end] (weekends + holidays).

    One schedule query instead of per-day is_trading_day() calls — this
    feeds date-picker disabled_days lists, which span a year or more.
    """
    start_d, end_d = _to_date(start), _to_date(end)
    valid = {d.date() for d in _nyse.valid_days(start_d, end_d)}
    out = []
    cur = start_d
    while cur <= end_d:
        if cur not in valid:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def get_default_target_day() -> date:
    """The session a prediction made right now would target.

    That is the first trading day whose close has not happened yet: before
    today's close it is today, after it (or on a weekend/holiday) the next
    trading day.
    """
    return get_next_trading_day(get_last_completed_trading_day())


def resolve_target_and_cutoff(
    value: Union[str, date, datetime, None] = None,
) -> tuple[date, date]:
    """Resolve a user-selected TARGET date to ``(target, data_cutoff)``.

    The target is the session whose close is being predicted, so it must be
    a trading day — a weekend/holiday selection snaps forward to the next
    one. Everything the models and the report are allowed to see is cut off
    at the *previous trading day*: a Monday target sees nothing after the
    preceding Friday's close, and a target after a holiday skips the holiday.

    Returns:
        (target, data_cutoff) — both trading days, cutoff strictly before target.
    """
    target = _to_date(value) if value else get_default_target_day()
    if not is_trading_day(target):
        target = get_next_trading_day(target)
    return target, get_previous_trading_day(target)
