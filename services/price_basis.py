"""Whether a stored price history is still on the adjustment basis of a
fresh fetch, and whether a scored prediction's entry and exit still are.

Yahoo's adjusted bars are rebased backwards every time a dividend goes ex or
a split takes effect: every bar before the event is scaled by the
adjustment factor. The price cache replaces only the dates a fetch covers
(a short refresh must not wipe a long history), so a fetch made after an
event writes post-event bars beside pre-event ones and the stored series
acquires a step the market never printed. Measured on 2026-09-06: up to
1.8% on off-watchlist names and at the two-year tail of several histories;
nothing on the watchlist, only because the daily job happens to refetch a
full year every morning.

Two independent tells, either of which means the rows older than the
fetch must be refetched on the new basis:

  - an adjustment event (dividend or split) dated after those rows were
    fetched. Events are read from both frames, so one that fell in the
    gap between the stored newest bar and the fetch window still counts
    when a stored row recorded it;
  - the same date closing at two different prices in the two frames. The
    oldest overlapping bar decides, since the newest of either frame can
    be a partial print fetched during the session.

Plain data in, a reason string (or None) out; nothing here touches the
database or the network.
"""

import pandas as pd

# Closes on the same basis agree to float precision. The smallest dividend
# on the watchlist (HWM, $0.14 on $290) moves the basis by 0.048%, well
# above this.
BASIS_TOLERANCE = 1e-4


def _event_dates(frame: pd.DataFrame) -> set:
    out = set()
    for col in ("dividends", "stock_splits"):
        if col in frame.columns:
            hit = pd.to_numeric(frame[col], errors="coerce").fillna(0) > 0
            out.update(pd.to_datetime(frame.loc[hit, "date"]).dt.normalize())
    return out


def stale_history(new: pd.DataFrame, stored: pd.DataFrame):
    """Why the stored bars older than ``new``'s first date can no longer sit
    beside it, or None when they can.

    Both frames carry ``date`` and ``close``, and ``dividends`` /
    ``stock_splits`` where known. ``stored`` also carries ``fetched_at``
    (tz-aware UTC); a row without one predates the column and is treated
    as older than any event.
    """
    if new is None or new.empty or stored is None or stored.empty:
        return None
    new = new.assign(date=pd.to_datetime(new["date"]).dt.normalize())
    stored = stored.assign(date=pd.to_datetime(stored["date"]).dt.normalize())
    first_new = new["date"].min()
    older = stored[stored["date"] < first_new]
    if older.empty:
        return None

    if "fetched_at" in older.columns:
        oldest_fetch = pd.to_datetime(
            older["fetched_at"], utc=True, errors="coerce").min()
    else:
        oldest_fetch = pd.NaT
    for ev in sorted(_event_dates(new) | _event_dates(stored)):
        # A row fetched during the ex-date's own session may or may not
        # carry the adjustment yet: the whole day counts as before it.
        if pd.isna(oldest_fetch) or \
                oldest_fetch < ev.tz_localize("UTC") + pd.Timedelta(days=1):
            when = "never stamped" if pd.isna(oldest_fetch) \
                else oldest_fetch.strftime("%Y-%m-%d %H:%MZ")
            return (f"dividend/split on {ev.date()} postdates rows "
                    f"fetched {when}")

    overlap = new[["date", "close"]].merge(
        stored[["date", "close"]], on="date", suffixes=("_new", "_stored"))
    # Neither frame's newest bar: either can be a partial print.
    overlap = overlap[(overlap["date"] < new["date"].max())
                      & (overlap["date"] < stored["date"].max())]
    if overlap.empty:
        return None
    row = overlap.sort_values("date").iloc[0]
    fetched, kept = float(row["close_new"]), float(row["close_stored"])
    if fetched > 0 and kept > 0 and abs(kept / fetched - 1) > BASIS_TOLERANCE:
        return (f"stored close {kept:.4f} vs fetched {fetched:.4f} on "
                f"{row['date'].date()}")
    return None


def adjustment_after(bars: pd.DataFrame) -> pd.Series:
    """For each stored bar, the factor every close on or before it has been
    scaled by since, from the adjustment events dated after it: Yahoo's
    ``1 - dividend / previous raw close`` per dividend and ``1 / ratio``
    per split, compounded. A bar with no later event maps to 1.0.

    The previous raw close is recovered from the stored adjusted one, so
    events are walked newest first: with G the product of the factors of
    all later events, a dividend d on a bar whose predecessor closed at
    adjusted p solves f = 1 / (1 + d * G / p).
    """
    bars = bars.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(bars["date"]).dt.normalize()
    closes = pd.to_numeric(bars["close"], errors="coerce")
    divs = pd.to_numeric(bars.get("dividends"), errors="coerce").fillna(0.0)
    splits = pd.to_numeric(bars.get("stock_splits"), errors="coerce").fillna(0.0)

    factor_at = {}          # event row index -> that event's own factor
    running = 1.0           # product of factors of events walked so far
    for i in range(len(bars) - 1, 0, -1):
        f = 1.0
        if splits.iloc[i] > 0:
            f *= 1.0 / float(splits.iloc[i])
        if divs.iloc[i] > 0:
            prev = closes.iloc[i - 1]
            if pd.notna(prev) and prev > 0:
                f *= 1.0 / (1.0 + float(divs.iloc[i]) * running / float(prev))
        if f != 1.0:
            factor_at[i] = f
            running *= f

    out = []
    acc = 1.0
    for i in range(len(bars) - 1, -1, -1):
        out.append(acc)
        acc *= factor_at.get(i, 1.0)
    return pd.Series(list(reversed(out)), index=dates, name="factor")


def explained_close(bars: pd.DataFrame, recorded, on_or_before,
                    lookback: int = 5, exact: bool = False):
    """The stored bar a recorded close came from, that bar's close on the
    cache's current basis, and the factor the bar has been scaled by since
    (1.0 when nothing has), as ``(date, close, factor)``; None when no bar
    explains the recorded value.

    A recorded close matches a bar when the bar's current close equals the
    recorded value scaled by the adjustments since that bar. With no
    adjustment since, that is plain equality; a recorded value that
    matches nothing (a partial print, a bar Yahoo has since revised) is
    left alone, since re-picking a different day's close would change what
    the row measures. ``exact`` demands the bar dated ``on_or_before``
    itself; otherwise the newest ``lookback`` bars on or before it are
    tried, newest first.
    """
    if bars is None or bars.empty or recorded is None:
        return None
    try:
        recorded = float(recorded)
    except (TypeError, ValueError):
        return None
    if not recorded > 0:
        return None
    factors = adjustment_after(bars)
    limit = pd.Timestamp(on_or_before).normalize()
    bars = bars.assign(date=pd.to_datetime(bars["date"]).dt.normalize())
    cands = bars[bars["date"] <= limit].sort_values("date", ascending=False)
    if exact:
        cands = cands[cands["date"] == limit]
    for _, row in cands.head(lookback).iterrows():
        close = row["close"]
        try:
            close = float(close)
        except (TypeError, ValueError):
            continue
        if not (close > 0 and close != float("inf")):
            continue
        factor = float(factors.get(row["date"], 1.0))
        expected = recorded * factor
        if abs(close / expected - 1) <= BASIS_TOLERANCE * 2:
            return row["date"].date(), close, factor
    return None


def price_moved(recorded, stored) -> bool:
    """A stored close no longer equal to the one a prediction row recorded,
    beyond float noise. Either side missing means nothing to say."""
    if recorded is None or stored is None:
        return False
    try:
        recorded, stored = float(recorded), float(stored)
    except (TypeError, ValueError):
        return False
    if not (recorded > 0 and stored > 0):
        return False
    return abs(stored / recorded - 1) > BASIS_TOLERANCE
