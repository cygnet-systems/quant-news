"""Durable history of the symbol groups a user has actually looked at.

Recording happens on the data store rather than on the raw selection, so a
typo'd ticker that returned nothing never enters history.

Storage keeps every distinct group. The subset-merge that makes the recent
chips readable (dropping REX and REX+WGO once REX+WGO+IOVA exists) is applied
at read time, because collapsing on write would discard exactly the past
searches this table exists to keep.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update

from db.session import get_session

logger = logging.getLogger(__name__)

MAX_SYMBOLS_PER_GROUP = 50


def _current_uid() -> str | None:
    """auth_service.effective_uid: the signed-in user, else the owner a
    scheduled subprocess runs as, the same rule predictions use, so a job's
    watchlist use lands under its owner instead of anonymous."""
    try:
        from services.auth_service import effective_uid
        return effective_uid()
    except Exception:
        return None


def _key(symbols) -> str:
    return ",".join(sorted({s.strip().upper() for s in symbols if s and s.strip()}))


def record_group(symbols: list[str]) -> bool:
    """Record a symbol group, bumping its counters if already seen.

    Returns True if anything was written. Never raises: a failure to record
    history must not break the data-fetch callback that triggers it.
    """
    key = _key(symbols or [])
    if not key:
        return False
    if len(key.split(",")) > MAX_SYMBOLS_PER_GROUP:
        logger.warning("Refusing to record a %d-symbol group",
                       len(key.split(",")))
        return False

    uid = _current_uid()
    try:
        from db.models import WatchlistHistory
        with get_session() as session:
            existing = session.execute(
                select(WatchlistHistory).where(
                    func.coalesce(WatchlistHistory.owner_uid, "") == (uid or ""),
                    WatchlistHistory.symbols_csv == key,
                )
            ).scalar_one_or_none()

            if existing:
                session.execute(
                    update(WatchlistHistory)
                    .where(WatchlistHistory.id == existing.id)
                    .values(last_used_at=datetime.now(timezone.utc),
                            use_count=WatchlistHistory.use_count + 1)
                )
            else:
                session.add(WatchlistHistory(owner_uid=uid, symbols_csv=key))
            session.commit()
        return True
    except Exception as e:
        logger.warning("Could not record watchlist group %s: %s", key, e)
        return False


def _rows(limit: int, symbol: str | None = None, since_days: int | None = None):
    from db.models import WatchlistHistory
    uid = _current_uid()
    with get_session() as session:
        q = (
            select(WatchlistHistory)
            .where(func.coalesce(WatchlistHistory.owner_uid, "") == (uid or ""))
            .order_by(WatchlistHistory.last_used_at.desc())
        )
        if since_days:
            q = q.where(WatchlistHistory.last_used_at
                        >= datetime.now(timezone.utc) - timedelta(days=since_days))
        rows = session.execute(q.limit(limit)).scalars().all()
        return [
            {
                "id": r.id,
                "symbols": r.symbols_csv.split(","),
                "symbols_csv": r.symbols_csv,
                "first_used_at": r.first_used_at.isoformat() if r.first_used_at else None,
                "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
                "use_count": r.use_count,
            }
            for r in rows
            # Symbol filtering in Python: the sets are small and a LIKE on a
            # comma-joined column would match SPY inside SPYG.
            if not symbol or symbol.strip().upper() in r.symbols_csv.split(",")
        ]


def recent_groups(limit: int = 5) -> list[list[str]]:
    """The most recent distinct groups, collapsed for the chip row.

    A group already contained by a more recent, larger one is dropped, so
    building a watchlist up one ticker at a time leaves a single chip rather
    than a trail of partial prefixes.
    """
    # Over-fetch: the merge below can discard most of what it reads.
    rows = _rows(limit=limit * 8)
    kept: list[set] = []
    out: list[list[str]] = []
    for r in rows:
        s = set(r["symbols"])
        if any(s <= k for k in kept):
            continue
        kept.append(s)
        out.append(r["symbols"])
        if len(out) >= limit:
            break
    return out


def search_history(limit: int = 200, symbol: str | None = None,
                   since_days: int | None = None) -> list[dict]:
    """Full, uncollapsed history for browsing on the Activity page."""
    return _rows(limit=limit, symbol=symbol, since_days=since_days)
