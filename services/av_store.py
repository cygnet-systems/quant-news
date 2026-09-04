"""Stored Alpha Vantage intelligence: congressional trades, insider Form 4s,
the Congress roster.

Why this is a store and not a fetch-per-run: both trade endpoints return the
symbol's ENTIRE history on every call and ignore date parameters (NVDA came
back with 408 congressional trades to 2016 and 6,920 Form 4 lines to 2003),
and POLITICIAN_METADATA always returns the whole 1,144-member roster whatever
you ask it for. Re-downloading that per symbol per run would burn the shared
70-calls/minute quota on data that has not changed since the last time. A
symbol is synced once, topped up weekly by the ``av_refresh`` job, and every
run filters the stored rows locally.

POINT-IN-TIME. Every row carries ``visible_from``, the date the disclosure
became public, and every read here filters ``visible_from <= as_of``. The two
sources get there differently:

* A congressional trade is public from its FILING: ``filed_date``, falling
  back to ``notification_date`` (Senate rows often carry no filed date). A row
  with neither is stored with ``visible_from`` NULL and is never served,
  because ``NULL <= as_of`` is not true.
* An insider Form 4 payload carries NO filing date at all, so visibility is a
  PROXY: the second trading day strictly after the transaction, which is the
  SEC filing deadline. Real filings often land sooner; using the deadline can
  hide a row a reader could have seen, which is the safe direction, and never
  reveals one they could not.

The unfiltered rows are reachable only through the sync functions. There is
no read helper without ``as_of``, deliberately.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import lru_cache
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, select

from db.session import get_session
from services.alpha_vantage import AlphaVantageUnavailable, fetch

logger = logging.getLogger(__name__)

CONGRESS_FUNCTION = "CONGRESS_TRADES"
INSIDER_FUNCTION = "INSIDER_TRANSACTIONS"
ROSTER_FUNCTION = "POLITICIAN_METADATA"
# av_fetch_log subject for the endpoints that take no symbol.
ALL_SUBJECTS = "ALL"

# How stale each source may be before a run spends a call. Disclosure lags
# the underlying trade by weeks either way, so a few days of staleness costs
# nothing a same-day fetch would have bought.
MAX_AGE_DAYS = {
    CONGRESS_FUNCTION: 7,
    INSIDER_FUNCTION: 7,
    ROSTER_FUNCTION: 30,
}

CONGRESS_WINDOW_DAYS = 180
INSIDER_WINDOW_DAYS = 180

# Honorifics the vendor mixes into name variants ("Hon. Dan Newhouse").
_HONORIFIC = re.compile(r"^(hon|mr|mrs|ms|dr|sen|rep|the honorable)\.?\s+")
_MISSING = "~"
# Every dollar column in these tables holds two decimals.
_CENTS = Decimal("0.01")
# Fetch stamps are shown next to report dates, which are market dates. A UTC
# stamp reads as tomorrow from mid-evening on.
_DISPLAY_TZ = ZoneInfo("US/Eastern")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _iso(value) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except (TypeError, ValueError):
        return None


def _dec(value) -> Optional[Decimal]:
    if value in (None, "", "None"):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _f(value) -> Optional[float]:
    """Decimal columns out of the database, into a float the prompt
    formatters and JSON can carry."""
    return float(value) if value is not None else None


def _text(value, limit: int) -> Optional[str]:
    s = str(value).strip() if value not in (None, "") else ""
    return s[:limit] or None


def _natural_key(*parts) -> str:
    """A stable digest of the identifying fields, with a sentinel for each
    missing one.

    A composite UNIQUE over the source columns cannot do this job: several
    are nullable and Postgres treats NULLs as distinct, so the same
    disclosure would insert again on every weekly refresh. Components are
    normalised (dates to ISO, amounts to a fixed scale) so that a vendor
    writing "15001" this week and "15001.00" next week is still one row.
    """
    joined = "|".join(_MISSING if p is None or p == "" else str(p)
                      for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _amount_part(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else f"{value:.2f}"


def _share_part(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else f"{value:.4f}"


def _money(value: Decimal) -> Decimal:
    """A dollar amount at the scale of the columns that hold one, rounded
    the way Postgres rounds on the way in (half away from zero)."""
    return value.quantize(_CENTS, rounding=ROUND_HALF_UP)


def alias_key(name: Optional[str]) -> Optional[str]:
    """The normalised form both sides of a name match use: lowercased,
    honorific dropped, punctuation stripped, whitespace collapsed."""
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = _HONORIFIC.sub("", s)
    s = re.sub(r"[^a-z0-9 '\-]", "", s).strip()
    return s[:128] or None


# ---------------------------------------------------------------------------
# Point-in-time visibility
# ---------------------------------------------------------------------------

def congress_visible_from(filed_date: Optional[date],
                          notification_date: Optional[date]) -> Optional[date]:
    """The date a STOCK Act disclosure became public.

    Filing first, notification second, None when the vendor supplied
    neither. Both congressional readers -- the counts block in
    political_service and the dossier -- filter on the column this writes,
    so this precedence is the one definition of "public" in the system.
    """
    return filed_date or notification_date


@lru_cache(maxsize=8192)
def _second_trading_day_after(d: date) -> date:
    from utils.trading_calendar import get_next_trading_day

    return get_next_trading_day(get_next_trading_day(d))


def insider_visible_from(transaction_date: date) -> date:
    """PROXY visibility for a Form 4: the second trading day strictly after
    the transaction, the SEC deadline for filing one.

    The payload carries no filing date, so this is the only date available.
    It is a deadline, not an observation: an insider who files on day one is
    treated here as invisible until day two.
    """
    return _second_trading_day_after(transaction_date)


# ---------------------------------------------------------------------------
# Fetch bookkeeping
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _insert_ignore(session, table, rows: list[dict]) -> None:
    """Insert rows, skipping any a concurrent writer got in first.

    The lock in ``ensure_fresh`` only serialises threads in one process, and
    the analysis subprocess is a second one: two syncs of the same subject
    can overlap, and a plain INSERT of a key the other just wrote aborts the
    whole transaction. Losing that race has to be a no-op, not a lost
    evidence block.
    """
    if not rows:
        return
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as _ins
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _ins
    else:
        session.execute(table.insert(), rows)
        return
    session.execute(_ins(table).on_conflict_do_nothing(), rows)


def _record_fetch(function: str, subject: str, rows_seen: int, ok: bool,
                  detail: Optional[str] = None) -> None:
    """Stamp the attempt. ``last_success_at`` only moves on a success, so a
    failed top-up cannot make the store look freshly synced."""
    from db.models import AvFetchLog

    now = _now()
    values = {"last_fetched_at": now, "rows_seen": rows_seen, "ok": ok,
              "detail": (detail or "")[:500] or None}
    if ok:
        values["last_success_at"] = now
    with get_session() as session:
        # Seed-then-update rather than get-then-add: the seed is a no-op for
        # the loser of a cross-process race and the update is the same
        # statement either way.
        _insert_ignore(session, AvFetchLog.__table__,
                       [{"function": function, "subject": subject, "ok": ok}])
        session.execute(
            AvFetchLog.__table__.update()
            .where(and_(AvFetchLog.__table__.c.function == function,
                        AvFetchLog.__table__.c.subject == subject))
            .values(**values))


def last_fetch(function: str, subject: str) -> Optional[dict]:
    """The av_fetch_log row for one (function, subject), or None."""
    from db.models import AvFetchLog

    with get_session() as session:
        row = session.get(AvFetchLog, (function, subject))
        if row is None:
            return None
        return {"function": row.function, "subject": row.subject,
                "last_fetched_at": row.last_fetched_at,
                "last_success_at": row.last_success_at,
                "rows_seen": row.rows_seen, "ok": row.ok,
                "detail": row.detail}


def last_synced_date(function: str, subject: str) -> Optional[str]:
    """The market date of the last SUCCESSFUL fetch, for a block that wants
    to date what it is serving.

    Deliberately not ``last_fetched_at``: that advances on failed attempts
    too, so a header reading from it would claim the store was synced today
    by the very top-up that just failed.
    """
    when = (last_fetch(function, subject) or {}).get("last_success_at")
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(_DISPLAY_TZ).date().isoformat()


def _age_days(when: Optional[datetime]) -> Optional[float]:
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (_now() - when).total_seconds() / 86400.0


SYNCS = {
    CONGRESS_FUNCTION: lambda subject: sync_congress_trades(subject),
    INSIDER_FUNCTION: lambda subject: sync_insider_transactions(subject),
    ROSTER_FUNCTION: lambda subject: sync_politicians(),
}

_SYNC_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_SYNC_LOCKS_GUARD = threading.Lock()


def _sync_lock(function: str, subject: str) -> threading.Lock:
    with _SYNC_LOCKS_GUARD:
        return _SYNC_LOCKS.setdefault((function, subject), threading.Lock())


def _fresh_reason(previous: Optional[dict], max_age_days: float
                  ) -> Optional[str]:
    """Why the stored copy still answers, or None when it does not. A failed
    attempt is not freshness, however recent it is."""
    if not previous or not previous["ok"]:
        return None
    age = _age_days(previous["last_fetched_at"])
    if age is None or age > max_age_days:
        return None
    return f"fetched {age:.1f}d ago"


def ensure_fresh(function: str, subject: str,
                 max_age_days: Optional[float] = None) -> dict:
    """Sync ``(function, subject)`` only when the stored copy is missing,
    stale, or was last written by a failed attempt.

    Returns {"function", "subject", "fetched", "rows", "reason"}. A failure
    is reported, not raised: this is what a run calls before reading, and a
    throttled top-up should leave the run on last week's rows rather than
    take the run down with it. ``fetched`` says whether a call was actually
    spent.
    """
    if function not in SYNCS:
        raise ValueError(f"no sync registered for {function!r}")
    subject = (subject or ALL_SUBJECTS).upper()
    if max_age_days is None:
        max_age_days = MAX_AGE_DAYS[function]

    previous = last_fetch(function, subject)
    reason = _fresh_reason(previous, max_age_days)
    if reason:
        return {"function": function, "subject": subject, "fetched": False,
                "rows": 0, "reason": reason}

    # Research fans out one thread per symbol and every one of them wants the
    # same roster on a cold database. Unserialised, the first run spends a
    # POLITICIAN_METADATA call per symbol and each loser of the race dies on
    # the keys the winner has just written.
    with _sync_lock(function, subject):
        previous = last_fetch(function, subject)
        reason = _fresh_reason(previous, max_age_days)
        if reason:
            return {"function": function, "subject": subject, "fetched": False,
                    "rows": 0, "reason": f"{reason} (concurrent refresh)"}
        try:
            rows = SYNCS[function](subject)
        except Exception as e:
            # Every failure degrades the same way. A rate-limiter timeout, a
            # transport error and a payload the parser could not use are all
            # "no top-up this run", never a reason to lose a block the stored
            # rows could still write.
            try:
                _record_fetch(function, subject, 0, False, str(e))
            except Exception:
                logger.warning("av_store: could not log the failed fetch of "
                               "%s %s", function, subject, exc_info=True)
            logger.warning("av_store: %s %s unavailable: %s", function,
                           subject, e)
            detail = str(e)[:120] or type(e).__name__
            return {"function": function, "subject": subject, "fetched": False,
                    "rows": 0, "reason": f"unavailable: {detail}"}
    return {"function": function, "subject": subject, "fetched": True,
            "rows": rows, "reason": "stale" if previous else "missing"}


# ---------------------------------------------------------------------------
# Sync: congressional trades
# ---------------------------------------------------------------------------

def _congress_rows(symbol: str, trades: Iterable[dict]) -> dict[str, dict]:
    """Payload rows keyed by natural key, deduplicated: the vendor does
    repeat identical lines, and two rows with one key would make the upsert
    non-idempotent."""
    out: dict[str, dict] = {}
    for t in trades or []:
        tx = _iso(t.get("transaction_date"))
        filed = _iso(t.get("filed_date"))
        notified = _iso(t.get("notification_date"))
        amount_min = _dec(t.get("amount_min"))
        amount_max = _dec(t.get("amount_max"))
        bioguide = _text(t.get("bioguide_id"), 16)
        owner = _text(t.get("owner_code"), 16)
        key = _natural_key(bioguide, symbol, tx.isoformat() if tx else None,
                           _text(t.get("transaction_type"), 8),
                           _amount_part(amount_min), _amount_part(amount_max),
                           owner, filed.isoformat() if filed else None)
        out[key] = {
            "natural_key": key,
            "symbol": symbol,
            "bioguide_id": bioguide,
            "politician_canonical": _text(
                t.get("politician_canonical") or t.get("politician"), 200),
            "chamber": _text(t.get("chamber"), 16),
            "party": _text(t.get("party"), 8),
            "state": _text(t.get("state"), 8),
            "state_district": _text(t.get("state_district"), 16),
            "asset_name": _text(t.get("asset_name"), 500),
            "asset_type_code": _text(t.get("asset_type_code"), 8),
            "transaction_type": _text(t.get("transaction_type"), 8),
            "transaction_date": tx,
            "notification_date": notified,
            "filed_date": filed,
            "amount_min": amount_min,
            "amount_max": amount_max,
            "owner_code": owner,
            "filing_status": _text(t.get("filing_status"), 16),
            "visible_from": congress_visible_from(filed, notified),
        }
    return out


_CONGRESS_MUTABLE = ("politician_canonical", "chamber", "party", "state",
                     "state_district", "asset_name", "asset_type_code",
                     "notification_date", "filed_date", "filing_status",
                     "visible_from", "bioguide_id")


def sync_congress_trades(symbol: str) -> int:
    """Store every disclosed congressional trade in ``symbol``.

    One call, full history. Returns the number of rows inserted or changed;
    a second identical sync returns 0 and leaves the row count alone.
    """
    from db.models import CongressTrade

    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("sync_congress_trades needs a symbol")

    data = fetch(CONGRESS_FUNCTION, symbol=symbol)
    incoming = _congress_rows(symbol, data.get("trades") or [])
    written = _upsert(CongressTrade, symbol, incoming, _CONGRESS_MUTABLE)
    _record_fetch(CONGRESS_FUNCTION, symbol, len(incoming), True,
                  f"{written} row(s) written")
    return written


# ---------------------------------------------------------------------------
# Sync: insider transactions
# ---------------------------------------------------------------------------

def _insider_rows(symbol: str, rows: Iterable[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows or []:
        tx = _iso(r.get("transaction_date"))
        if tx is None:
            # Without a transaction date there is no visibility date, and a
            # row that cannot be placed in time cannot be served.
            continue
        shares = _dec(r.get("shares"))
        price = _dec(r.get("share_price"))
        executive = _text(r.get("executive"), 200)
        security = _text(r.get("security_type"), 200)
        disposal = _text(r.get("acquisition_or_disposal"), 2)
        key = _natural_key(symbol, executive, tx.isoformat(), security,
                           disposal, _share_part(shares), _share_part(price))
        # A grant, gift or option exercise comes through at price 0. Any
        # dollar figure derived from those rows is invented, so there is not
        # one: value_usd stays NULL and the reader says so.
        #
        # Quantized to the scale of the column it lands in. A fractional
        # share count times a four-decimal price carries more precision than
        # Numeric(24, 2) can hold, and the unrounded product would never
        # equal the rounded value read back, so _upsert would see every one
        # of those rows as changed on every weekly re-sync.
        value = (_money(shares * price)
                 if shares is not None and price is not None and price > 0
                 else None)
        out[key] = {
            "natural_key": key,
            "symbol": symbol,
            "executive": executive,
            "executive_title": _text(r.get("executive_title"), 200),
            "security_type": security,
            "acquisition_or_disposal": disposal,
            "transaction_date": tx,
            "shares": shares,
            "share_price": price,
            "value_usd": value,
            "visible_from": insider_visible_from(tx),
        }
    return out


_INSIDER_MUTABLE = ("executive_title", "value_usd", "visible_from")


def sync_insider_transactions(symbol: str) -> int:
    """Store every disclosed Form 4 line for ``symbol``. One call, full
    history. Returns rows inserted or changed."""
    from db.models import InsiderTransaction

    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("sync_insider_transactions needs a symbol")

    data = fetch(INSIDER_FUNCTION, symbol=symbol)
    incoming = _insider_rows(symbol, data.get("data") or [])
    written = _upsert(InsiderTransaction, symbol, incoming, _INSIDER_MUTABLE)
    _record_fetch(INSIDER_FUNCTION, symbol, len(incoming), True,
                  f"{written} row(s) written")
    return written


def _upsert(model, symbol: str, incoming: dict[str, dict],
            mutable: tuple[str, ...]) -> int:
    """Insert the rows this symbol does not have and refresh the fields the
    vendor can revise (a late filing date, a corrected title). Existing keys
    are read in one indexed query on symbol, so a 6,900-row re-sync is one
    SELECT and no writes."""
    if not incoming:
        return 0
    written = 0
    with get_session() as session:
        existing = {
            row.natural_key: row
            for row in session.execute(
                select(model).where(model.symbol == symbol)).scalars()
        }
        now = _now()
        fresh: list[dict] = []
        for key, values in incoming.items():
            row = existing.get(key)
            if row is None:
                fresh.append({**values, "fetched_at": now})
                written += 1
                continue
            changed = [f for f in mutable
                       if getattr(row, f) != values.get(f)]
            if changed:
                for field in changed:
                    setattr(row, field, values.get(field))
                row.fetched_at = now
                written += 1
        _insert_ignore(session, model.__table__, fresh)
    return written


# ---------------------------------------------------------------------------
# Sync: the Congress roster
# ---------------------------------------------------------------------------

def sync_politicians() -> int:
    """Store the full roster and its alias table.

    POLITICIAN_METADATA ignores every parameter and always returns all 1,144
    members, so this is one call for everybody, not one per name. Returns the
    number of members written.
    """
    from db.models import Politician, PoliticianAlias

    data = fetch(ROSTER_FUNCTION)
    members = data.get("politicians") or data.get("data") or []
    if isinstance(members, dict):
        members = list(members.values())

    parsed: dict[str, dict] = {}
    for m in members:
        bioguide = _text(m.get("bioguide_id"), 16)
        if not bioguide:
            continue
        display = _text(m.get("display_name"), 200)
        aliases = {alias_key(a) for a in (m.get("aliases") or [])}
        aliases.add(alias_key(display))
        parsed[bioguide] = {
            "display_name": display,
            "chamber": _text(m.get("chamber"), 8),
            "state": _text(m.get("state"), 4),
            "district": _text(m.get("district"), 8),
            "party": _text(m.get("party"), 4),
            "aliases": sorted(a for a in aliases if a),
            "terms": m.get("terms") or None,
        }
    if not parsed:
        # A payload that yields nobody is a failure, not an empty roster, and
        # it has to raise: reported as a success it would leave alias_index()
        # empty week after week while the refresh job mailed "0 members, all
        # good". The envelope key is named so a vendor rename is diagnosable
        # from the log line alone.
        raise AlphaVantageUnavailable(
            f"{ROSTER_FUNCTION} returned no members; payload keys were "
            f"{sorted(str(k) for k in data)[:6]}")

    now = _now()
    with get_session() as session:
        stored = {row.bioguide_id: row for row in
                  session.execute(select(Politician)).scalars()}
        fresh: list[dict] = []
        for bioguide, values in parsed.items():
            row = stored.get(bioguide)
            if row is None:
                fresh.append({"bioguide_id": bioguide, "synced_at": now,
                              **values})
                continue
            for field, value in values.items():
                setattr(row, field, value)
            row.synced_at = now
        _insert_ignore(session, Politician.__table__, fresh)

        # The alias table is the denormalised copy the news matcher reads.
        # Rebuilt per member rather than diffed: a member's variants change
        # only when the roster does, and a stale alias would attribute an
        # article to the wrong person.
        have = set(session.execute(
            select(PoliticianAlias.alias, PoliticianAlias.bioguide_id)).all())
        wanted = {(a, b) for b, v in parsed.items() for a in v["aliases"]}
        _insert_ignore(session, PoliticianAlias.__table__,
                       [{"alias": a, "bioguide_id": b}
                        for a, b in sorted(wanted - have)])
        for alias, bioguide in have - wanted:
            if bioguide in parsed:
                session.execute(
                    PoliticianAlias.__table__.delete().where(and_(
                        PoliticianAlias.alias == alias,
                        PoliticianAlias.bioguide_id == bioguide)))

    _record_fetch(ROSTER_FUNCTION, ALL_SUBJECTS, len(parsed), True,
                  f"{len(parsed)} member(s)")
    return len(parsed)


# ---------------------------------------------------------------------------
# Point-in-time reads
# ---------------------------------------------------------------------------

def _as_of(value) -> date:
    d = _iso(value)
    if d is None:
        raise ValueError(f"as_of is required and must be a date: {value!r}")
    return d


def _congress_dict(row) -> dict:
    return {
        "symbol": row.symbol,
        "bioguide_id": row.bioguide_id,
        "politician": row.politician_canonical,
        "party": row.party,
        "chamber": row.chamber,
        "state": row.state_district or row.state,
        "type": (row.transaction_type or "").upper(),
        "owner": row.owner_code,
        "asset_name": row.asset_name,
        "transaction_date": row.transaction_date.isoformat()
                            if row.transaction_date else None,
        "filed_date": row.visible_from.isoformat() if row.visible_from else None,
        "amount_min": _f(row.amount_min),
        "amount_max": _f(row.amount_max),
    }


def congress_trades_for(symbol: str, as_of, days: int = CONGRESS_WINDOW_DAYS
                        ) -> list[dict]:
    """Trades in ``symbol`` public on or before ``as_of`` whose transaction
    falls in the preceding ``days``, newest transaction first."""
    from db.models import CongressTrade

    as_of_d = _as_of(as_of)
    floor = as_of_d - timedelta(days=days)
    stmt = (select(CongressTrade)
            .where(CongressTrade.symbol == (symbol or "").upper(),
                   CongressTrade.visible_from <= as_of_d,
                   CongressTrade.transaction_date >= floor)
            .order_by(CongressTrade.transaction_date.desc(),
                      CongressTrade.id.desc()))
    with get_session() as session:
        return [_congress_dict(r) for r in session.execute(stmt).scalars()]


def congress_trades_count(symbol: str, as_of) -> int:
    """How many disclosures for ``symbol`` were public by ``as_of``, over the
    whole stored history. The denominator behind "none in this window": for a
    small cap with nothing on record at all, silence means nothing."""
    from db.models import CongressTrade

    as_of_d = _as_of(as_of)
    stmt = (select(func.count())
            .select_from(CongressTrade)
            .where(CongressTrade.symbol == (symbol or "").upper(),
                   CongressTrade.visible_from <= as_of_d))
    with get_session() as session:
        return int(session.execute(stmt).scalar() or 0)


def congress_trades_by_politician(bioguide_id: str, as_of,
                                  limit: int = 25) -> list[dict]:
    """Every symbol one member has been disclosed trading, public on or
    before ``as_of``. The cross-symbol view behind "this member has been
    trading the sector, not just this name"."""
    from db.models import CongressTrade

    as_of_d = _as_of(as_of)
    stmt = (select(CongressTrade)
            .where(CongressTrade.bioguide_id == bioguide_id,
                   CongressTrade.visible_from <= as_of_d)
            .order_by(CongressTrade.transaction_date.desc(),
                      CongressTrade.id.desc())
            .limit(limit))
    with get_session() as session:
        return [_congress_dict(r) for r in session.execute(stmt).scalars()]


def insider_transactions_for(symbol: str, as_of,
                             days: int = INSIDER_WINDOW_DAYS) -> list[dict]:
    """Form 4 lines for ``symbol`` whose filing deadline had passed by
    ``as_of``, transacted within the preceding ``days``.

    value_usd is None on grants, gifts and option exercises (share_price 0):
    a caller summing dollar flow must skip those rows rather than treat them
    as zero-dollar trades.
    """
    from db.models import InsiderTransaction

    as_of_d = _as_of(as_of)
    floor = as_of_d - timedelta(days=days)
    stmt = (select(InsiderTransaction)
            .where(InsiderTransaction.symbol == (symbol or "").upper(),
                   InsiderTransaction.visible_from <= as_of_d,
                   InsiderTransaction.transaction_date >= floor)
            .order_by(InsiderTransaction.transaction_date.desc(),
                      InsiderTransaction.id.desc()))
    with get_session() as session:
        return [{
            "symbol": r.symbol,
            "executive": r.executive,
            "title": r.executive_title,
            "security_type": r.security_type,
            "side": r.acquisition_or_disposal,
            "transaction_date": r.transaction_date.isoformat()
                                if r.transaction_date else None,
            "visible_from": r.visible_from.isoformat(),
            "shares": _f(r.shares),
            "share_price": _f(r.share_price),
            "value_usd": _f(r.value_usd),
        } for r in session.execute(stmt).scalars()]


def politician(bioguide_id: str) -> Optional[dict]:
    """One roster member by id."""
    from db.models import Politician

    with get_session() as session:
        row = session.get(Politician, bioguide_id)
        return _politician_dict(row) if row else None


def politician_by_alias(name: str) -> Optional[dict]:
    """The member a name found in an article refers to, or None.

    None also when the alias matches more than one member: attributing a
    trade to the wrong member of Congress is worse than not attributing it.
    """
    from db.models import Politician, PoliticianAlias

    key = alias_key(name)
    if not key:
        return None
    with get_session() as session:
        ids = session.execute(
            select(PoliticianAlias.bioguide_id)
            .where(PoliticianAlias.alias == key)).scalars().all()
        if len(ids) != 1:
            if ids:
                logger.debug("av_store: alias %r matches %d members", key,
                             len(ids))
            return None
        row = session.get(Politician, ids[0])
        return _politician_dict(row) if row else None


def matchable_alias(alias: Optional[str]) -> bool:
    """Whether an alias is specific enough to match against running prose.

    It needs two tokens of at least two characters. That is not fussiness:
    the vendor stores initial forms, and the roster's 1,144 members put 69
    aliases beginning with the bare article "a" into the table ("a green",
    "a gray", "a king"). Scanned over article text those match ordinary
    English -- "regulators gave the deal a green light" would name Al Green
    of Texas as a sitting member appearing in the story -- and the report
    prompt then asserts it as fact. A bare surname is dropped for the
    milder version of the same reason: "Kelly said" identifies nobody.

    Measured on the live roster: 2,153 of 3,246 aliases survive this and no
    member loses every alias they have.
    """
    return sum(1 for token in (alias or "").split() if len(token) >= 2) >= 2


def alias_index() -> dict[str, str]:
    """Every unambiguous, matchable alias -> bioguide_id, for scanning
    article text.

    An alias resolving to more than one member is dropped for the reason
    ``politician_by_alias`` gives: naming the wrong member of Congress is
    worse than naming none. One too generic to be a name is dropped by
    ``matchable_alias``.
    """
    from db.models import PoliticianAlias

    with get_session() as session:
        rows = session.execute(select(PoliticianAlias.alias,
                                      PoliticianAlias.bioguide_id)).all()
    owners: dict[str, set] = {}
    for alias, bioguide in rows:
        if matchable_alias(alias):
            owners.setdefault(alias, set()).add(bioguide)
    return {alias: next(iter(ids)) for alias, ids in owners.items()
            if len(ids) == 1}


def _politician_dict(row) -> dict:
    return {"bioguide_id": row.bioguide_id, "display_name": row.display_name,
            "chamber": row.chamber, "state": row.state,
            "district": row.district, "party": row.party,
            "aliases": list(row.aliases or []), "terms": row.terms}


# ---------------------------------------------------------------------------
# The weekly job body
# ---------------------------------------------------------------------------

def watchlist_symbols() -> list[str]:
    """Every symbol the scheduled analysis jobs run, so the refresh tops up
    exactly the names a report will ask about."""
    from db.models import ScheduledJob

    symbols: list[str] = []
    with get_session() as session:
        rows = session.execute(
            select(ScheduledJob.symbols_csv)
            .where(ScheduledJob.kind == "analysis")).scalars().all()
    for csv_value in rows:
        for part in (csv_value or "").split(","):
            sym = part.strip().upper()
            if sym and sym not in symbols:
                symbols.append(sym)
    return symbols


def refresh_all(symbols: Iterable[str]) -> dict:
    """The av_refresh job: roster once, then congress and insider top-ups
    for each symbol that is missing or stale.

    Never raises. A symbol the vendor would not serve is counted in
    ``failed`` and named in ``problems``; the rest of the watchlist still
    gets refreshed, and the stored rows from last week keep answering.
    """
    summary = {"politicians": 0, "symbols": 0, "congress_rows": 0,
               "insider_rows": 0, "calls": 0, "skipped": 0, "failed": 0,
               "problems": []}

    roster = ensure_fresh(ROSTER_FUNCTION, ALL_SUBJECTS)
    summary["politicians"] = roster["rows"]
    summary["calls"] += int(roster["fetched"])
    if not roster["fetched"] and roster["reason"].startswith("unavailable"):
        summary["failed"] += 1
        summary["problems"].append(f"roster: {roster['reason']}")

    for symbol in symbols:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            continue
        summary["symbols"] += 1
        for function, bucket in ((CONGRESS_FUNCTION, "congress_rows"),
                                 (INSIDER_FUNCTION, "insider_rows")):
            try:
                result = ensure_fresh(function, symbol)
            except Exception as e:
                # A transport error or a bad payload for one symbol must not
                # end the refresh for the other nineteen.
                summary["failed"] += 1
                summary["problems"].append(f"{symbol} {function}: {str(e)[:120]}")
                logger.warning("av_refresh: %s %s failed: %s", symbol,
                               function, e)
                continue
            summary[bucket] += result["rows"]
            if result["fetched"]:
                summary["calls"] += 1
            elif result["reason"].startswith("unavailable"):
                summary["failed"] += 1
                summary["problems"].append(f"{symbol} {function}: "
                                           f"{result['reason']}")
            else:
                summary["skipped"] += 1
    return summary
