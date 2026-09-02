"""Local symbol lookup behind the Run dialog typeahead.

The dialog used to take free text: a typo ran the whole pipeline and failed
at the price fetch. The `tickers` table answers "is this a name, what is it
called" from a local cache instead of a vendor call per keystroke. It is
filled three ways, and the row's `source` says which and how far to trust
it, highest first:

    index      S&P 500 / Russell 2000 constituent lists, refreshed weekly
    validated  an unknown name that one price lookup accepted
    run        a symbol some run used (from history, or the run path)

An upsert never lowers a row's source and only overwrites name/exchange
from an equal-or-better source, so a weekly index refresh cannot be undone
by the next run that mentions the symbol, and a run's bare symbol never
blanks the name the index gave it. Membership tags merge, so a name in
both indexes carries both.

Every seeder returns a count and swallows network failure: a refresh that
cannot reach Wikipedia or iShares this week logs and returns 0, and the
cache keeps last week's rows.
"""

import csv
import logging
import re
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import case, func, or_, select

from db.session import get_session

logger = logging.getLogger(__name__)

SOURCE_RANK = {"run": 0, "validated": 1, "index": 2}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# The product page publishes this path (the older "*.ajax?fileType=csv"
# download now serves the HTML page instead).
IWM_HOLDINGS_URL = ("https://www.ishares.com/us/products/239710/"
                    "ishares-russell-2000-etf/latest-holdings.csv")
FETCH_TIMEOUT = 30
# Wikipedia and iShares both refuse the default requests User-Agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh) quant-news/1.0"}

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,15}$")


def normalize_symbol(symbol: Optional[str]) -> Optional[str]:
    """Upper-cased, stripped, in the vendor's spelling (BRK.B -> BRK-B, the
    form yfinance quotes and every stored prediction already uses).
    None when it cannot be a ticker at all."""
    s = (symbol or "").strip().upper().replace(".", "-")
    return s if _SYMBOL_RE.match(s) else None


def _clean_exchange(value: Optional[str]) -> Optional[str]:
    v = (value or "").strip()
    if not v or v == "-" or v.upper().startswith("NO MARKET"):
        return None
    return v.upper()[:16]


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------

def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search(query: str, limit: int = 12) -> list[dict]:
    """Rows matching `query`: symbol-prefix matches first (shortest symbol
    first, so 'A' lists A before AAPL), then name substring matches by
    name. Case-insensitive; empty query returns nothing."""
    from db.models import Ticker

    q = (query or "").strip()
    if not q or limit <= 0:
        return []
    sym_prefix = _escape_like(q.upper().replace(".", "-")) + "%"
    name_sub = "%" + _escape_like(q) + "%"
    is_prefix = Ticker.symbol.like(sym_prefix, escape="\\")
    # A name that starts with the query ("nvi" -> NVIDIA) outranks one that
    # merely contains it (ENVIRI); with a 12-row cap the alphabetical order
    # alone pushed the obvious answer off the list.
    name_starts = Ticker.name.ilike(_escape_like(q) + "%", escape="\\")
    stmt = (
        select(Ticker.symbol, Ticker.name, Ticker.exchange)
        .where(or_(is_prefix, Ticker.name.ilike(name_sub, escape="\\")))
        # Prefix hits by symbol length then symbol; name hits by name.
        .order_by(case((is_prefix, 0), (name_starts, 1), else_=2),
                  case((is_prefix, func.length(Ticker.symbol)), else_=0),
                  case((is_prefix, Ticker.symbol), else_=Ticker.name),
                  Ticker.symbol)
        .limit(limit)
    )
    with get_session() as session:
        rows = session.execute(stmt).all()
    return [{"symbol": s, "name": n, "exchange": e} for s, n, e in rows]


def get(symbol: str) -> Optional[dict]:
    from db.models import Ticker

    sym = normalize_symbol(symbol)
    if not sym:
        return None
    with get_session() as session:
        row = session.get(Ticker, sym)
        if row is None:
            return None
        return {"symbol": row.symbol, "name": row.name,
                "exchange": row.exchange,
                "membership": list(row.membership or []),
                "source": row.source}


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------

def upsert(symbols_with_meta: Iterable[dict], source: str) -> int:
    """Insert or refresh rows. Each item: {symbol, name?, exchange?,
    membership?}. Returns the number of symbols written (new or updated).

    Idempotent: a second identical call changes nothing but updated_at.
    Precedence: the stored source is never lowered; name and exchange are
    overwritten only by an equal-or-better source, or when the row has
    none; membership tags are unioned.
    """
    from db.models import Ticker

    if source not in SOURCE_RANK:
        raise ValueError(f"unknown ticker source {source!r}")
    incoming: dict[str, dict] = {}
    for item in symbols_with_meta:
        sym = normalize_symbol(item.get("symbol"))
        if not sym:
            continue
        merged = incoming.setdefault(sym, {"name": None, "exchange": None,
                                           "membership": set()})
        if item.get("name"):
            merged["name"] = str(item["name"]).strip()
        if item.get("exchange"):
            merged["exchange"] = _clean_exchange(item["exchange"])
        merged["membership"].update(t for t in (item.get("membership") or [])
                                    if t)
    if not incoming:
        return 0

    now = datetime.now(timezone.utc)
    rank = SOURCE_RANK[source]
    with get_session() as session:
        existing = {
            row.symbol: row for row in session.execute(
                select(Ticker).where(Ticker.symbol.in_(list(incoming)))
            ).scalars()
        }
        for sym, meta in incoming.items():
            row = existing.get(sym)
            if row is None:
                session.add(Ticker(
                    symbol=sym, name=meta["name"], exchange=meta["exchange"],
                    membership=sorted(meta["membership"]) or None,
                    source=source, updated_at=now,
                ))
                continue
            stronger = rank >= SOURCE_RANK.get(row.source, 0)
            if rank > SOURCE_RANK.get(row.source, 0):
                row.source = source
            if meta["name"] and (stronger or not row.name):
                row.name = meta["name"]
            if meta["exchange"] and (stronger or not row.exchange):
                row.exchange = meta["exchange"]
            tags = set(row.membership or []) | meta["membership"]
            if tags != set(row.membership or []):
                row.membership = sorted(tags)
            row.updated_at = now
    return len(incoming)


def ensure_symbols(symbols: Iterable[str], source: str = "run") -> int:
    """The run path: every symbol a run uses is in the cache afterwards,
    as a bare row unless something better is already stored."""
    return upsert(({"symbol": s} for s in symbols), source)


def _fetch_info(symbol: str):
    """The same vendor call that fills stock_info rows. Indirection so the
    tests can stub it."""
    from services.stock_data import get_stock_info
    return get_stock_info(symbol)


def validate_symbol(symbol: str) -> dict:
    """{ok, name, symbol} for a name the cache does not know. A cached row
    answers without a fetch; otherwise one info lookup decides, and a hit
    joins the cache as 'validated' so the next lookup is local."""
    sym = normalize_symbol(symbol)
    if not sym:
        return {"ok": False, "symbol": None, "name": None,
                "reason": "not a ticker"}
    known = get(sym)
    if known is not None:
        return {"ok": True, "symbol": sym, "name": known["name"]}
    try:
        info = _fetch_info(sym)
    except Exception as e:
        logger.info("validate_symbol %s: %s", sym, e)
        return {"ok": False, "symbol": sym, "name": None,
                "reason": "no price data for this symbol"}
    name = (getattr(info, "name", None) or "").strip() or None
    upsert([{"symbol": sym, "name": name}], "validated")
    return {"ok": True, "symbol": sym, "name": name}


# ---------------------------------------------------------------------------
# Seeders
# ---------------------------------------------------------------------------

def _get_text(url: str) -> str:
    import requests
    resp = requests.get(url, headers=_HEADERS, timeout=FETCH_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def fetch_sp500() -> list[dict]:
    """S&P 500 constituents from the Wikipedia table (Symbol, Security)."""
    import io
    import pandas as pd

    html = _get_text(SP500_URL)
    table = pd.read_html(io.StringIO(html), match="Symbol")[0]
    out = []
    for sym, name in zip(table["Symbol"], table["Security"]):
        if normalize_symbol(str(sym)):
            out.append({"symbol": str(sym), "name": str(name),
                        "membership": ["sp500"]})
    return out


def parse_iwm_holdings(text: str) -> list[dict]:
    """Equity rows of the iShares holdings CSV. The file opens with a fund
    preamble; the table starts at the 'Ticker,' header. Cash, money-market,
    futures and unlisted CVR/escrow lines ('-' ticker) are skipped."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("Ticker,")), None)
    if start is None:
        raise ValueError("no 'Ticker,' header in holdings CSV")
    out = []
    for row in csv.DictReader(lines[start:]):
        if (row.get("Asset Class") or "").strip() != "Equity":
            continue
        sym = normalize_symbol(row.get("Ticker"))
        if not sym:
            continue
        out.append({"symbol": sym, "name": (row.get("Name") or "").strip(),
                    "exchange": _clean_exchange(row.get("Exchange")),
                    "membership": ["r2000"]})
    return out


def fetch_r2000() -> list[dict]:
    return parse_iwm_holdings(_get_text(IWM_HOLDINGS_URL))


def seed_from_indexes() -> int:
    """Refresh the index rows. Each list is fetched and written on its own,
    so one source being down does not lose the other. Returns rows written;
    0 means nothing could be fetched."""
    total = 0
    for label, fetch in (("sp500", fetch_sp500), ("r2000", fetch_r2000)):
        try:
            rows = fetch()
        except Exception as e:
            logger.warning("ticker refresh: %s list unavailable: %s", label, e)
            continue
        if not rows:
            logger.warning("ticker refresh: %s list came back empty", label)
            continue
        n = upsert(rows, "index")
        logger.info("ticker refresh: %s %d symbols", label, n)
        total += n
    return total


def seed_from_history() -> int:
    """Every symbol the platform has run: predictions, reports and the
    stock_info cache (which also supplies a name). Source 'run'."""
    from db.models import ModelPrediction, StockInfo, TradingAgentReport

    try:
        with get_session() as session:
            names = dict(session.execute(
                select(StockInfo.symbol, StockInfo.name)).all())
            symbols = set(names)
            for col in (ModelPrediction.symbol, TradingAgentReport.symbol):
                symbols.update(session.execute(select(col).distinct()).scalars())
    except Exception as e:
        logger.warning("ticker refresh: history scan failed: %s", e)
        return 0
    return upsert(({"symbol": s, "name": names.get(s)} for s in symbols),
                  "run")


def refresh() -> dict:
    """The weekly job body: both seeders, counts for the run summary."""
    return {"indexes": seed_from_indexes(), "history": seed_from_history()}
