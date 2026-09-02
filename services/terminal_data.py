"""Read-only bridge to the Terminal's historical tables.

The CygnetResearchTerminal cron collects, nightly, into its Postgres:
  * finviz_snapshots   — whole-market screener rows (JSONB payload per name)
  * edgar_events       — every 8-K and Form 4 filing (SEC daily index)
  * edgar_form4_tx     — parsed insider transactions with direction/size

This module turns those into POINT-IN-TIME prompt blocks for the research
and synthesis layers: every query filters on the SEC's own filed_date or the
snapshot_date, both of which are stamped at collection time, so a backtest
can never see the future. Strictly read-only and never raises — a missing
table or dead connection degrades to an empty block and the run proceeds
without the evidence (the block's absence is itself visible in the prompt).

Connection: HISTORICAL_DATABASE_URL — the Terminal's historical-data
Postgres (a different database from its warm-cache/Redis tier, hence a
different env var than terminal_cache's). Falls back to
TERMINAL_CACHE_DATABASE_URL for single-database dev setups.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_engine = None
_down = False


def _get_engine():
    global _engine, _down
    if _down:
        return None
    with _lock:
        if _engine is None:
            try:
                from sqlalchemy import create_engine
                url = (os.environ.get("HISTORICAL_DATABASE_URL")
                       or os.environ.get("TERMINAL_HISTORY_DATABASE_URL")
                       or os.environ.get(
                           "TERMINAL_CACHE_DATABASE_URL",
                           "postgresql+psycopg2://cygnet:dev@localhost:5432/"
                           "cygnet_dev"))
                _engine = create_engine(url, pool_pre_ping=True,
                                        pool_size=2, max_overflow=2)
            except Exception as e:
                logger.warning(f"terminal data bridge unavailable: {e}")
                _down = True
    return _engine


def _rows(sql: str, params: dict) -> list:
    eng = _get_engine()
    if eng is None:
        return []
    try:
        from sqlalchemy import text
        with eng.connect() as c:
            return c.execute(text(sql), params).fetchall()
    except Exception as e:
        logger.debug(f"terminal data query failed: {e}")
        return []


# ---------------------------------------------------------------------------
# SEC filings block
# ---------------------------------------------------------------------------

def filings_block(symbol: str, as_of: str, days: int = 45) -> str:
    """Recent 8-Ks + insider transaction summary, filed on or before as_of."""
    eights_all = _rows("""
        SELECT filed_date::text, url FROM edgar_events
        WHERE ticker = :sym AND form = '8-K'
          AND filed_date <= :as_of
          AND filed_date > (CAST(:as_of AS date) - :days)
        ORDER BY filed_date DESC
    """, {"sym": symbol.upper(), "as_of": as_of, "days": days})
    eights = eights_all[:6]

    tx = _rows("""
        SELECT code, count(*), sum(value_usd), bool_or(is_officer)
        FROM edgar_form4_tx t JOIN edgar_events e USING (accession)
        WHERE t.ticker = :sym AND e.filed_date <= :as_of
          AND e.filed_date > (CAST(:as_of AS date) - :days)
          AND code IN ('P', 'S')
        GROUP BY code
    """, {"sym": symbol.upper(), "as_of": as_of, "days": days})

    if not eights and not tx:
        return ""

    lines = [f"[SEC filings — point-in-time through {as_of}]"]
    if eights:
        shown = (f"{len(eights)} most recent of {len(eights_all)} "
                 if len(eights_all) > len(eights) else "")
        lines.append(f"8-K filings (last {days}d, {shown}newest first): "
                     + "; ".join(f"{d}" for d, _ in eights)
                     + " — an 8-K is a material corporate event; a price gap "
                       "near one of these dates likely has a filed cause.")
    else:
        lines.append(f"No 8-K filings in the last {days}d.")

    buys = next((r for r in tx if r[0] == "P"), None)
    sells = next((r for r in tx if r[0] == "S"), None)
    if buys or sells:
        b_n, b_usd = (buys[1], buys[2] or 0) if buys else (0, 0)
        s_n, s_usd = (sells[1], sells[2] or 0) if sells else (0, 0)
        officer_note = (" (incl. officers)" if (buys and buys[3]) else "")
        lines.append(
            f"Insider transactions (Form 4, {days}d): "
            f"{b_n} open-market BUY totaling ${b_usd:,.0f}{officer_note}; "
            f"{s_n} SELL totaling ${s_usd:,.0f}. "
            f"Net: ${b_usd - s_usd:+,.0f}. Open-market buys are the "
            f"informative side — sales are often scheduled/diversification.")
    else:
        lines.append(f"No open-market insider buys or sells in {days}d.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Finviz market-context block
# ---------------------------------------------------------------------------

def _pct(payload: dict, *keys) -> float | None:
    for k in keys:
        v = payload.get(k)
        if v in (None, "", "-"):
            continue
        try:
            return float(str(v).replace("%", "").replace(",", ""))
        except ValueError:
            continue
    return None


def market_context_block(symbol: str, as_of: str) -> str:
    """Whole-market breadth + the symbol's own screener row, as of the
    latest snapshot on or before as_of."""
    day = _rows("""
        SELECT MAX(snapshot_date)::text FROM finviz_snapshots
        WHERE snapshot_date <= :as_of
    """, {"as_of": as_of})
    snap_date = day[0][0] if day and day[0][0] else None
    if not snap_date:
        return ""

    breadth = _rows("""
        SELECT count(*),
               avg(CASE WHEN (payload->>'Change') LIKE '-%' THEN 0.0 ELSE 1.0 END),
               percentile_cont(0.5) WITHIN GROUP
                   (ORDER BY NULLIF(replace(payload->>'Relative Volume', ',', ''), '')::float)
        FROM finviz_snapshots WHERE snapshot_date = :d
    """, {"d": snap_date})

    own = _rows("""
        SELECT payload, sector, industry FROM finviz_snapshots
        WHERE snapshot_date = :d AND ticker = :sym
    """, {"d": snap_date, "sym": symbol.upper()})

    lines = [f"[Market context — finviz screener snapshot {snap_date}]"]
    if breadth and breadth[0][0]:
        n, up_frac, med_rvol = breadth[0]
        lines.append(f"Universe breadth ({n} names): "
                     f"{(up_frac or 0) * 100:.0f}% advanced on the day"
                     + (f"; median relative volume {med_rvol:.2f}"
                        if med_rvol else ""))
    if own:
        p, sector, industry = own[0][0] or {}, own[0][1], own[0][2]
        bits = []
        for label, keys in (
                ("RSI", ("Relative Strength Index (14)", "RSI", "RSI (14)")),
                ("rel volume", ("Relative Volume", "Rel Volume")),
                ("perf week %", ("Performance (Week)", "Perf Week")),
                ("perf month %", ("Performance (Month)", "Perf Month")),
                ("short float %", ("Short Float", "Float Short")),
                ("inst own %", ("Institutional Ownership", "Inst Own"))):
            v = _pct(p, *keys)
            if v is not None:
                bits.append(f"{label} {v:g}")
        if bits:
            lines.append(f"{symbol.upper()} screener row"
                         + (f" ({sector} / {industry})" if sector else "")
                         + ": " + ", ".join(bits))
    return "\n".join(lines) if len(lines) > 1 else ""
