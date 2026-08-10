"""Read-only client for CygnetResearchTerminal's warmed L2 data cache.

The Terminal's cache warmer (scripts/warm_options_cache.py, nightly post-close)
pre-computes per-symbol put/call summaries (`md_putcall`), ATM option snapshots
(`md_atm_opt`) and `.info` blobs (`md_info`) into a shared L2 — Redis when
REDIS_URL is set there, else its Postgres `data_cache_entries` table. This
module reads those entries so quant-news reuses the precomputed data instead
of re-fetching from Alpha Vantage / yfinance.

Contract (mirrors the Terminal's data_cache.py — do not drift):
  key   = sha256(json.dumps({"ns": namespace, **kwargs}, sort_keys=True,
                            default=str))
  redis = JSON envelope {"ts": epoch, "ns": ..., "p": payload_json_string}
  pg    = data_cache_entries(cache_key, namespace, ts, payload)

Entries are keyed by market session ("YYYY-MM-DD" of the last finalized
session), so no TTL check is needed here — a row either matches the session
asked for or the key misses. Strictly read-only; never raises: any backend
problem degrades to None and the caller's own fetch path.
"""

import hashlib
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

# Defaults match the Terminal's local dev endpoints; override (or set empty
# to disable a tier) in .env for deployments where the warm cache lives
# elsewhere (e.g. Railway Redis).
_REDIS_URL_ENV = "TERMINAL_CACHE_REDIS_URL"
_PG_URL_ENV = "TERMINAL_CACHE_DATABASE_URL"
_DEFAULT_REDIS_URL = "redis://localhost:6379/0"
_DEFAULT_PG_URL = "postgresql+psycopg2://cygnet:dev@localhost:5432/cygnet_dev"

_lock = threading.Lock()
_redis_client = None
_pg_engine = None
_redis_down = False
_pg_down = False


def cache_key(namespace: str, **kwargs) -> str:
    raw = json.dumps({"ns": namespace, **kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _redis_url() -> str:
    return os.environ.get(_REDIS_URL_ENV, _DEFAULT_REDIS_URL)


def _pg_url() -> str:
    return os.environ.get(_PG_URL_ENV, _DEFAULT_PG_URL)


def _get_redis(key: str):
    global _redis_client, _redis_down
    url = _redis_url()
    if not url or _redis_down:
        return None
    try:
        if _redis_client is None:
            with _lock:
                if _redis_client is None:
                    import redis
                    _redis_client = redis.Redis.from_url(
                        url, decode_responses=True,
                        socket_connect_timeout=0.5, socket_timeout=1.0,
                    )
        raw = _redis_client.get(key)
        if raw is None:
            return None
        env = json.loads(raw)
        payload = env.get("p")
        return json.loads(payload) if isinstance(payload, str) else payload
    except Exception as e:
        _redis_down = True   # one warn per process, then silent fallthrough
        logger.warning(f"terminal warm cache (redis) unavailable: {e}")
        return None


def _get_pg(key: str):
    global _pg_engine, _pg_down
    url = _pg_url()
    if not url or _pg_down:
        return None
    try:
        import sqlalchemy as sa
        if _pg_engine is None:
            with _lock:
                if _pg_engine is None:
                    _pg_engine = sa.create_engine(
                        url, pool_pre_ping=True,
                        connect_args={"connect_timeout": 2},
                    )
        with _pg_engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT payload FROM data_cache_entries "
                        "WHERE cache_key = :k"),
                {"k": key},
            ).fetchone()
        if row is None:
            return None
        payload = row[0]
        return json.loads(payload) if isinstance(payload, str) else payload
    except Exception as e:
        _pg_down = True
        logger.warning(f"terminal warm cache (postgres) unavailable: {e}")
        return None


def get(namespace: str, **kwargs):
    """Warm-cache entry for (namespace, kwargs), or None. Redis tier first
    (the Terminal's preferred L2), then the Postgres table."""
    key = cache_key(namespace, **kwargs)
    data = _get_redis(key)
    if data is None:
        data = _get_pg(key)
    return data


def get_putcall_summary(symbol: str, session: str, cutoff_days: int = 90):
    """The warmer's per-symbol put/call summary for a market session:
    {"spot": float, "expiries": [{"expiry", "call_oi", "put_oi", "call_vol",
    "put_vol"}, ...]} covering expirations within `cutoff_days` (warmer
    default 90). None on a miss."""
    data = get("md_putcall", symbol=symbol.upper(), cutoff=cutoff_days,
               session=str(session)[:10])
    return data if data and data.get("expiries") else None


def get_info(symbol: str, session: str):
    """The warmed full `.info` blob for a market session, or None."""
    data = get("md_info", symbol=symbol.upper(), session=str(session)[:10])
    return data or None
