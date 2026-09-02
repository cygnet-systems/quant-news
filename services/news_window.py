"""Point-in-time news windowing, lookahead-safe.

Ported from TradingAgents v0.3.1 (`dataflows/yfinance_news.py`) invariants,
adapted to our Alpha Vantage / yfinance article shapes. The three rules:

1. Normalize every operand to UTC before comparing. A naive value is *assumed*
   UTC (``.replace``); an aware value is *converted* (``.astimezone``). Alpha
   Vantage ``time_published`` is naive-UTC, so "assume UTC" is correct here.
2. Half-open window ``[start, end + 1 day)``: keeps all of the ``end`` day
   (through 23:59:59) but rejects an article stamped exactly midnight-after.
3. An article with **no publish date** is kept only if the window reaches the
   present (``end >= now - 1 day``). A historical/backtest window *excludes*
   undated items: you cannot prove they are not future news.

This is the countermeasure to the coverage-gap/leak class of bug our own
lookahead audits chase: the previous caller-side guard was a naive string
compare (``published_at[:19] <= "{date}T23:59:59"``) in app.py, and the news
was sourced from a rolling 7-day-from-*now* window, so backtests older than a
week silently ran on empty news.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from zoneinfo import ZoneInfo

# One default for every entry point (dialog, CLI, seeded job, library
# fallback): config.MODEL.NEWS_LOOKBACK_DAYS. The dialog said 7 while the
# library said 14, and the report footer quoted the library value for a
# window the dialog had cut in half.

class RunParameterMissing(ValueError):
    """A run parameter the frontend owns (news window, article cap) did not
    arrive. Raised instead of substituting a default: a number the user did
    not choose must never silently shape a run."""


logger = logging.getLogger(__name__)

_ET = ZoneInfo("US/Eastern")


def as_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC-aware; a naive value is assumed to be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_dt(value: Union[str, datetime, None]) -> Optional[datetime]:
    """Parse a YYYY-MM-DD string or pass a datetime through; None on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def in_news_window(
    pub_date: Optional[datetime],
    start_dt: datetime,
    end_dt: datetime,
    *,
    now: Optional[datetime] = None,
) -> bool:
    """Whether an article belongs in the half-open window ``[start, end + 1 day)``.

    ``now`` is injectable for deterministic tests; defaults to real UTC now.
    """
    end = as_utc(end_dt)
    if pub_date is not None:
        return as_utc(start_dt) <= as_utc(pub_date) < end + timedelta(days=1)
    # Undated: keep only if the window reaches the present.
    reference = now if now is not None else datetime.now(timezone.utc)
    return end >= as_utc(reference) - timedelta(days=1)


def _article_pub_date(article) -> Optional[datetime]:
    """Extract published_at from a NewsArticle dataclass or a dict."""
    if hasattr(article, "published_at"):
        return getattr(article, "published_at")
    if isinstance(article, dict):
        val = article.get("published_at") or article.get("published_date")
        if isinstance(val, datetime):
            return val
        if isinstance(val, str) and val:
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                pass
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(val[:19], fmt)
                except ValueError:
                    continue
    return None


def _article_relevance(article) -> float:
    if hasattr(article, "ticker_relevance_score"):
        rel = getattr(article, "ticker_relevance_score")
    elif isinstance(article, dict):
        rel = article.get("ticker_relevance_score")
    else:
        rel = None
    try:
        return float(rel) if rel is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def article_to_dict(a) -> dict:
    """Flatten a NewsArticle (or pass a dict through) to the one shape the
    stores, the models and the prompts read.

    The three copies of this that used to live in app.py and
    analysis_runner.py had already drifted (one assumed ``published_at`` was
    never None); this is the only one now.
    """
    if isinstance(a, dict):
        return a
    published = getattr(a, "published_at", None)
    return {
        "id": getattr(a, "id", None),
        "symbol": getattr(a, "symbol", None),
        "title": getattr(a, "title", ""),
        "source": getattr(a, "source", None),
        "url": getattr(a, "url", None),
        "published_at": (published.isoformat()
                         if isinstance(published, datetime) else published),
        "summary": getattr(a, "summary", None),
        "sentiment": getattr(a, "sentiment", None),
        "sentiment_score": getattr(a, "sentiment_score", None),
        "impact": getattr(a, "impact", None),
        "price_change_percent": getattr(a, "price_change_percent", None),
        "ticker_relevance_score": getattr(a, "ticker_relevance_score", None),
        "topics": getattr(a, "topics", None),
        "overall_sentiment_score": getattr(a, "overall_sentiment_score", None),
        "overall_sentiment_label": getattr(a, "overall_sentiment_label", None),
    }


def article_span(articles: list) -> tuple[Optional[str], Optional[str]]:
    """(oldest, newest) publish dates as YYYY-MM-DD, or (None, None)."""
    dates = [as_utc(pub) for a in articles
             if (pub := _article_pub_date(a)) is not None]
    if not dates:
        return None, None
    return min(dates).strftime("%Y-%m-%d"), max(dates).strftime("%Y-%m-%d")


def sort_newest_first(articles: list) -> list:
    """Stable newest-first order; undated articles sink to the end."""
    floor = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        articles,
        key=lambda a: (as_utc(pub) if (pub := _article_pub_date(a)) is not None
                       else floor),
        reverse=True,
    )


def select_spread(articles: list, n: int) -> list:
    """Pick up to ``n`` articles spread across the time span of ``articles``.

    Every prompt used to take ``articles[:n]`` off a newest-first list, so a
    30-day window fed the model the last three days and silently dropped the
    catalyst three weeks back. This splits the span into ``n`` equal time
    strata, keeps the most relevant article of each (newest on ties), then
    fills any empty strata with the most relevant leftovers. Returned
    newest-first. ``n <= 0`` means no cap.
    """
    if n <= 0 or len(articles) <= n:
        return sort_newest_first(articles)
    dated = [(as_utc(pub), a) for a in articles
             if (pub := _article_pub_date(a)) is not None]
    if not dated:
        return list(articles[:n])
    dated.sort(key=lambda t: t[0])
    lo, hi = dated[0][0], dated[-1][0]
    span = (hi - lo).total_seconds() or 1.0
    strata: list[list] = [[] for _ in range(n)]
    for ts, a in dated:
        idx = min(n - 1, int((ts - lo).total_seconds() / span * n))
        strata[idx].append((ts, a))

    def _rank(t):
        return (_article_relevance(t[1]), t[0])

    chosen: list = []
    for bucket in strata:
        if bucket:
            bucket.sort(key=_rank, reverse=True)
            chosen.append(bucket[0])
    if len(chosen) < n:
        taken = {id(a) for _, a in chosen}
        leftovers = [(ts, a) for ts, a in dated if id(a) not in taken]
        leftovers.sort(key=_rank, reverse=True)
        chosen.extend(leftovers[: n - len(chosen)])
    chosen.sort(key=lambda t: t[0], reverse=True)
    return [a for _, a in chosen]


def cap_newest(articles: list, max_articles: int,
               as_of: Union[str, datetime, None] = None) -> tuple[list, dict]:
    """Keep the newest ``max_articles`` (0 = all) and say what that did.

    The stats dict is what makes the cap visible downstream: ``fetched`` vs
    ``kept``, whether it bit, and the date span actually kept, so a trace
    can print "requested 30d, effective 6d" instead of implying the model
    saw the whole month.
    """
    ordered = sort_newest_first(articles)
    fetched = len(ordered)
    cap = int(max_articles or 0)
    kept = ordered[:cap] if cap and fetched > cap else ordered
    oldest, newest = article_span(kept)
    effective_days = None
    end_dt = _coerce_dt(as_of) if as_of is not None else None
    if oldest and end_dt is not None:
        effective_days = (end_dt.date()
                          - datetime.strptime(oldest, "%Y-%m-%d").date()).days + 1
    return kept, {
        "fetched": fetched,
        "kept": len(kept),
        "cap": cap,
        "capped": len(kept) < fetched,
        "oldest": oldest,
        "newest": newest,
        "effective_days": effective_days,
    }


def filter_articles_as_of(
    articles: list,
    as_of: Union[str, datetime],
    lookback_days: int,
    *,
    now: Optional[datetime] = None,
) -> list:
    """Keep only articles published within ``[as_of - lookback, as_of]``.

    Replaces the fragile string-slice cutoff in the callers. Undated articles
    are dropped for historical ``as_of`` and kept for a live (present) ``as_of``.
    Articles are returned in their original order.
    """
    end_dt = _coerce_dt(as_of)
    if end_dt is None:
        return articles
    start_dt = end_dt - timedelta(days=lookback_days)
    return [
        a for a in articles
        if in_news_window(_article_pub_date(a), start_dt, end_dt, now=now)
    ]


def av_time_bounds(
    as_of: Union[str, datetime],
    lookback_days: int,
) -> tuple[str, str]:
    """Alpha Vantage ``time_from``/``time_to`` (``YYYYMMDDTHHMM``) for a point-in-time
    fetch: from ``as_of - lookback`` at 00:00 through the *close* of ``as_of``.

    The upper bound is the end of the ``as_of`` day because a decision made at
    the close of day T predicting day T+1 may use all of day T's news.
    """
    end_dt = _coerce_dt(as_of) or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    return start_dt.strftime("%Y%m%dT0000"), end_dt.strftime("%Y%m%dT2359")


# Point-in-time fetches are cached per (symbol, as_of, lookback, cap, …) in
# TWO layers: this process's dict, and a diskcache directory shared with
# every other process on the host. The Run dialog's report callback (server
# process) and its model stage (background subprocess) need the same
# window for the same click; without the shared layer each paid the vendor
# once: two passes per symbol per run. The per-key lock makes the second
# process WAIT for the first fetch instead of racing it.
#
# A historical window is immutable; a window ending today (or the overnight
# gap before an open that has not happened yet) keeps growing, so those
# entries expire after LIVE_TTL_S instead of freezing the first fetch.
_PIT_NEWS_CACHE: dict = {}
LIVE_TTL_S = 600.0
HISTORICAL_TTL_S = 24 * 3600.0
_DISK_CACHE_DIR = "cache/news_window"
_disk = None


def _disk_cache():
    global _disk
    if _disk is None:
        import diskcache
        _disk = diskcache.Cache(_DISK_CACHE_DIR)
    return _disk


def clear_pit_news_cache() -> None:
    _PIT_NEWS_CACHE.clear()
    try:
        _disk_cache().clear()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"news disk cache clear failed: {e}")


def _cache_get(key: tuple):
    hit = _PIT_NEWS_CACHE.get(key)
    if hit is not None:
        value, expires_at = hit
        if expires_at is None or time.monotonic() < expires_at:
            return value
        del _PIT_NEWS_CACHE[key]
    try:
        value = _disk_cache().get(repr(key))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"news disk cache read failed: {e}")
        return None
    if value is not None:
        _PIT_NEWS_CACHE[key] = (value, None)
    return value


def _cache_put(key: tuple, value, *, live: bool) -> None:
    ttl = LIVE_TTL_S if live else HISTORICAL_TTL_S
    _PIT_NEWS_CACHE[key] = (value, time.monotonic() + ttl)
    try:
        _disk_cache().set(repr(key), value, expire=ttl)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"news disk cache write failed: {e}")


class _fetch_lock:
    """Cross-process lock per cache key; a no-op when diskcache is unusable."""

    def __init__(self, key: tuple):
        self._lock = None
        try:
            import diskcache
            self._lock = diskcache.Lock(_disk_cache(), f"lock:{key!r}", expire=300)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"news fetch lock unavailable: {e}")

    def __enter__(self):
        if self._lock is not None:
            self._lock.acquire()
        return self

    def __exit__(self, *exc):
        if self._lock is not None:
            try:
                self._lock.release()
            except Exception:  # noqa: BLE001
                pass
        return False


def _window_is_live(end_day: Union[str, datetime, None]) -> bool:
    end = _coerce_dt(end_day)
    return end is None or end.date() >= datetime.now(timezone.utc).date()


def fetch_point_in_time_news(
    symbol: str,
    as_of: Union[str, datetime],
    lookback_days: int,
    max_articles: int = 0,
    relevance_threshold: float = 0.5,
) -> list:
    """Articles only: see :func:`fetch_point_in_time_news_with_stats`.
    ``max_articles`` 0 = everything the window holds (the default for
    library callers; run paths pass the frontend's cap explicitly)."""
    return fetch_point_in_time_news_with_stats(
        symbol, as_of, lookback_days, max_articles, relevance_threshold)[0]


def fetch_point_in_time_news_with_stats(
    symbol: str,
    as_of: Union[str, datetime],
    lookback_days: int,
    max_articles: int = 0,
    relevance_threshold: float = 0.5,
) -> tuple[list, dict]:
    """Fetch ticker news bounded to ``[as_of - lookback, as_of]`` (no lookahead).

    Pushes the window to Alpha Vantage via ``time_from``/``time_to`` and then
    re-applies :func:`filter_articles_as_of` as a belt-and-suspenders guard (a
    vendor's inclusive bound can still leak the midnight-after article). Falls
    back to yfinance news (also as-of filtered) when AV returns nothing.

    ``max_articles`` keeps the newest N of the window (0 = everything the
    window holds). The returned stats (see :func:`cap_newest`) record what
    the cap did so no caller has to infer it.
    """
    max_articles = int(max_articles or 0)

    # max_articles and relevance are part of the key: a smaller earlier fetch
    # must not be served to a caller asking for the full window.
    cache_key = (symbol.upper(), str(as_of)[:10], lookback_days,
                 max_articles, relevance_threshold)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    with _fetch_lock(cache_key):
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        return _fetch_point_in_time_uncached(
            symbol, as_of, lookback_days, max_articles, relevance_threshold,
            cache_key)


def _fetch_point_in_time_uncached(symbol, as_of, lookback_days, max_articles,
                                  relevance_threshold, cache_key) -> tuple[list, dict]:
    end_day = _coerce_dt(as_of).date()
    start_day = end_day - timedelta(days=lookback_days)
    articles = _store_window(symbol, start_day, end_day, relevance_threshold)
    windowed = filter_articles_as_of(articles, as_of, lookback_days)
    result = cap_newest(windowed, max_articles, as_of)
    _cache_put(cache_key, result, live=_window_is_live(as_of))
    return result


# ---------------------------------------------------------------------------
# The durable article store (Postgres historical_news + news_coverage)
# ---------------------------------------------------------------------------
# Every article the vendor returns for a (symbol, day) is kept for
# NEWS_RETENTION_DAYS; a window is served from the store and only the days
# with no coverage row are fetched. A day at the live edge (today, and the
# session before it, the vendor keeps indexing a session's articles for
# hours) is re-fetched when its coverage is older than LIVE_TTL_S.

LIVE_EDGE_DAYS = 2
_last_prune_day: Optional[str] = None


def _prune_once_a_day() -> None:
    global _last_prune_day
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _last_prune_day == today:
        return
    _last_prune_day = today
    try:
        from config import MODEL
        from services.cache_service import get_cache
        get_cache().prune_news(MODEL.NEWS_RETENTION_DAYS)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"news store prune skipped: {e}")


def coverage_gaps(days: list, covered: dict, *, now: datetime,
                  live_ttl_s: float = LIVE_TTL_S,
                  live_edge_days: int = LIVE_EDGE_DAYS) -> list[tuple]:
    """Contiguous (start, end) day ranges that must be fetched.

    A day is missing when it has no coverage row, or when it is within
    ``live_edge_days`` of today and its row is older than ``live_ttl_s``.
    Pure, so it is testable without a database.
    """
    today = now.date()
    need = []
    for d in days:
        fetched = covered.get(d)
        if fetched is None:
            need.append(d)
            continue
        if (today - d).days < live_edge_days:
            age = (now - as_utc(fetched)).total_seconds()
            if age >= live_ttl_s:
                need.append(d)
    ranges: list[tuple] = []
    for d in need:
        if ranges and (d - ranges[-1][1]).days == 1:
            ranges[-1] = (ranges[-1][0], d)
        else:
            ranges.append((d, d))
    return ranges


def _store_window(symbol: str, start_day, end_day, relevance_threshold: float) -> list:
    """Articles for [start_day, end_day] from the store, fetching only the
    uncovered days from the vendor first. Raises NewsUnavailable when a
    needed day cannot be fetched. An incomplete window is not served as a
    complete one."""
    from services.cache_service import get_cache
    from services.news_service import fetch_alpha_vantage_news, fetch_yfinance_news

    _prune_once_a_day()
    cache = get_cache()
    today = datetime.now(timezone.utc).date()
    end_day = min(end_day, today)
    days = [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]
    gaps = coverage_gaps(days, cache.news_coverage(symbol, start_day, end_day),
                         now=datetime.now(timezone.utc))
    for gap_start, gap_end in gaps:
        # Everything the vendor returns is stored (relevance is applied at
        # read time, since callers use different thresholds).
        fetched = fetch_alpha_vantage_news(
            symbol,
            time_from=gap_start.strftime("%Y%m%dT0000"),
            time_to=gap_end.strftime("%Y%m%dT2359"),
            relevance_threshold=0.0,
        )
        if fetched:
            cache.store_historical_news([_article_row(a) for a in fetched])
        covered_days = [gap_start + timedelta(days=i)
                        for i in range((gap_end - gap_start).days + 1)]
        cache.mark_news_coverage(symbol, covered_days)
        logger.info(f"{symbol}: news store fetched {len(fetched)} articles for "
                    f"{gap_start}→{gap_end}")

    rows = cache.get_historical_news(symbol, start_date=start_day.isoformat(),
                                     end_date=end_day.isoformat())
    articles = [_row_to_article(r) for r in rows]
    if relevance_threshold > 0:
        articles = [a for a in articles
                    if a.ticker_relevance_score is not None
                    and a.ticker_relevance_score >= relevance_threshold]
    if not articles and not gaps:
        return articles
    if not articles:
        # The vendor answered with nothing for a freshly fetched window; the
        # yfinance fallback has no server-side window, fetch then filter.
        return fetch_yfinance_news(symbol, max_articles=0)
    return articles


def _article_row(a) -> dict:
    """NewsArticle → the historical_news row shape."""
    pub = getattr(a, "published_at", None)
    return {
        "id": a.id, "symbol": a.symbol, "title": a.title, "summary": a.summary,
        "url": a.url, "source": a.source,
        "published_at": pub,
        "published_date": pub.strftime("%Y-%m-%d") if pub else None,
        "topics": a.topics,
        "overall_sentiment_score": a.overall_sentiment_score,
        "overall_sentiment_label": a.overall_sentiment_label,
        "ticker_sentiment_score": a.sentiment_score,
        "ticker_relevance_score": a.ticker_relevance_score,
    }


def _row_to_article(r: dict):
    """historical_news row → NewsArticle (the shape every consumer reads)."""
    from services.news_service import NewsArticle, _extract_stock_impact
    pub = r.get("published_at")
    if pub is None and r.get("published_date"):
        d = r["published_date"]
        pub = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
    score = r.get("ticker_sentiment_score")
    if score is None:
        sentiment = None
    elif score >= 0.15:
        sentiment = "bullish"
    elif score <= -0.15:
        sentiment = "bearish"
    else:
        sentiment = "neutral"
    return NewsArticle(
        id=r.get("id"), symbol=r.get("symbol"), title=r.get("title") or "",
        source=r.get("source"), url=r.get("url"), published_at=pub,
        summary=r.get("summary"), sentiment=sentiment, sentiment_score=score,
        impact=_extract_stock_impact(r.get("title") or "", r.get("summary") or ""),
        ticker_relevance_score=r.get("ticker_relevance_score"),
        topics=r.get("topics"),
        overall_sentiment_score=r.get("overall_sentiment_score"),
        overall_sentiment_label=r.get("overall_sentiment_label"),
    )


def overnight_window_bounds(
    anchor_date: Union[str, datetime],
    target_date: Union[str, datetime],
    start_time_et: str = "16:00",
    end_time_et: str = "09:30",
) -> tuple[datetime, datetime]:
    """UTC bounds of the overnight gap: anchor close → target open.

    The premise of the overnight filter: by the anchor session's close, that
    day's intraday news is already in the closing price. Only what breaks
    between the close (16:00 ET) and the next session's open (09:30 ET) is
    new information for the close→open move being predicted. Both boundary
    times are parameters, not constants. They are a formula the app owner
    tunes, not a law.
    """
    anchor = _coerce_dt(anchor_date)
    target = _coerce_dt(target_date)
    if anchor is None or target is None:
        raise ValueError(f"unparseable window dates: {anchor_date!r}, {target_date!r}")
    sh, sm = (int(x) for x in start_time_et.split(":"))
    eh, em = (int(x) for x in end_time_et.split(":"))
    start_et = datetime(anchor.year, anchor.month, anchor.day, sh, sm, tzinfo=_ET)
    end_et = datetime(target.year, target.month, target.day, eh, em, tzinfo=_ET)
    return start_et.astimezone(timezone.utc), end_et.astimezone(timezone.utc)


def fetch_overnight_news(
    symbol: str,
    anchor_date: Union[str, datetime],
    target_date: Union[str, datetime],
    relevance_threshold: float = 0.7,
    max_articles: int = 500,
    start_time_et: str = "16:00",
    end_time_et: str = "09:30",
) -> list:
    """Articles only: see :func:`fetch_overnight_news_with_stats`."""
    return fetch_overnight_news_with_stats(
        symbol, anchor_date, target_date, relevance_threshold, max_articles,
        start_time_et, end_time_et)[0]


def fetch_overnight_news_with_stats(
    symbol: str,
    anchor_date: Union[str, datetime],
    target_date: Union[str, datetime],
    relevance_threshold: float = 0.7,
    max_articles: int = 500,
    start_time_et: str = "16:00",
    end_time_et: str = "09:30",
) -> tuple[list, dict]:
    """Fetch only the articles in the overnight gap before ``target_date``.

    Live, the window is additionally clamped to *now*. A pre-open run simply
    sees the overnight news that exists so far. The half-open comparison
    excludes an article stamped exactly at the open. ``max_articles`` as in
    :func:`fetch_point_in_time_news_with_stats` (0 = all).
    """
    start_utc, end_utc = overnight_window_bounds(
        anchor_date, target_date, start_time_et, end_time_et)
    now_utc = datetime.now(timezone.utc)
    end_utc = min(end_utc, now_utc)
    max_articles = int(max_articles or 0)
    if end_utc <= start_utc:
        return cap_newest([], max_articles, anchor_date)

    cache_key = (symbol.upper(), "overnight", str(anchor_date)[:10],
                 str(target_date)[:10], relevance_threshold, max_articles,
                 start_time_et, end_time_et)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    with _fetch_lock(cache_key):
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        return _fetch_overnight_uncached(
            symbol, start_utc, end_utc, now_utc, anchor_date, max_articles,
            relevance_threshold, cache_key)


def _fetch_overnight_uncached(symbol, start_utc, end_utc, now_utc, anchor_date,
                              max_articles, relevance_threshold, cache_key):
    articles = _store_window(symbol, start_utc.date(), end_utc.date(),
                             relevance_threshold)
    # Strict client-side window, unlike the lookback path an undated
    # article can never qualify (its position relative to a 17.5-hour window
    # is unknowable).
    windowed = [
        a for a in articles
        if (pub := _article_pub_date(a)) is not None
        and start_utc <= as_utc(pub) < end_utc
    ]
    result = cap_newest(windowed, max_articles, anchor_date)
    # Live while the gap is still open (the clamp to now trimmed it).
    _cache_put(cache_key, result, live=(end_utc >= now_utc - timedelta(minutes=1)))
    return result


# ---------------------------------------------------------------------------
# The one news fetch every run path uses
# ---------------------------------------------------------------------------

def normalize_lookback(value) -> tuple[bool, int]:
    """(overnight?, lookback_days) from the value the frontend sent.

    Accepts "overnight", an int, or a numeric string. Anything else:
    including None, the value Dash hands a callback whose State component
    is missing: raises RunParameterMissing. There is deliberately no
    fallback: the dialog, the job form and the CLI all carry their own
    default (config NEWS_LOOKBACK_DAYS), so a missing value here means the
    wiring broke, and a run on a window the user did not pick is worse than
    no run.
    """
    if isinstance(value, str) and value.strip().lower() == "overnight":
        return True, 1
    try:
        days = int(value)
    except (TypeError, ValueError):
        raise RunParameterMissing(
            f"news window not supplied by the frontend (got {value!r})") from None
    if days < 1:
        raise RunParameterMissing(f"news window must be ≥ 1 day (got {days})")
    return False, days


def normalize_article_cap(value) -> int:
    """Article cap from the frontend (0 = all). None/blank raises, same
    rule as the window: no number the user did not choose."""
    try:
        cap = int(value)
    except (TypeError, ValueError):
        raise RunParameterMissing(
            f"article cap not supplied by the frontend (got {value!r})") from None
    if cap < 0:
        raise RunParameterMissing(f"article cap must be ≥ 0 (got {cap})")
    return cap


def fetch_run_news(
    symbols: list[str],
    as_of: str,
    target: str,
    *,
    overnight: bool,
    lookback_days: int,
    max_articles: int,
    retries: int = 3,
    sleep=None,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Point-in-time news for a run, one implementation for every path.

    Returns ``(articles_by_symbol, stats_by_symbol)``. Articles are dicts (the
    store/model/prompt shape). Each stats entry is :func:`cap_newest`'s dict
    plus ``status``: ``ok`` (articles), ``empty`` (the source answered with
    nothing), or ``unavailable`` (the source failed after ``retries``, with
    ``error``). "The source was down" and "the week was quiet" produce the
    same empty list but mean opposite things; downstream must not conflate
    them.
    """
    import time as _time
    from config import MODEL
    from services.news_service import NewsUnavailable

    sleep = sleep or _time.sleep
    cap = normalize_article_cap(max_articles)
    if not overnight:
        _, lookback_days = normalize_lookback(lookback_days)
    by_symbol: dict[str, list[dict]] = {}
    stats_by_symbol: dict[str, dict] = {}
    for sym in symbols:
        articles: list = []
        stats: dict = cap_newest([], cap, as_of)[1]
        error: Optional[str] = None
        for attempt in range(1, retries + 1):
            try:
                if overnight:
                    articles, stats = fetch_overnight_news_with_stats(
                        sym, as_of, target,
                        relevance_threshold=MODEL.NEWS_OVERNIGHT_RELEVANCE,
                        max_articles=cap,
                        start_time_et=MODEL.NEWS_OVERNIGHT_START_ET,
                        end_time_et=MODEL.NEWS_OVERNIGHT_END_ET,
                    )
                else:
                    articles, stats = fetch_point_in_time_news_with_stats(
                        sym, as_of, lookback_days=lookback_days,
                        max_articles=cap)
                error = None
                break
            except NewsUnavailable as e:
                # The vendor throttles, and a loop over a whole watchlist is
                # exactly the shape that trips it. Only NewsUnavailable is
                # retried; a genuine empty window is not a failure.
                error = str(e)
                if attempt < retries:
                    sleep(2.0 * attempt)
            except Exception as e:  # noqa: BLE001 - reported, not hidden
                error = f"{type(e).__name__}: {e}"
                break
        by_symbol[sym] = [article_to_dict(a) for a in articles]
        entry = dict(stats)
        entry["status"] = ("unavailable" if error
                           else "ok" if articles else "empty")
        if error:
            entry["error"] = error[:200]
        stats_by_symbol[sym] = entry
    return by_symbol, stats_by_symbol


def news_window_payload(*, overnight: bool, lookback_days: int,
                        max_articles: int, as_of: str, target: str,
                        stats_by_symbol: dict[str, dict]) -> dict:
    """The structured trace event for a run's news window.

    Carries what was REQUESTED (window, cap) and what was actually KEPT per
    symbol (count, span, whether the cap bit), so the trace page can show
    "requested 30d → effective 6d (capped 512→100)" instead of the request
    alone.
    """
    from config import MODEL

    return {
        "event": "news_window",
        "filter": "overnight" if overnight else "lookback",
        "lookback_days": lookback_days,
        "max_articles": int(max_articles or 0),
        "relevance_threshold": (MODEL.NEWS_OVERNIGHT_RELEVANCE
                                if overnight else 0.5),
        "as_of": as_of,
        "target_date": target,
        "articles": sum(int(s.get("kept") or 0) for s in stats_by_symbol.values()),
        "articles_by_symbol": {s: int(v.get("kept") or 0)
                               for s, v in stats_by_symbol.items()},
        "source_status": {s: v.get("status", "ok")
                          for s, v in stats_by_symbol.items()},
        "by_symbol": {
            s: {k: v.get(k) for k in
                ("fetched", "kept", "capped", "oldest", "newest",
                 "effective_days", "status", "error")}
            for s, v in stats_by_symbol.items()
        },
    }


def describe_news_window(payload: dict) -> str:
    """One human line for the progress feed, honest about truncation."""
    from config import MODEL

    if payload.get("filter") == "overnight":
        head = (f"Overnight window {payload.get('as_of')} "
                f"{MODEL.NEWS_OVERNIGHT_START_ET} ET → {payload.get('target_date')} "
                f"{MODEL.NEWS_OVERNIGHT_END_ET} ET "
                f"(relevance ≥ {payload.get('relevance_threshold')})")
    else:
        head = (f"News window {payload.get('lookback_days')}d ending "
                f"{payload.get('as_of')} (target {payload.get('target_date')} "
                f"minus 1 trading day)")
    cap = int(payload.get("max_articles") or 0)
    head += f", cap {cap or 'none'}/symbol"
    line = f"{head}: {payload.get('articles', 0)} articles fetched (point-in-time)"
    by_sym = payload.get("by_symbol") or {}
    capped = [f"{s} {v.get('fetched')}→{v.get('kept')} "
              f"(effective {v.get('effective_days')}d)"
              for s, v in sorted(by_sym.items()) if v.get("capped")]
    if capped:
        line += ". CAP HIT: oldest articles dropped: " + ", ".join(capped)
    return line
