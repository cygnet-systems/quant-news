"""Point-in-time news windowing — lookahead-safe.

Ported from TradingAgents v0.3.1 (`dataflows/yfinance_news.py`) invariants,
adapted to our Alpha Vantage / yfinance article shapes. The three rules:

1. Normalize every operand to UTC before comparing. A naive value is *assumed*
   UTC (``.replace``); an aware value is *converted* (``.astimezone``). Alpha
   Vantage ``time_published`` is naive-UTC, so "assume UTC" is correct here.
2. Half-open window ``[start, end + 1 day)``: keeps all of the ``end`` day
   (through 23:59:59) but rejects an article stamped exactly midnight-after.
3. An article with **no publish date** is kept only if the window reaches the
   present (``end >= now - 1 day``). A historical/backtest window *excludes*
   undated items — you cannot prove they are not future news.

This is the countermeasure to the coverage-gap/leak class of bug our own
lookahead audits chase: the previous caller-side guard was a naive string
compare (``published_at[:19] <= "{date}T23:59:59"``) in app.py, and the news
was sourced from a rolling 7-day-from-*now* window, so backtests older than a
week silently ran on empty news.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from zoneinfo import ZoneInfo

DEFAULT_NEWS_LOOKBACK_DAYS = 14  # wide enough for drawdown-catalyst forensics

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


def filter_articles_as_of(
    articles: list,
    as_of: Union[str, datetime],
    lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
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
    lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
) -> tuple[str, str]:
    """Alpha Vantage ``time_from``/``time_to`` (``YYYYMMDDTHHMM``) for a point-in-time
    fetch: from ``as_of - lookback`` at 00:00 through the *close* of ``as_of``.

    The upper bound is the end of the ``as_of`` day because a decision made at
    the close of day T predicting day T+1 may use all of day T's news.
    """
    end_dt = _coerce_dt(as_of) or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    return start_dt.strftime("%Y%m%dT0000"), end_dt.strftime("%Y%m%dT2359")


# Cache point-in-time fetches per (symbol, as_of, lookback) so repeated calls
# for the same date — e.g. multiple backtest arms — don't re-hit the vendor.
_PIT_NEWS_CACHE: dict = {}


def clear_pit_news_cache() -> None:
    _PIT_NEWS_CACHE.clear()


def fetch_point_in_time_news(
    symbol: str,
    as_of: Union[str, datetime],
    lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    max_articles: Optional[int] = None,
    relevance_threshold: float = 0.5,
) -> list:
    """Fetch ticker news bounded to ``[as_of - lookback, as_of]`` (no lookahead).

    Pushes the window to Alpha Vantage via ``time_from``/``time_to`` and then
    re-applies :func:`filter_articles_as_of` as a belt-and-suspenders guard (a
    vendor's inclusive bound can still leak the midnight-after article). Falls
    back to yfinance news (also as-of filtered) when AV returns nothing.

    ``max_articles`` defaults to the config ceiling (NEWS_MAX_ARTICLES) — a
    hardcoded 50 here was what silently capped the report/preview paths after
    the fetch itself learned to paginate.
    """
    if max_articles is None:
        from config import MODEL
        max_articles = MODEL.NEWS_MAX_ARTICLES

    # max_articles and relevance are part of the key: a smaller earlier fetch
    # must not be served to a caller asking for the full window.
    cache_key = (symbol.upper(), str(as_of)[:10], lookback_days,
                 max_articles, relevance_threshold)
    if cache_key in _PIT_NEWS_CACHE:
        return _PIT_NEWS_CACHE[cache_key]

    # Imported here to avoid a circular import at module load.
    from services.news_service import (
        NewsUnavailable,
        fetch_alpha_vantage_news,
        fetch_yfinance_news,
    )

    time_from, time_to = av_time_bounds(as_of, lookback_days)
    av_error = None
    try:
        articles = fetch_alpha_vantage_news(
            symbol,
            max_articles=max_articles,
            time_from=time_from,
            time_to=time_to,
            relevance_threshold=relevance_threshold,
        )
    except NewsUnavailable as e:
        av_error = e
        articles = []

    if not articles:
        # yfinance news has no server-side window; fetch then filter.
        articles = fetch_yfinance_news(symbol, max_articles=max_articles)
        # Both sources failed. Raising is the point: an empty list here is
        # indistinguishable from a quiet week, and the caller would store a
        # prediction made on no news without recording that it had none.
        if not articles and av_error is not None:
            raise av_error

    result = filter_articles_as_of(articles, as_of, lookback_days)
    _PIT_NEWS_CACHE[cache_key] = result
    return result


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
    times are parameters, not constants — they are a formula the app owner
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
    """Fetch only the articles in the overnight gap before ``target_date``.

    Live, the window is additionally clamped to *now* — a pre-open run simply
    sees the overnight news that exists so far. The half-open comparison
    excludes an article stamped exactly at the open.
    """
    start_utc, end_utc = overnight_window_bounds(
        anchor_date, target_date, start_time_et, end_time_et)
    now_utc = datetime.now(timezone.utc)
    end_utc = min(end_utc, now_utc)
    if end_utc <= start_utc:
        return []

    cache_key = (symbol.upper(), "overnight", str(anchor_date)[:10],
                 str(target_date)[:10], relevance_threshold,
                 start_time_et, end_time_et)
    if cache_key in _PIT_NEWS_CACHE:
        return _PIT_NEWS_CACHE[cache_key]

    from services.news_service import fetch_alpha_vantage_news

    articles = fetch_alpha_vantage_news(
        symbol,
        max_articles=max_articles,
        time_from=start_utc.strftime("%Y%m%dT%H%M"),
        time_to=end_utc.strftime("%Y%m%dT%H%M"),
        relevance_threshold=relevance_threshold,
    )
    # Strict client-side re-check — the vendor bound is advisory here, and
    # unlike the lookback path an undated article can never qualify (its
    # position relative to a 17.5-hour window is unknowable).
    result = [
        a for a in articles
        if (pub := _article_pub_date(a)) is not None
        and start_utc <= as_utc(pub) < end_utc
    ]
    _PIT_NEWS_CACHE[cache_key] = result
    return result
