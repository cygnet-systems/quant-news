"""News service for fetching and aggregating stock news.

This module provides functions to fetch news from multiple sources:
- yfinance (built-in, no API key needed)
- Alpha Vantage News API (requires free API key)
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import requests

from config import API
from services.rate_limiter import alpha_vantage_bucket

logger = logging.getLogger(__name__)


class NewsUnavailable(RuntimeError):
    """The news source could not be reached or refused the request.

    Distinct from "the window genuinely contains no articles". Callers that
    cannot tell the two apart end up reporting a vendor outage as a quiet
    news week.
    """


# How far behind the window end the cached training news may fall before it is
# refetched. Two days absorbs a weekend with no coverage without letting a
# symbol's news window quietly freeze on the day it was first fetched.
_HISTORICAL_NEWS_MAX_STALENESS_DAYS = 2

# AV's server-side page ceiling, and a runaway guard on the cursor walk.
# 10 pages x 1000 articles covers any realistic single-symbol window.
_AV_PAGE_LIMIT = 1000
_MAX_NEWS_PAGES = 10


@dataclass
class NewsArticle:
    """Represents a news article."""

    id: str
    symbol: str
    title: str
    source: str
    url: str
    published_at: datetime
    summary: Optional[str] = None
    sentiment: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    sentiment_score: Optional[float] = None
    impact: Optional[str] = None  # How the stock is affected (e.g., "price target raised")
    price_change_percent: Optional[float] = None  # Current stock price change %
    ticker_relevance_score: Optional[float] = None  # AV ticker relevance (0-1)
    topics: Optional[list[dict]] = None  # AV topic list [{topic, relevance_score}, ...]
    overall_sentiment_score: Optional[float] = None  # AV overall sentiment (-1 to 1)
    overall_sentiment_label: Optional[str] = None  # AV overall sentiment label


def _extract_stock_impact(title: str, summary: str = "") -> Optional[str]:
    """Extract how the stock is affected from the article title and summary.

    Args:
        title: Article title.
        summary: Article summary (optional).

    Returns:
        Description of the stock impact, or None if not determinable.
    """
    text = f"{title} {summary}".lower()

    # Define impact patterns (order matters - more specific first)
    impact_patterns = [
        # Price targets and ratings
        (["price target raised", "raises price target", "ups price target"], "Price target raised"),
        (["price target lowered", "lowers price target", "cuts price target"], "Price target lowered"),
        (["price target"], "Price target updated"),
        (["upgrade", "upgraded"], "Stock upgraded"),
        (["downgrade", "downgraded"], "Stock downgraded"),
        (["initiates coverage", "starts coverage", "begins coverage"], "Coverage initiated"),
        (["buy rating", "outperform rating", "overweight"], "Positive rating"),
        (["sell rating", "underperform rating", "underweight"], "Negative rating"),
        # Earnings and financials
        (["beats estimates", "beats expectations", "tops estimates", "earnings beat"], "Earnings beat"),
        (["misses estimates", "misses expectations", "earnings miss"], "Earnings miss"),
        (["revenue growth", "sales growth", "revenue up", "sales up"], "Revenue growth"),
        (["revenue decline", "sales decline", "revenue down", "sales down"], "Revenue decline"),
        (["profit increase", "profit growth", "net income up"], "Profit growth"),
        (["profit decline", "net income down", "loss reported"], "Profit decline"),
        (["guidance raised", "raises guidance", "raises outlook"], "Guidance raised"),
        (["guidance lowered", "lowers guidance", "cuts outlook"], "Guidance lowered"),
        (["dividend increase", "raises dividend", "dividend hike"], "Dividend increased"),
        (["dividend cut", "suspends dividend", "dividend reduced"], "Dividend cut"),
        (["stock buyback", "share repurchase", "buyback program"], "Stock buyback announced"),
        # Corporate actions
        (["acquisition", "acquires", "to acquire", "buyout"], "Acquisition news"),
        (["merger", "to merge", "merging with"], "Merger news"),
        (["ipo", "initial public offering", "goes public"], "IPO news"),
        (["stock split", "split announced"], "Stock split"),
        (["spinoff", "spin-off", "spins off"], "Spinoff announced"),
        # Legal and regulatory
        (["lawsuit", "sued", "legal action", "litigation"], "Legal action"),
        (["sec investigation", "regulatory probe", "investigation"], "Regulatory investigation"),
        (["fda approval", "drug approved", "receives approval"], "FDA/Regulatory approval"),
        (["fda rejection", "drug rejected", "approval denied"], "FDA/Regulatory rejection"),
        (["settles", "settlement", "agrees to pay"], "Legal settlement"),
        # Business operations
        (["layoffs", "job cuts", "workforce reduction", "cutting jobs"], "Layoffs announced"),
        (["hiring", "adding jobs", "workforce expansion"], "Hiring expansion"),
        (["new product", "product launch", "launches", "unveils"], "Product launch"),
        (["partnership", "partners with", "collaboration"], "Partnership announced"),
        (["contract win", "wins contract", "awarded contract"], "Contract awarded"),
        (["loses contract", "contract loss"], "Contract lost"),
        (["expansion", "expands into", "new market"], "Market expansion"),
        (["restructuring", "reorganization"], "Restructuring"),
        # Market sentiment
        (["insider buying", "insiders buy", "ceo buys"], "Insider buying"),
        (["insider selling", "insiders sell", "ceo sells"], "Insider selling"),
        (["short interest", "heavily shorted", "short squeeze"], "Short interest news"),
        (["analyst bullish", "analysts optimistic"], "Bullish analyst sentiment"),
        (["analyst bearish", "analysts pessimistic"], "Bearish analyst sentiment"),
        # External factors
        (["tariff", "trade war", "import duty"], "Trade/Tariff impact"),
        (["supply chain", "chip shortage", "supply shortage"], "Supply chain impact"),
        (["recall", "product recall"], "Product recall"),
    ]

    for keywords, impact in impact_patterns:
        if any(keyword in text for keyword in keywords):
            return impact

    return None


def _generate_article_id(title: str, source: str, published_at: datetime) -> str:
    """Generate unique ID for an article.

    Args:
        title: Article title.
        source: News source name.
        published_at: Publication timestamp.

    Returns:
        MD5 hash string as unique identifier.
    """
    content = f"{title}|{source}|{published_at.isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()


def fetch_yfinance_news(symbol: str, max_articles: int = 10) -> list[NewsArticle]:
    """Fetch news from yfinance.

    Args:
        symbol: Stock ticker symbol.
        max_articles: Maximum number of articles to return (0 = all).

    Returns:
        List of NewsArticle objects.
    """
    try:
        from services.stock_data import get_ticker

        ticker = get_ticker(symbol)
        news = ticker.news

        if not news:
            return []

        articles = []
        for item in (news[:max_articles] if max_articles else news):
            # Newer yfinance nests the article under "content" with an ISO
            # pubDate; older builds carry an epoch providerPublishTime. An
            # article with neither is skipped. Stamping it 1970 put it
            # outside every window while still counting as "fetched".
            content = item.get("content") or {}
            epoch = item.get("providerPublishTime")
            iso = content.get("pubDate") or content.get("displayTime")
            if epoch:
                pub_time = datetime.fromtimestamp(epoch)
            elif iso:
                try:
                    pub_time = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
                except ValueError:
                    logger.debug(f"{symbol}: yfinance article has unparseable date {iso!r}")
                    continue
            else:
                logger.debug(f"{symbol}: yfinance article has no publish time, skipped")
                continue

            title = item.get("title") or content.get("title", "")
            article = NewsArticle(
                id=_generate_article_id(
                    title,
                    item.get("publisher", ""),
                    pub_time,
                ),
                symbol=symbol.upper(),
                title=title,
                source=item.get("publisher", "Unknown"),
                url=item.get("link", ""),
                published_at=pub_time,
                summary=None,  # yfinance doesn't provide summaries
                sentiment=None,
                impact=_extract_stock_impact(title),
            )
            articles.append(article)

        return articles

    except Exception as e:
        # Lenient on purpose: this is the fallback source. Its caller raises
        # the primary source's NewsUnavailable when both come back empty.
        logger.warning("%s: yfinance news failed: %s", symbol, e)
        return []


def _parse_av_feed_item(item: dict, symbol: str) -> Optional[dict]:
    """Parse a single AV NEWS_SENTIMENT feed item into a standardized dict.

    Shared parser for all AV news consumers. Extracts every field needed
    by any downstream consumer (UI, DeBERTa, feature_builder, LLM service).

    Args:
        item: Raw feed item from AV API response.
        symbol: Ticker symbol (or "" for global/macro news).

    Returns:
        Standardized article dict, or None if timestamp is unparseable.
    """
    time_str = item.get("time_published", "")
    try:
        pub_time = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
    except ValueError:
        return None

    title = item.get("title", "")
    summary = item.get("summary", "")
    source = item.get("source", "Unknown")
    url = item.get("url", "")
    topics = item.get("topics", [])
    overall_sentiment_score = float(item.get("overall_sentiment_score", 0))
    overall_sentiment_label = item.get("overall_sentiment_label", "")

    # Extract ticker-specific sentiment and relevance
    ticker_sentiment = None
    sentiment_score = None
    ticker_sentiment_score = None
    ticker_relevance_score = None

    if symbol:
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == symbol.upper():
                ticker_sentiment_score = float(ts.get("ticker_sentiment_score", 0))
                ticker_relevance_score = float(ts.get("relevance_score", 0))
                sentiment_score = ticker_sentiment_score
                label = ts.get("ticker_sentiment_label", "")
                if "bullish" in label.lower():
                    ticker_sentiment = "bullish"
                elif "bearish" in label.lower():
                    ticker_sentiment = "bearish"
                else:
                    ticker_sentiment = "neutral"
                break

    # Fallback to overall sentiment if no ticker-specific data
    if ticker_sentiment is None:
        if overall_sentiment_score > 0.15:
            ticker_sentiment = "bullish"
        elif overall_sentiment_score < -0.15:
            ticker_sentiment = "bearish"
        else:
            ticker_sentiment = "neutral"
        sentiment_score = overall_sentiment_score

    return {
        "id": _generate_article_id(title, source, pub_time),
        "symbol": symbol.upper() if symbol else "_GLOBAL",
        "title": title,
        "summary": summary,
        "source": source,
        "url": url,
        "published_at": pub_time,
        "published_date": pub_time.strftime("%Y-%m-%d"),
        "sentiment": ticker_sentiment,
        "sentiment_score": sentiment_score,
        "ticker_sentiment_score": ticker_sentiment_score,
        "ticker_relevance_score": ticker_relevance_score,
        "topics": topics,
        "overall_sentiment_score": overall_sentiment_score,
        "overall_sentiment_label": overall_sentiment_label,
        "impact": _extract_stock_impact(title, summary),
    }


def fetch_alpha_vantage_news(
    symbol: str,
    max_articles: int = 0,
    topics: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "LATEST",
    relevance_threshold: float = 0.5,
) -> list[NewsArticle]:
    """Fetch news from Alpha Vantage News API, paginating past the page cap.

    AV serves at most 1000 articles per request. Pages are walked newest-first
    by moving ``time_to`` to just before the oldest article of the previous
    page, so a busy window is no longer silently truncated, the old
    single-request version capped every symbol at 50 articles and dropped the
    oldest (earliest-catalyst) articles first.

    Args:
        symbol: Stock ticker symbol.
        max_articles: Keep only the newest N AFTER relevance filtering; 0
            (the default) returns everything the window holds. The
            point-in-time callers pass 0 and apply their own, reported cap.
        topics: Comma-separated topics to filter by.
        time_from: Start time in YYYYMMDDTHHMM format. Defaults to 7 days ago.
        time_to: End time in YYYYMMDDTHHMM format.
        sort: Sort order - "LATEST", "EARLIEST", or "RELEVANCE".
        relevance_threshold: Minimum ticker_relevance_score (0-1). Default 0.5.

    Returns:
        List of NewsArticle objects with sentiment data, pre-filtered by relevance.
    """
    if not API.ALPHA_VANTAGE_API_KEY:
        raise NewsUnavailable("no Alpha Vantage API key configured")

    try:
        if not time_from:
            seven_days_ago = datetime.now() - timedelta(days=7)
            time_from = seven_days_ago.strftime("%Y%m%dT0000")

        raw_items: list[dict] = []
        seen_urls: set[str] = set()
        page_time_to = time_to
        # Pagination only walks backwards through LATEST-sorted pages; other
        # sort orders get the single-request behaviour they always had.
        paginate = sort == "LATEST"

        for _page in range(_MAX_NEWS_PAGES):
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol.upper(),
                "limit": _AV_PAGE_LIMIT,
                "apikey": API.ALPHA_VANTAGE_API_KEY,
                "sort": sort,
            }
            if topics:
                params["topics"] = topics
            if time_from:
                params["time_from"] = time_from
            if page_time_to:
                params["time_to"] = page_time_to

            # Pace against the shared quota before spending a call.
            alpha_vantage_bucket().acquire(timeout=API.DEFAULT_TIMEOUT * 4)
            response = requests.get(
                API.ALPHA_VANTAGE_BASE_URL,
                params=params,
                timeout=API.DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            # Alpha Vantage signals throttling and bad requests with HTTP 200
            # and an explanatory body, so raise_for_status sees nothing wrong.
            # Mapping that to an empty list is what made a rate-limited symbol
            # look like a symbol with no news: the models then ran on nothing
            # and the report stated there was no news, which reads as a
            # finding rather than a gap.
            for key in ("Note", "Information", "Error Message"):
                if key in data:
                    raise NewsUnavailable(f"Alpha Vantage {key}: "
                                          f"{str(data[key])[:200]}")

            if "feed" not in data:
                raise NewsUnavailable(
                    "Alpha Vantage response contained no 'feed' key "
                    f"(keys: {sorted(data)[:6]})")

            feed = data["feed"]
            new_items = [i for i in feed
                         if i.get("url") and i["url"] not in seen_urls]
            for item in new_items:
                seen_urls.add(item["url"])
            raw_items.extend(new_items)

            # Last page: short page, nothing new, or pagination disabled.
            if not paginate or len(feed) < _AV_PAGE_LIMIT or not new_items:
                break
            oldest = min((i.get("time_published", "") for i in new_items
                          if i.get("time_published")), default="")
            if len(oldest) < 13:
                break
            # Next page ends the minute before this page's oldest article.
            cursor = (datetime.strptime(oldest[:13], "%Y%m%dT%H%M")
                      - timedelta(minutes=1)).strftime("%Y%m%dT%H%M")
            if time_from and cursor <= time_from:
                break
            page_time_to = cursor

        articles = []
        for item in raw_items:
            parsed = _parse_av_feed_item(item, symbol)
            if parsed is None:
                continue

            # Pre-filter by relevance. A None score means the ticker isn't
            # in this article's ticker_sentiment list (i.e. it's market-wide
            # noise that merely surfaced under a LATEST query), drop it too.
            # When relevance_threshold is 0, filtering is disabled entirely.
            rel = parsed["ticker_relevance_score"]
            if relevance_threshold > 0 and (rel is None or rel < relevance_threshold):
                continue

            articles.append(NewsArticle(
                id=parsed["id"],
                symbol=symbol.upper(),
                title=parsed["title"],
                source=parsed["source"],
                url=parsed["url"],
                published_at=parsed["published_at"],
                summary=parsed["summary"],
                sentiment=parsed["sentiment"],
                sentiment_score=parsed["sentiment_score"],
                impact=parsed["impact"],
                ticker_relevance_score=parsed["ticker_relevance_score"],
                topics=parsed["topics"],
                overall_sentiment_score=parsed["overall_sentiment_score"],
                overall_sentiment_label=parsed["overall_sentiment_label"],
            ))

        if max_articles and len(articles) > max_articles:
            logger.info(f"{symbol}: {len(articles)} relevant articles in "
                        f"window, keeping newest {max_articles}")
            articles = articles[:max_articles]
        return articles

    except NewsUnavailable:
        raise
    except Exception as e:
        # A transport or parse failure is not "there is no news" either.
        raise NewsUnavailable(f"{type(e).__name__}: {e}") from e


def _get_stock_price_change(symbol: str) -> Optional[float]:
    """Get the current day's price change percentage for a stock.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Price change percentage, or None if unavailable.
    """
    try:
        from services.stock_data import get_stock_info

        info = get_stock_info(symbol)
        return round(info.day_change_percent, 2)
    except Exception:
        return None


def fetch_news(
    symbol: str,
    max_articles: int = 50,
    prefer_alpha_vantage: bool = True,
    include_price_change: bool = True,
    topics: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "LATEST",
    relevance_threshold: float = 0.5,
) -> list[NewsArticle]:
    """Fetch news from available sources.

    Tries Alpha Vantage first (if API key available), falls back to yfinance.

    Args:
        symbol: Stock ticker symbol.
        max_articles: Maximum number of articles to return.
        prefer_alpha_vantage: If True, try Alpha Vantage first.
        include_price_change: If True, fetch and include current stock price change.
        topics: Comma-separated topics to filter by (Alpha Vantage only).
        time_from: Start time in YYYYMMDDTHHMM format (Alpha Vantage only).
        time_to: End time in YYYYMMDDTHHMM format (Alpha Vantage only).
        sort: Sort order - "LATEST", "EARLIEST", or "RELEVANCE" (Alpha Vantage only).
        relevance_threshold: Minimum ticker relevance score (Alpha Vantage only).

    Returns:
        List of NewsArticle objects, sorted by date (newest first).
    """
    articles: list[NewsArticle] = []

    # Try Alpha Vantage first (has sentiment)
    av_error = None
    if prefer_alpha_vantage and API.ALPHA_VANTAGE_API_KEY:
        try:
            articles = fetch_alpha_vantage_news(
                symbol,
                max_articles,
                topics=topics,
                time_from=time_from,
                time_to=time_to,
                sort=sort,
                relevance_threshold=relevance_threshold,
            )
        except NewsUnavailable as e:
            # Named, not swallowed: falling back is fine, pretending the
            # primary source said "no news" is not.
            av_error = e
            logger.warning("%s: Alpha Vantage unavailable (%s), "
                           "falling back to yfinance", symbol, e)

    # Fallback to yfinance if no results
    if not articles:
        articles = fetch_yfinance_news(symbol, max_articles)
        if not articles and av_error is not None:
            raise av_error

    # Fetch current stock price change if requested
    if include_price_change and articles:
        price_change = _get_stock_price_change(symbol)
        if price_change is not None:
            for article in articles:
                article.price_change_percent = price_change

    # Sort by date (newest first)
    articles.sort(key=lambda x: x.published_at, reverse=True)

    return articles[:max_articles] if max_articles else articles


def fetch_news_cached(
    symbol: str,
    max_articles: int = 50,
    cache_minutes: int = 15,
) -> list[NewsArticle]:
    """Fetch news with DuckDB caching support.

    Checks cache first and returns cached articles if fresh enough.
    Otherwise fetches from API and caches the results.

    Args:
        symbol: Stock ticker symbol.
        max_articles: Maximum number of articles.
        cache_minutes: Cache validity in minutes (default 15).

    Returns:
        List of NewsArticle objects.
    """
    from services.cache_service import get_cache

    cache = get_cache()

    # Try cache first
    cached = cache.get_cached_news(symbol, cache_minutes)
    if cached:
        articles = [
            NewsArticle(
                id=a["id"],
                symbol=a["symbol"],
                title=a["title"],
                source=a["source"],
                url=a["url"],
                published_at=a["published_at"],
                summary=a.get("summary"),
                sentiment=a.get("sentiment"),
                sentiment_score=a.get("sentiment_score"),
                impact=a.get("impact"),
                ticker_relevance_score=a.get("ticker_relevance_score"),
                topics=a.get("topics"),
                overall_sentiment_score=a.get("overall_sentiment_score"),
                overall_sentiment_label=a.get("overall_sentiment_label"),
            )
            for a in cached[:max_articles]
        ]
        # Always fetch fresh price change even for cached articles
        price_change = _get_stock_price_change(symbol)
        if price_change is not None:
            for article in articles:
                article.price_change_percent = price_change
        return articles

    # Fetch fresh data
    articles = fetch_news(symbol, max_articles)

    # Cache the results
    if articles:
        cache.cache_news(symbol, articles)

    return articles


def get_sentiment_summary(articles: list[NewsArticle]) -> dict:
    """Calculate sentiment summary from articles.

    Args:
        articles: List of NewsArticle objects.

    Returns:
        Dictionary with sentiment counts and overall assessment.
    """
    if not articles:
        return {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
            "total": 0,
            "overall": "neutral",
            "score": 0.0,
        }

    bullish = sum(1 for a in articles if a.sentiment == "bullish")
    bearish = sum(1 for a in articles if a.sentiment == "bearish")
    neutral = sum(1 for a in articles if a.sentiment == "neutral" or a.sentiment is None)
    total = len(articles)

    # Calculate average sentiment score
    scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Determine overall sentiment
    if bullish > bearish and bullish > neutral:
        overall = "bullish"
    elif bearish > bullish and bearish > neutral:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "total": total,
        "overall": overall,
        "score": round(avg_score, 3),
    }


def fetch_historical_av_news(
    symbol: str,
    months: int = 3,
    as_of: Optional[str] = None,
) -> dict[str, list[dict]]:
    """Fetch AV news for the N months ending at ``as_of``, grouped by date.

    Results are cached in the Postgres historical_news table. Subsequent
    calls for the same symbol+date range hit the cache.

    Args:
        symbol: Stock ticker symbol (or "" for global news).
        months: Number of months of history to fetch.
        as_of: Window end (YYYY-MM-DD). None = now. A backtest MUST pass its
            cutoff: the window used to end at "now" regardless, so the
            tree models trained on news published after the session they
            were predicting.

    Returns:
        Dict mapping date string (YYYY-MM-DD) to list of article dicts.

    Raises:
        ValueError: If ALPHA_VANTAGE_API_KEY is not configured.
    """
    if not API.ALPHA_VANTAGE_API_KEY:
        raise ValueError(
            "ALPHA_VANTAGE_API_KEY required for model training features. "
            "Set it in .env or environment variables."
        )

    from services.cache_service import get_cache

    cache = get_cache()
    cache_symbol = symbol.upper() if symbol else "_GLOBAL"

    # Check cache first
    end_date = (datetime.strptime(str(as_of)[:10], "%Y-%m-%d").replace(
        hour=23, minute=59) if as_of else datetime.now())
    start_date = end_date - timedelta(days=30 * months)
    cached = cache.get_historical_news(
        cache_symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if cached:
        result: dict[str, list[dict]] = {}
        for article in cached:
            date_key = str(article.get("published_date", ""))[:10]
            if date_key:
                result.setdefault(date_key, []).append(article)
        if result:
            # Existence is not coverage. The window slides forward every day
            # but these rows do not: the old check returned any hit inside
            # [now-90d, now], so the FIRST fetch froze a symbol's training
            # news forever and every day after it trained on a window with a
            # growing hole at the recent end. Silently, since a partial
            # window looks exactly like a quiet news period.
            newest = max(result)
            staleness = (end_date.date() - date.fromisoformat(newest)).days
            if staleness <= _HISTORICAL_NEWS_MAX_STALENESS_DAYS:
                logger.info(f"AV news cache hit: {cache_symbol}, "
                            f"{len(cached)} articles through {newest}")
                return result
            logger.info(
                f"AV news cache for {cache_symbol} ends {newest} "
                f"({staleness}d before the window end), refetching"
            )

    # Fetch from AV API in monthly windows
    all_articles: list[dict] = []
    seen_urls: set[str] = set()
    failed_months: list[str] = []

    for m in range(months):
        month_end = end_date - timedelta(days=30 * m)
        month_start = end_date - timedelta(days=30 * (m + 1))
        time_from = month_start.strftime("%Y%m%dT0000")
        time_to = month_end.strftime("%Y%m%dT2359")

        try:
            params: dict = {
                "function": "NEWS_SENTIMENT",
                "limit": _AV_PAGE_LIMIT,
                "apikey": API.ALPHA_VANTAGE_API_KEY,
                "sort": "LATEST",
                "time_from": time_from,
                "time_to": time_to,
            }
            if symbol:
                params["tickers"] = symbol.upper()
            else:
                params["topics"] = (
                    "economy_macro,economy_fiscal,economy_monetary,financial_markets"
                )

            # Pace against the shared quota before spending a call.
            alpha_vantage_bucket().acquire(timeout=API.DEFAULT_TIMEOUT * 4)
            response = requests.get(
                API.ALPHA_VANTAGE_BASE_URL,
                params=params,
                timeout=API.DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            # Same HTTP-200 throttle signalling as the live fetch above. A
            # quota response here used to yield an empty corpus, so the model
            # trained newsless and was cached for the rest of the day with no
            # warning anywhere. Abort the whole fetch: the remaining slices
            # share the quota and would fail the same way.
            for key in ("Note", "Information", "Error Message"):
                if key in data:
                    raise NewsUnavailable(
                        f"Alpha Vantage {key} (historical {cache_symbol}): "
                        f"{str(data[key])[:200]}"
                    )
            if "feed" not in data:
                raise NewsUnavailable(
                    f"Alpha Vantage historical response for {cache_symbol} "
                    f"contained no 'feed' key (keys: {sorted(data)[:6]})"
                )

            for item in data.get("feed", []):
                url = item.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                parsed = _parse_av_feed_item(item, symbol)
                if parsed is None:
                    continue

                all_articles.append(parsed)

        except NewsUnavailable:
            raise
        except Exception as e:
            # Not `continue` in silence: a transport error here dropped a
            # whole month from the training corpus and the model trained on
            # the hole with nothing recording it.
            logger.warning(f"AV news fetch error (month {m}): {e}")
            failed_months.append(f"{time_from[:8]}-{time_to[:8]}: {e}")
    if failed_months:
        raise NewsUnavailable(
            f"historical news for {cache_symbol} incomplete: "
            f"{len(failed_months)}/{months} month slices failed: "
            + "; ".join(failed_months)[:300])

    # Store in cache
    if all_articles:
        cache.store_historical_news(all_articles)
        logger.info(
            f"Stored {len(all_articles)} AV articles for {cache_symbol}"
        )

    # Group by date
    result = {}
    for article in all_articles:
        date_key = article["published_date"]
        result.setdefault(date_key, []).append(article)

    return result


