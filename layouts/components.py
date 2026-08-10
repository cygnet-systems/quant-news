"""Reusable UI components for the dashboard.

This module provides styled components following the design system
defined in PROJECT.md and config.py.
"""

from datetime import datetime
from typing import Any, Optional

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import COLORS, MODEL, SPACING


def calculate_period_label(start_date: str, end_date: str) -> str:
    """Calculate human-readable period label from date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Human-readable period label (e.g., "1Y Return", "6M Return", "2Y Return")
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Calculate difference in days
        days_diff = (end - start).days

        # Calculate approximate months and years
        months = round(days_diff / 30.44)  # Average days per month
        years = round(days_diff / 365.25, 1)  # Account for leap years

        # Determine best label
        if days_diff < 45:  # Less than 1.5 months
            return f"{days_diff}D Return"
        elif days_diff < 365:  # Less than 1 year
            return f"{months}M Return"
        elif years < 2:  # Between 1-2 years
            return "1Y Return"
        else:
            # For longer periods, show years (e.g., "2Y Return", "3Y Return")
            years_int = round(years)
            return f"{years_int}Y Return"
    except (ValueError, TypeError):
        # Fallback if date parsing fails
        return "Period Return"


def create_metric_card(
    title: str,
    value: str,
    change: Optional[str] = None,
    change_positive: Optional[bool] = None,
    subtitle: Optional[str] = None,
) -> dbc.Card:
    """Create a summary metric card.

    Args:
        title: Card title/label.
        value: Main value to display.
        change: Optional change value (e.g., "+2.5%").
        change_positive: True if positive, False if negative, None for neutral.
        subtitle: Optional subtitle text.

    Returns:
        Styled Bootstrap card component.
    """
    # Determine change color
    change_color = COLORS.TEXT_SECONDARY
    if change_positive is True:
        change_color = COLORS.POSITIVE
    elif change_positive is False:
        change_color = COLORS.NEGATIVE

    card_content = [
        html.Div(title, className="metric-label"),
        html.Div(
            [
                html.Span(value, className="metric-value"),
                html.Span(
                    change or "",
                    className="metric-change",
                    style={"color": change_color} if change else {},
                ),
            ],
            className="metric-row",
        ),
    ]

    if subtitle:
        card_content.append(
            html.Div(subtitle, className="metric-subtitle")
        )

    return dbc.Card(
        dbc.CardBody(card_content, className="metric-card-body"),
        className="metric-card",
    )


def create_news_card(
    title: str,
    source: str,
    time_ago: str,
    summary: Optional[str] = None,
    sentiment: Optional[str] = None,
    url: Optional[str] = None,
    impact: Optional[str] = None,
    price_change_percent: Optional[float] = None,
    symbol: Optional[str] = None,
) -> dbc.Card:
    """Create a news article card.

    Args:
        title: Article headline.
        source: News source name.
        time_ago: Relative time (e.g., "2h ago").
        summary: Optional article summary.
        sentiment: Optional sentiment ('bullish', 'bearish', 'neutral').
        url: Optional article URL.
        impact: Optional stock impact description (e.g., "Price target raised").
        price_change_percent: Optional current stock price change percentage.
        symbol: Optional stock ticker symbol.

    Returns:
        Styled news card component.
    """
    # Sentiment badge
    sentiment_badge = None
    if sentiment:
        badge_color = {
            "bullish": "success",
            "bearish": "danger",
            "neutral": "secondary",
        }.get(sentiment, "secondary")

        sentiment_badge = dbc.Badge(
            sentiment.upper(),
            color=badge_color,
            className="sentiment-badge me-2",
        )

    # Price change badge (e.g., "RGTI -2.10%")
    price_badge = None
    if price_change_percent is not None and symbol:
        is_positive = price_change_percent >= 0
        price_color = "success" if is_positive else "danger"
        price_sign = "+" if is_positive else ""
        price_text = f"{symbol} {price_sign}{price_change_percent:.2f}%"

        price_badge = dbc.Badge(
            price_text,
            color=price_color,
            className="price-badge me-2",
            style={"fontWeight": "600"},
        )

    # Impact badge
    impact_badge = None
    if impact:
        # Determine badge color based on impact type
        impact_lower = impact.lower()
        if any(word in impact_lower for word in ["raised", "beat", "growth", "upgraded", "positive", "approval", "buying", "bullish", "increased", "expansion", "awarded"]):
            impact_color = "info"
        elif any(word in impact_lower for word in ["lowered", "miss", "decline", "downgraded", "negative", "rejection", "selling", "bearish", "cut", "layoffs", "lost", "recall"]):
            impact_color = "warning"
        else:
            impact_color = "light"

        impact_badge = dbc.Badge(
            impact,
            color=impact_color,
            className="impact-badge me-2",
            style={"fontWeight": "500"},
        )

    # Title (optionally linked)
    title_element = html.A(
        title,
        href=url,
        target="_blank",
        className="news-title",
    ) if url else html.Span(title, className="news-title")

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        sentiment_badge,
                        price_badge,
                        impact_badge,
                        html.Span(source, className="news-source"),
                        html.Span(" | ", className="news-divider"),
                        html.Span(time_ago, className="news-time"),
                    ],
                    className="news-meta",
                ),
                title_element,
                html.P(summary, className="news-summary") if summary else None,
            ],
            className="news-card-body",
        ),
        className="news-card mb-2",
    )


def create_period_selector(selected: str = "1y") -> dbc.ButtonGroup:
    """Create time period selector buttons.

    Args:
        selected: Currently selected period.

    Returns:
        Button group for period selection.
    """
    # Organized short -> long in three clusters: days, trading weeks, then
    # calendar months/years. Sub-6mo values are DISPLAY windows — the data
    # layer keeps a 6-month daily floor underneath (see app._PERIOD_CONFIG),
    # and 1D/3D/1W render intraday bars on the price chart.
    periods = [
        ("1D", "1d"), ("3D", "3d"),
        ("1W", "1wk"), ("2W", "2wk"), ("3W", "3wk"),
        ("1M", "1mo"), ("3M", "3mo"), ("6M", "6mo"),
        ("YTD", "ytd"), ("1Y", "1y"), ("2Y", "2y"), ("5Y", "5y"),
    ]
    cluster_starts = {"1wk", "1mo", "ytd"}

    return dbc.ButtonGroup(
        [
            dbc.Button(
                label,
                id={"type": "period-btn", "period": value},
                color="primary" if value == selected else "secondary",
                outline=value != selected,
                size="sm",
                className=("period-btn period-cluster-start"
                           if value in cluster_starts else "period-btn"),
            )
            for label, value in periods
        ],
        className="period-selector",
    )


def create_indicator_toggles() -> html.Div:
    """Create toggles for chart indicators with tooltips.

    Returns:
        Div containing indicator checkboxes with explanatory tooltips.
    """
    # (label, value, default_checked, tooltip_text)
    indicators = [
        ("SMA 20", "sma_20", True, "20-day Simple Moving Average: Average of the last 20 closing prices"),
        ("SMA 50", "sma_50", True, "50-day Simple Moving Average: Medium-term trend indicator"),
        ("SMA 200", "sma_200", False, "200-day Simple Moving Average: Long-term trend indicator"),
        ("Bollinger", "bollinger", False, "Bollinger Bands: 20-day SMA ± 2 standard deviations"),
        ("Volume", "volume", True, "Trading volume with 20-day moving average"),
    ]

    indicator_items = []
    for label, value, checked, tooltip in indicators:
        item_id = f"indicator-{value}"
        indicator_items.append(
            html.Div(
                [
                    dbc.Checkbox(
                        id={"type": "indicator-check", "indicator": value},
                        value=checked,
                        className="indicator-checkbox",
                    ),
                    html.Span(
                        label,
                        id=item_id,
                        className="indicator-label",
                    ),
                    dbc.Tooltip(
                        tooltip,
                        target=item_id,
                        placement="right",
                    ),
                ],
                className="indicator-item",
            )
        )

    # Hidden checklist to maintain compatibility with existing callback
    return html.Div(
        [
            dbc.Checklist(
                id="indicator-toggles",
                options=[
                    {"label": label, "value": value}
                    for label, value, _, _ in indicators
                ],
                value=[value for _, value, checked, _ in indicators if checked],
                inline=True,
                className="indicator-toggles",
            ),
        ],
        className="indicator-toggle-container",
    )


def create_loading_spinner(component_id: str) -> dcc.Loading:
    """Create a loading spinner wrapper.

    Args:
        component_id: ID of component to wrap.

    Returns:
        Loading component with spinner.
    """
    return dcc.Loading(
        id=f"{component_id}-loading",
        type="circle",
        color=COLORS.ACCENT_PRIMARY,
    )


def create_cache_status_badge(
    last_updated: Optional[str] = None,
    record_count: Optional[int] = None,
) -> html.Div:
    """Create cache status indicator.

    Args:
        last_updated: Last update timestamp.
        record_count: Number of cached records.

    Returns:
        Status badge component.
    """
    if last_updated:
        status_text = f"Cached: {last_updated}"
        if record_count:
            status_text += f" ({record_count} records)"
        color = COLORS.POSITIVE
    else:
        status_text = "Not cached"
        color = COLORS.TEXT_MUTED

    return html.Div(
        [
            html.Span("", className="cache-dot", style={"backgroundColor": color}),
            html.Span(status_text, className="cache-text"),
        ],
        className="cache-status",
    )


def create_data_actions(include_refresh: bool = True) -> html.Div:
    """Page-local data actions (refresh, export, view).

    Rendered on the pages whose data they act on — Analyze (all three) and
    Performance (export/view only). The ids are fixed and shared across the
    two pages; only one page is mounted at a time, so this is the same
    single-mount pattern as reports-new-btn. Their callbacks take these as
    allow_optional Inputs because four of the six routes render neither.
    """
    buttons = []
    if include_refresh:
        buttons.append(dbc.Button(
            [html.I(className="bi bi-arrow-clockwise me-1"), "Refresh"],
            id="refresh-data-btn",
            color="secondary",
            size="sm",
            outline=True,
        ))
    buttons.extend([
        dbc.Button(
            [html.I(className="bi bi-download me-1"), "Export"],
            id="export-data-btn",
            color="secondary",
            size="sm",
            outline=True,
        ),
        dbc.Button(
            [html.I(className="bi bi-table me-1"), "View Data"],
            id="view-data-btn",
            color="secondary",
            size="sm",
            outline=True,
        ),
    ])
    return html.Div(buttons, className="data-actions")


def create_empty_state(message: str = "No data to display") -> html.Div:
    """Create an empty state placeholder.

    Args:
        message: Message to display.

    Returns:
        Styled empty state component.
    """
    return html.Div(
        [
            html.Div(className="empty-icon"),
            html.P(message, className="empty-message"),
        ],
        className="empty-state",
    )


def create_error_alert(message: str) -> dbc.Alert:
    """Create an error alert.

    Args:
        message: Error message to display.

    Returns:
        Bootstrap alert component.
    """
    return dbc.Alert(
        message,
        color="danger",
        dismissable=True,
        className="error-alert",
    )


def create_recommendation_banner(
    recommendation: str,
    confidence: Optional[float] = None,
    article_count: int = 0,
    date_range: Optional[str] = None,
) -> html.Div:
    """Create the prominent recommendation banner for Overview tab.

    Args:
        recommendation: One of BULLISH, CAUTIOUS_BULLISH, NEUTRAL,
                       CAUTIOUS_BEARISH, BEARISH, or LOADING
        confidence: Optional confidence score 0-1
        article_count: Number of articles analyzed
        date_range: Date range string (e.g., "Jan 15-18")

    Returns:
        Styled recommendation banner component.
    """
    # Handle loading state — prompt user to run AI Report
    if recommendation == "LOADING":
        return html.Div(
            [
                html.Div("Awaiting Analysis", className="recommendation-label"),
                html.Div('Run "AI Report" to analyze', className="recommendation-meta"),
            ],
            className="recommendation-banner loading",
        )

    # Map recommendation to CSS class
    rec_lower = recommendation.lower().replace(" ", "_")
    if "bullish" in rec_lower:
        css_class = "bullish"
    elif "bearish" in rec_lower:
        css_class = "bearish"
    else:
        css_class = "neutral"

    # Format display text
    display_text = recommendation.upper().replace("_", " ")

    # Build children
    children = [html.Div(display_text, className="recommendation-label")]

    # Add confidence text
    if confidence is not None:
        confidence_pct = int(confidence * 100)
        children.append(
            html.Div(f"Confidence: {confidence_pct}%", className="recommendation-confidence")
        )

    # Build meta text
    meta_parts = []
    if article_count > 0:
        meta_parts.append(f"Based on {article_count} article{'s' if article_count != 1 else ''}")
    if date_range:
        meta_parts.append(f"from {date_range}")
    if meta_parts:
        children.append(
            html.Div(" ".join(meta_parts), className="recommendation-meta")
        )

    return html.Div(
        children,
        className=f"recommendation-banner {css_class}",
    )


def create_sentiment_breakdown(
    bullish: int,
    neutral: int,
    bearish: int,
) -> html.Div:
    """Create visual sentiment breakdown with progress bars.

    Args:
        bullish: Count of bullish articles
        neutral: Count of neutral articles
        bearish: Count of bearish articles

    Returns:
        Styled sentiment breakdown component.
    """
    total = bullish + neutral + bearish
    if total == 0:
        return html.Div(
            "No sentiment data available",
            className="sentiment-breakdown",
            style={"color": COLORS.TEXT_MUTED, "fontSize": "13px"},
        )

    # Calculate percentages
    bullish_pct = (bullish / total) * 100
    neutral_pct = (neutral / total) * 100
    bearish_pct = (bearish / total) * 100

    return html.Div(
        [
            html.Div("Sentiment Analysis", className="section-title"),
            html.Div(
                [
                    # Bullish row
                    html.Div(
                        [
                            html.Span("Bullish", className="sentiment-bar-label"),
                            html.Div(
                                html.Div(
                                    className="sentiment-bar-fill bullish",
                                    style={"width": f"{bullish_pct}%"},
                                ),
                                className="sentiment-bar-track",
                            ),
                            html.Span(str(bullish), className="sentiment-bar-count"),
                        ],
                        className="sentiment-bar-row",
                    ),
                    # Neutral row
                    html.Div(
                        [
                            html.Span("Neutral", className="sentiment-bar-label"),
                            html.Div(
                                html.Div(
                                    className="sentiment-bar-fill neutral",
                                    style={"width": f"{neutral_pct}%"},
                                ),
                                className="sentiment-bar-track",
                            ),
                            html.Span(str(neutral), className="sentiment-bar-count"),
                        ],
                        className="sentiment-bar-row",
                    ),
                    # Bearish row
                    html.Div(
                        [
                            html.Span("Bearish", className="sentiment-bar-label"),
                            html.Div(
                                html.Div(
                                    className="sentiment-bar-fill bearish",
                                    style={"width": f"{bearish_pct}%"},
                                ),
                                className="sentiment-bar-track",
                            ),
                            html.Span(str(bearish), className="sentiment-bar-count"),
                        ],
                        className="sentiment-bar-row",
                    ),
                ],
                className="sentiment-bars",
            ),
        ],
        className="sentiment-breakdown",
    )


def create_top_headlines(
    articles: list[dict],
    max_count: int = 3,
) -> html.Div:
    """Create top headlines preview for Overview tab.

    Args:
        articles: List of article dictionaries with title, url, source, published_at
        max_count: Maximum number of headlines to show

    Returns:
        Styled headlines component with "See all" link.
    """
    if not articles:
        return html.Div(
            [
                html.Div("Top Headlines", className="section-title"),
                html.Div(
                    "No headlines available",
                    style={"color": COLORS.TEXT_MUTED, "fontSize": "13px"},
                ),
            ],
            className="top-headlines",
        )

    # Get top articles
    top_articles = articles[:max_count]
    total_count = len(articles)

    headline_items = []
    for article in top_articles:
        title = article.get("title", "Untitled")
        url = article.get("url", "#")
        source = article.get("source", "")

        headline_items.append(
            html.Li(
                [
                    html.A(
                        title,
                        href=url,
                        target="_blank",
                        className="headline-link",
                    ),
                    html.Div(source, className="headline-meta") if source else None,
                ],
                className="headline-item",
            )
        )

    children = [
        html.Div("Top Headlines", className="section-title"),
        html.Ul(headline_items, className="headline-list"),
    ]

    # Add "See all" link if there are more articles
    if total_count > max_count:
        children.append(
            html.Span(
                f"See all {total_count} articles →",
                id="see-all-news-link",
                className="see-all-link",
            )
        )

    return html.Div(children, className="top-headlines")


def create_news_quick_stats(
    article_count: int,
    source_count: int,
    date_range: Optional[str] = None,
    symbols: Optional[list[str]] = None,
) -> html.Div:
    """Create quick stats summary for Overview tab.

    Args:
        article_count: Total number of articles
        source_count: Number of unique sources
        date_range: Date range string (e.g., "Jan 15-18")
        symbols: List of stock symbols covered

    Returns:
        Styled quick stats grid component.
    """
    stats = [
        {"value": str(article_count), "label": "Articles"},
        {"value": str(source_count), "label": "Sources"},
    ]

    if symbols:
        stats.append({"value": str(len(symbols)), "label": "Stocks"})

    if date_range:
        stats.append({"value": date_range, "label": "Date Range"})

    return html.Div(
        [
            html.Div(
                [
                    html.Div(stat["value"], className="quick-stat-value"),
                    html.Div(stat["label"], className="quick-stat-label"),
                ],
                className="quick-stat-item",
            )
            for stat in stats
        ],
        className="quick-stats",
    )


def create_overview_empty_state() -> html.Div:
    """Create empty state for Overview tab when no data is available.

    Returns:
        Styled empty state component.
    """
    return html.Div(
        [
            html.I(className="bi bi-graph-up empty-icon"),
            html.P(
                "Select stocks to view news analysis and recommendations",
                className="empty-message",
            ),
        ],
        className="overview-empty-state",
    )


def create_ensemble_config_drawer() -> dbc.Offcanvas:
    """Create the ensemble configuration offcanvas drawer.

    Contains per-model enable/disable switches and weight sliders.
    Triggered by the gear icon on the ensemble card.
    """
    models = [
        ("kronos_mini", "Kronos"),
        ("xgboost_shap", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("deberta_sentiment", "DeBERTa"),
        ("trading_agents", "TradingAgents"),
    ]

    default_enabled = set(MODEL.ENSEMBLE_DEFAULT_ENABLED)
    default_weights = dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)

    model_rows = []
    for model_id, display_name in models:
        is_enabled = model_id in default_enabled
        weight = default_weights.get(model_id, 1.0)

        model_rows.append(
            html.Div(
                [
                    # Top row: switch + model name + current decision placeholder
                    html.Div(
                        [
                            dbc.Switch(
                                id={"type": "ensemble-model-switch", "model": model_id},
                                value=is_enabled,
                                className="ensemble-model-switch",
                            ),
                            html.Span(
                                display_name,
                                className="ensemble-model-label",
                            ),
                            html.Span(
                                id={"type": "ensemble-model-decision", "model": model_id},
                                className="ensemble-model-decision",
                            ),
                        ],
                        className="ensemble-model-header",
                    ),
                    # Weight slider + number input
                    html.Div(
                        [
                            html.Span(
                                "Weight:",
                                className="ensemble-weight-label",
                            ),
                            dcc.Slider(
                                id={"type": "ensemble-weight-slider", "model": model_id},
                                min=0.1,
                                max=2.0,
                                step=0.1,
                                value=weight,
                                marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0"},
                                className="ensemble-weight-slider",
                                disabled=not is_enabled,
                            ),
                            dcc.Input(
                                id={"type": "ensemble-weight-input", "model": model_id},
                                type="number",
                                min=0.1,
                                max=2.0,
                                step=0.1,
                                value=weight,
                                className="ensemble-weight-input",
                                disabled=not is_enabled,
                            ),
                        ],
                        className="ensemble-weight-row",
                    ),
                ],
                className="ensemble-model-config-row",
            )
        )

    return dbc.Offcanvas(
        [
            html.Div(
                [
                    html.P(
                        "Choose which models contribute to the ensemble "
                        "prediction and adjust their relative weights.",
                        className="ensemble-config-description",
                    ),
                    html.Div(model_rows, className="ensemble-config-models"),
                    html.Hr(style={"borderColor": COLORS.BORDER_SUBTLE}),
                    dbc.Button(
                        "Reset to Defaults",
                        id="ensemble-reset-btn",
                        outline=True,
                        color="secondary",
                        size="sm",
                        className="ensemble-reset-btn",
                    ),
                ],
                className="ensemble-config-content",
            ),
        ],
        id="ensemble-config-drawer",
        title="Ensemble Configuration",
        placement="end",
        is_open=False,
        style={"width": "380px", "backgroundColor": COLORS.BG_SECONDARY},
    )
