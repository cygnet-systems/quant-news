"""Analyze page: the per-symbol and portfolio tab content.

These builders are pure -- they take store payloads and return components, so
the callbacks that own the stores stay in app.py.
"""

from datetime import datetime

import dash_bootstrap_components as dbc
from dash import dcc, html

from layouts.components import (
    create_data_actions,
    create_indicator_toggles,
    create_news_quick_stats,
    create_period_selector,
    create_recommendation_banner,
    create_sentiment_breakdown,
    create_top_headlines,
)
from layouts.signal_components import create_signal_cards
from layouts.strategy_components import create_strategy_section


def restore_tab(stored: str | None, available: list[str], default: str) -> str:
    """Return the stored tab if it still exists, else the default."""
    return stored if stored in available else default


def get_date_range(articles: list) -> str:
    """Get formatted date range from articles list."""
    if not articles:
        return ""

    dates = []
    for a in articles:
        pub = a.get("published_at")
        if pub:
            try:
                if isinstance(pub, str):
                    dt = datetime.fromisoformat(pub)
                else:
                    dt = pub
                # Articles mix tz-aware and naive timestamps depending on
                # source/cache — strip tzinfo so min/max can compare them.
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                dates.append(dt)
            except (ValueError, TypeError):
                continue

    if not dates:
        return ""

    oldest = min(dates)
    newest = max(dates)

    if oldest.date() == newest.date():
        return newest.strftime("%b %d")
    else:
        return f"{oldest.strftime('%b %d')} - {newest.strftime('%b %d')}"


def create_loading_state(symbols: list, stage: str = "news") -> html.Div:
    """Create loading state while fetching news data or generating AI analysis.

    Args:
        symbols: List of symbols being loaded
        stage: Loading stage - "news" for fetching news, "analysis" for AI analysis

    Returns:
        Loading state component with spinner and status message
    """
    symbols_text = ", ".join(symbols) if len(symbols) <= 3 else f"{len(symbols)} stocks"

    if stage == "news":
        status_text = f"Fetching news for {symbols_text}..."
        subtext = "Retrieving latest articles from sources"
    else:
        status_text = "Generating AI analysis..."
        subtext = f"Analyzing sentiment for {symbols_text}"

    return html.Div(
        [
            html.Div(
                [
                    # Spinner
                    html.Div(
                        [
                            html.Div(className="loading-spinner"),
                        ],
                        className="loading-spinner-container",
                    ),
                    # Status text
                    html.Div(
                        status_text,
                        className="loading-status-text",
                    ),
                    # Sub-text
                    html.Div(
                        subtext,
                        className="loading-subtext",
                    ),
                ],
                className="news-loading-state",
            ),
        ],
        className="news-loading-container",
    )


def create_ai_loading_indicator() -> html.Div:
    """Prompt to generate AI analysis from the toolbar."""
    return html.Div(
        [
            html.Div(
                [
                    html.I(className="bi bi-file-text", style={"fontSize": "1.2rem", "color": "#17a2b8"}),
                    html.Span(
                        'Run "AI Report" from the toolbar to generate insights',
                        className="loading-inline-text",
                        style={"color": "var(--text-secondary)"},
                    ),
                ],
                className="ai-loading-inline",
            ),
        ],
        className="ai-loading-section",
    )


def create_ai_failure_indicator() -> html.Div:
    """Show an error message with a retry button when AI analysis fails."""
    return html.Div(
        [
            html.Div(
                [
                    html.I(
                        className="bi bi-exclamation-triangle",
                        style={"fontSize": "1.2rem", "color": "#FFD700"},
                    ),
                    html.Span(
                        "AI analysis unavailable — the LLM provider didn't respond.",
                        className="loading-inline-text",
                        style={"color": "var(--text-secondary)", "marginLeft": "8px"},
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise"), " Retry"],
                        id="ai-retry-btn",
                        size="sm",
                        color="secondary",
                        outline=True,
                        style={"marginLeft": "12px"},
                    ),
                ],
                className="ai-loading-inline",
            ),
        ],
        className="ai-loading-section",
    )


def create_recommendations_section(rec_data: dict, symbol: str | None = None) -> html.Div | None:
    """Render recommendations from the synthesis model.

    When symbol is None, renders the overall portfolio view.
    When symbol is provided, renders per-symbol view (rec_data is already the symbol's dict).
    """
    if not rec_data:
        return None

    ACCENT = "#7B61FF"

    if symbol is None:
        # Overall view
        overall = rec_data.get("overall", {})
        if not overall:
            return None

        model_used = rec_data.get("model_used", "")

        children = [
            html.Div(
                [
                    html.I(className="bi bi-lightning-fill", style={"color": ACCENT, "marginRight": "6px"}),
                    html.Span("Recommendations", style={"color": ACCENT}),
                    html.Span(
                        f" — {model_used}" if model_used else "",
                        style={"color": "var(--text-muted)", "fontSize": "0.75rem", "marginLeft": "8px"},
                    ),
                    html.Span(
                        {"research+signals": "based on research + predictions",
                         "news+signals": "based on news analysis + predictions",
                         "signals": "based on predictions only",
                         }.get(rec_data.get("basis"), ""),
                        style={"color": "var(--text-muted)", "fontSize": "0.72rem",
                               "marginLeft": "auto", "fontStyle": "italic"},
                    ),
                ],
                className="section-title",
                style={"display": "flex", "alignItems": "center"},
            ),
        ]

        if overall.get("portfolio_action"):
            children.append(
                html.Div(
                    overall["portfolio_action"],
                    className="recommendations-action-banner",
                    style={
                        "borderLeft": f"3px solid {ACCENT}",
                        "padding": "8px 12px",
                        "marginBottom": "10px",
                        "fontWeight": "600",
                        "fontSize": "0.9rem",
                        "backgroundColor": "rgba(123, 97, 255, 0.08)",
                        "borderRadius": "4px",
                    },
                )
            )

        if overall.get("summary"):
            children.append(
                html.Div(
                    overall["summary"],
                    style={"fontSize": "0.85rem", "lineHeight": "1.5", "marginBottom": "10px",
                           "color": "var(--text-secondary)"},
                )
            )

        conflicts = overall.get("key_conflicts", [])
        if conflicts:
            conflict_items = []
            for conflict in conflicts:
                conflict_items.append(
                    html.Div(
                        [
                            html.I(className="bi bi-exclamation-triangle me-2",
                                   style={"color": "#FFB020"}),
                            html.Span(conflict, style={"fontSize": "0.82rem"}),
                        ],
                        className="conflict-card",
                    )
                )
            children.append(html.Div(conflict_items, className="mb-2"))

        if overall.get("risk_assessment"):
            children.append(
                html.Div(
                    [
                        html.Span("Risk: ", style={"fontWeight": "600", "color": "var(--text-muted)", "fontSize": "0.8rem"}),
                        html.Span(overall["risk_assessment"], style={"fontSize": "0.82rem", "color": "var(--text-secondary)"}),
                    ],
                    style={"marginBottom": "8px"},
                )
            )

        luna_watch = overall.get("watch_items") or []
        if isinstance(luna_watch, list) and luna_watch:
            children.append(html.Div(
                [
                    html.Span("Watch: ", style={"fontWeight": "600",
                                                "color": "var(--text-muted)",
                                                "fontSize": "0.8rem"}),
                    html.Ul(
                        [html.Li(str(w)) for w in luna_watch[:4]],
                        className="watch-items-list",
                        style={"display": "inline-block", "margin": 0},
                    ),
                ],
                style={"marginBottom": "8px"},
            ))

        # Per-symbol recommendations table
        by_symbol = rec_data.get("by_symbol", {})
        if by_symbol:
            rows = []
            for sym, sym_rec in by_symbol.items():
                action = sym_rec.get("action", "HOLD")
                conviction = sym_rec.get("conviction", "")
                action_color = (
                    "var(--positive)" if action == "BUY" else
                    "var(--negative)" if action == "SELL" else
                    "var(--text-secondary)"
                )
                # Full reasoning (the old 100-char cut hid the actual case
                # for the call) + the level/trigger that make it actionable.
                reason_cell = [html.Div(sym_rec.get("reasoning", ""))]
                if sym_rec.get("key_level"):
                    reason_cell.append(html.Div(
                        f"Key level: {sym_rec['key_level']}",
                        className="rec-key-level",
                    ))
                if sym_rec.get("change_trigger"):
                    reason_cell.append(html.Div(
                        f"Flips on: {sym_rec['change_trigger']}",
                        className="rec-change-trigger",
                    ))
                rows.append(html.Tr([
                    html.Td(sym, style={"fontWeight": "600"}),
                    html.Td(action, style={"color": action_color, "fontWeight": "600"}),
                    html.Td(conviction, style={"fontSize": "0.8rem"}),
                    html.Td(
                        reason_cell,
                        style={"fontSize": "0.8rem", "color": "var(--text-secondary)"},
                    ),
                ]))
            children.append(
                dbc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Symbol"), html.Th("Action"),
                            html.Th("Conviction"), html.Th("Reasoning"),
                        ])),
                        html.Tbody(rows),
                    ],
                    bordered=True, color="dark", size="sm",
                    style={"fontSize": "0.82rem"},
                )
            )

        return html.Div(children, className="recommendations-section")

    else:
        # Per-symbol view — rec_data is already the symbol's dict
        action = rec_data.get("action", "")
        if not action:
            return None

        conviction = rec_data.get("conviction", "")
        reasoning = rec_data.get("reasoning", "")
        conflicts = rec_data.get("conflicts", [])
        model_notes = rec_data.get("model_notes", "")

        action_color = (
            "var(--positive)" if action == "BUY" else
            "var(--negative)" if action == "SELL" else
            "var(--text-secondary)"
        )

        children = [
            html.Div(
                [
                    html.I(className="bi bi-lightning-fill", style={"color": ACCENT, "marginRight": "6px"}),
                    html.Span("Recommendation", style={"color": ACCENT}),
                ],
                className="section-title",
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                [
                    html.Span(
                        action,
                        className="action-badge",
                        style={
                            "color": action_color,
                            "backgroundColor": f"{action_color}22" if action != "HOLD" else "var(--bg-tertiary)",
                            "padding": "2px 10px",
                            "borderRadius": "4px",
                            "fontWeight": "700",
                            "fontSize": "0.85rem",
                            "marginRight": "8px",
                        },
                    ),
                    html.Span(
                        conviction,
                        className="conviction-badge",
                        style={
                            "color": "var(--text-muted)",
                            "fontSize": "0.8rem",
                            "fontWeight": "600",
                        },
                    ) if conviction else html.Span(),
                ],
                style={"marginBottom": "8px"},
            ),
        ]

        if reasoning:
            children.append(
                html.Div(
                    reasoning,
                    style={"fontSize": "0.82rem", "lineHeight": "1.5", "marginBottom": "8px",
                           "color": "var(--text-secondary)"},
                )
            )

        if rec_data.get("key_level"):
            children.append(html.Div(
                f"Key level: {rec_data['key_level']}",
                className="rec-key-level",
            ))
        if rec_data.get("change_trigger"):
            children.append(html.Div(
                f"Flips on: {rec_data['change_trigger']}",
                className="rec-change-trigger",
            ))

        if conflicts:
            for conflict in conflicts:
                children.append(
                    html.Div(
                        [
                            html.I(className="bi bi-exclamation-triangle me-2",
                                   style={"color": "#FFB020"}),
                            html.Span(conflict, style={"fontSize": "0.8rem"}),
                        ],
                        className="conflict-card",
                    )
                )

        if model_notes:
            children.append(
                html.Div(
                    [
                        html.I(className="bi bi-info-circle me-2",
                               style={"color": "var(--text-muted)"}),
                        html.Span(model_notes, style={"fontSize": "0.8rem", "color": "var(--text-secondary)"}),
                    ],
                    style={"marginTop": "6px"},
                )
            )

        return html.Div(children, className="recommendations-section")


def build_overall_tab_content(
    articles_by_symbol: dict,
    analysis_by_symbol: dict,
    overall_analysis: dict,
    symbols: list,
    ai_failed: bool = False,
    recommendations: dict | None = None,
) -> html.Div:
    """Build content for the Overall tab with summary table and combined analysis.

    Args:
        articles_by_symbol: Dict mapping symbol -> list of articles
        analysis_by_symbol: Dict mapping symbol -> analysis dict
        overall_analysis: Combined analysis across all symbols
        symbols: List of all selected symbols

    Returns:
        Overall tab content component
    """
    children = []

    # Collect all articles for stats
    all_articles = []
    for sym_articles in articles_by_symbol.values():
        all_articles.extend(sym_articles or [])

    # -- AI Summary (digest of all symbols) --
    # Show loading, failure, or pending prompt depending on state
    has_ai_analysis = bool(analysis_by_symbol) or bool(overall_analysis)
    if not has_ai_analysis and all_articles:
        if ai_failed:
            children.append(create_ai_failure_indicator())
        else:
            children.append(create_ai_loading_indicator())

    # Build a comprehensive summary from per-symbol analyses
    summary_parts = []

    if analysis_by_symbol:
        # Count recommendations
        rec_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for symbol in symbols:
            sym_analysis = analysis_by_symbol.get(symbol, {})
            rec = sym_analysis.get("recommendation", "").lower()
            if "bullish" in rec:
                rec_counts["bullish"] += 1
            elif "bearish" in rec:
                rec_counts["bearish"] += 1
            else:
                rec_counts["neutral"] += 1

        # Build summary text
        total_symbols = len(symbols)
        if rec_counts["bullish"] > 0:
            summary_parts.append(f"{rec_counts['bullish']} of {total_symbols} stocks show bullish signals")
        if rec_counts["bearish"] > 0:
            summary_parts.append(f"{rec_counts['bearish']} of {total_symbols} stocks show bearish signals")
        if rec_counts["neutral"] > 0 and rec_counts["bullish"] == 0 and rec_counts["bearish"] == 0:
            summary_parts.append(f"All {total_symbols} stocks show neutral sentiment")

    # Add overall key developments if available
    if overall_analysis and overall_analysis.get("key_developments"):
        summary_parts.append(overall_analysis.get("key_developments", ""))

    if summary_parts:
        ai_summary = html.Div(
            [
                html.Div(
                    [
                        html.Span("AI Summary", className="section-title mb-0"),
                        html.Div(
                            [
                                html.Button(
                                    [html.I(className="bi bi-braces me-1"), "View data"],
                                    id="ai-json-view-btn", className="ai-json-btn",
                                    title="Inspect the raw report payload",
                                ),
                            ],
                            className="ms-auto",
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                ),
                html.Div(
                    ". ".join(summary_parts) if len(summary_parts) > 1 else summary_parts[0],
                    className="key-developments-content",
                ),
            ],
            className="key-developments",
        )
        children.append(ai_summary)

        # Portfolio-level provenance: which model compiled this and from what.
        if overall_analysis and overall_analysis.get("model_used"):
            src = overall_analysis.get("sources") or {}
            bits = [f"Compiled by {overall_analysis['model_used']}"]
            if src.get("articles") is not None:
                bits.append(f"{src['articles']} articles across "
                            f"{len(symbols or [])} symbols")
            if src.get("as_of"):
                bits.append(f"as-of {src['as_of']}")
            if src.get("analysis_tier") == "sentiment_fallback":
                bits.append("sentiment-count fallback (LLM parse failed)")
            children.append(html.Div(
                " · ".join(bits),
                className="research-report-footnote",
                style={"marginTop": "-4px", "marginBottom": "8px"},
            ))

        if overall_analysis and overall_analysis.get("risk_factors"):
            risk_children = [
                html.Div(
                    [html.I(className="bi bi-exclamation-triangle me-2"), "Risk Factors"],
                    className="section-title",
                    style={"color": "var(--negative)"},
                ),
                html.Div(
                    overall_analysis["risk_factors"],
                    className="key-developments-content",
                    style={"borderLeft": "3px solid var(--negative)", "paddingLeft": "10px"},
                ),
            ]
            if overall_analysis.get("risks_read"):
                risk_children.append(html.Div(
                    [html.I(className="bi bi-arrow-return-right me-1"),
                     overall_analysis["risks_read"]],
                    className="analysis-read-line",
                ))
            children.append(html.Div(risk_children, className="key-developments"))

        overall_watch = (overall_analysis or {}).get("watch_items") or []
        if isinstance(overall_watch, list) and overall_watch:
            children.append(html.Div(
                [
                    html.Div(
                        [html.I(className="bi bi-binoculars me-2"), "Watch Items"],
                        className="section-title",
                    ),
                    html.Ul(
                        [html.Li(str(w)) for w in overall_watch[:4]],
                        className="watch-items-list",
                    ),
                ],
                className="key-developments",
            ))

    # -- Per-Symbol Recommendations Table --
    if symbols and analysis_by_symbol:
        table_rows = []
        for symbol in symbols:
            sym_analysis = analysis_by_symbol.get(symbol, {})
            sym_articles = articles_by_symbol.get(symbol, [])

            rec = sym_analysis.get("recommendation", "—")
            confidence = sym_analysis.get("confidence")
            article_count = len(sym_articles)

            # Determine color class for recommendation
            rec_lower = rec.lower() if rec != "—" else ""
            if "bullish" in rec_lower:
                rec_class = "rec-bullish"
            elif "bearish" in rec_lower:
                rec_class = "rec-bearish"
            else:
                rec_class = "rec-neutral"

            # Format recommendation display
            rec_display = rec.replace("_", " ") if rec != "—" else "—"

            # Format confidence
            conf_display = f"{int(confidence * 100)}%" if confidence else "—"

            table_rows.append(
                html.Tr(
                    [
                        html.Td(symbol, className="symbol-cell"),
                        html.Td(
                            rec_display,
                            className=f"recommendation-cell {rec_class}",
                        ),
                        html.Td(conf_display, className="confidence-cell"),
                        html.Td(str(article_count), className="articles-cell"),
                    ]
                )
            )

        recommendations_table = html.Div(
            [
                html.Div("Recommendations by Symbol", className="section-title"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Symbol"),
                                    html.Th("Recommendation"),
                                    html.Th("Confidence"),
                                    html.Th("Articles"),
                                ]
                            )
                        ),
                        html.Tbody(table_rows),
                    ],
                    className="recommendations-table",
                ),
            ],
            className="recommendations-section",
        )
        children.append(recommendations_table)

    # -- Aggregated Sentiment Breakdown --
    sentiment_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for a in all_articles:
        s = (a.get("sentiment") or "neutral").lower()
        if "bullish" in s:
            sentiment_counts["bullish"] += 1
        elif "bearish" in s:
            sentiment_counts["bearish"] += 1
        else:
            sentiment_counts["neutral"] += 1

    if any(sentiment_counts.values()):
        sentiment_breakdown = create_sentiment_breakdown(
            bullish=sentiment_counts["bullish"],
            neutral=sentiment_counts["neutral"],
            bearish=sentiment_counts["bearish"],
        )
        children.append(sentiment_breakdown)

    # -- Quick Stats --
    if all_articles:
        sources = list(set(a.get("source", "") for a in all_articles if a.get("source")))
        quick_stats = create_news_quick_stats(
            article_count=len(all_articles),
            source_count=len(sources),
            date_range=get_date_range(all_articles),
            symbols=symbols,
        )
        children.append(quick_stats)

    # -- Recommendations Section (Full Analysis) --
    if recommendations and not recommendations.get("error"):
        rec_section = create_recommendations_section(recommendations)
        if rec_section:
            children.insert(0, rec_section)

    # Handle empty state
    if not children:
        children.append(
            html.Div(
                [
                    html.I(className="bi bi-newspaper", style={"fontSize": "24px", "opacity": "0.5", "marginBottom": "8px"}),
                    html.P("No news available for selected stocks", style={"color": "#6B7280", "margin": "0"}),
                ],
                className="tab-empty-state",
                style={"textAlign": "center", "padding": "32px 16px"},
            )
        )

    return html.Div(children, className="tab-content-inner")


_QUALITY_FLAG_STYLE = {
    "clean": ("CLEAN", "var(--positive)"),
    "caution": ("CAUTION", "var(--warning, #e6a817)"),
    "bad_apple": ("BAD APPLE", "var(--negative)"),
}


def create_positioning_quality_section(analysis: dict) -> html.Div | None:
    """Options positioning + Bad Apples quality screen panel.

    Renders whatever of the two datasets the payload carries; None when a
    run predates them or both fetches failed.
    """
    pos = (analysis or {}).get("positioning") or {}
    quality = (analysis or {}).get("quality") or {}
    if not pos and not quality.get("total_checks"):
        return None

    children = [html.Div(
        [html.I(className="bi bi-activity me-2"), "Positioning & Quality"],
        className="section-title",
    )]

    if pos.get("pc_volume") is not None or pos.get("pc_oi") is not None:
        pcv = f"{pos['pc_volume']:.2f}" if pos.get("pc_volume") is not None else "n/a"
        pcoi = f"{pos['pc_oi']:.2f}" if pos.get("pc_oi") is not None else "n/a"
        read = pos.get("read", "")
        read_color = ("var(--negative)" if read == "put-tilted"
                      else "var(--positive)" if read == "call-tilted"
                      else "var(--text-secondary)")
        children.append(html.Div([
            html.Strong("Options flow: "),
            f"P/C volume {pcv} ({pos.get('put_volume', 0):,} puts / "
            f"{pos.get('call_volume', 0):,} calls) · P/C open interest {pcoi} ",
            html.Span(read.replace("-", " "),
                      style={"color": read_color, "fontWeight": "600"}),
            html.Span(f" · chain as of {pos.get('as_of', '')}",
                      style={"color": "var(--text-secondary)", "fontSize": "0.78rem"}),
        ], className="key-developments-content", style={"marginBottom": "8px"}))

    if quality.get("total_checks"):
        label, color = _QUALITY_FLAG_STYLE.get(
            quality.get("flag", ""), (str(quality.get("flag", "")).upper(),
                                      "var(--text-secondary)"))
        q_children = [
            html.Strong("Quality screen: "),
            html.Span(label, style={"color": color, "fontWeight": "600"}),
            html.Span(f" — {quality['total_fails']}/{quality['total_checks']}"
                      f" checks failed"),
        ]
        failed = quality.get("failed_checks") or []
        if failed:
            q_children.append(html.Ul(
                [html.Li(f"{f['check']}: {f['value']}"
                         + (f" ({f['note']})" if f.get("note") else ""))
                 for f in failed[:8]],
                style={"marginTop": "6px", "marginBottom": "0",
                       "paddingLeft": "18px", "fontSize": "0.8rem",
                       "lineHeight": "1.5",
                       "color": "var(--text-secondary)"},
            ))
        children.append(html.Div(q_children, className="key-developments-content"))

    red_flags = quality.get("red_flags") or []
    if red_flags:
        children.append(html.Div([
            html.Strong("News red flags: "),
            html.Ul(
                [html.Li([
                    html.Span(f"[{h['category']}] ",
                              style={"color": "var(--negative)",
                                     "fontWeight": "600"}),
                    f"{h['headline']}" + (f" ({h['date']})" if h.get("date") else ""),
                ]) for h in red_flags[:6]],
                style={"marginTop": "6px", "marginBottom": "0",
                       "paddingLeft": "18px", "fontSize": "0.8rem",
                       "lineHeight": "1.5"},
            ),
        ], className="key-developments-content", style={"marginTop": "8px"}))

    children.append(html.Div(
        "Positioning and quality shade conviction and sizing — neither is a "
        "standalone timing signal.",
        className="research-report-footnote",
    ))
    return html.Div(children, className="key-developments")


def build_tab_content(
    articles: list,
    analysis: dict,
    symbols: list,
    is_overall: bool = False,
    model_signals: dict | None = None,
    strategy_metrics: list | None = None,
    strategy_evaluations: list | None = None,
    ai_failed: bool = False,
    recommendations: dict | None = None,
) -> html.Div:
    """Build content for a single tab (overall or per-symbol).

    Args:
        articles: List of article dictionaries for this tab
        analysis: AI analysis dictionary for this tab
        symbols: List of symbols (single symbol for per-symbol tab)
        is_overall: True if this is the Overall tab
        model_signals: Model prediction results for this symbol

    Returns:
        Tab content component
    """
    children = []

    # -- Unify the two research-text sources --
    # The research report reaches this tab through EITHER the AI Report flow
    # (analysis["research"]) OR the prediction pipeline (trading_agents entry
    # in the signals store — the only path in Full Analysis, which no longer
    # runs the shallow per-symbol pass). One resolution, one rendering.
    research = (analysis or {}).get("research") or {}
    if not research.get("raw_response") and model_signals and not is_overall:
        ta_sig = model_signals.get("trading_agents")
        if isinstance(ta_sig, dict) and not ta_sig.get("error"):
            det = ta_sig.get("details") or {}
            if det.get("raw_response"):
                research = {
                    "decision": ta_sig.get("decision", "HOLD"),
                    "confidence": ta_sig.get("confidence"),
                    "raw_response": det["raw_response"],
                    "triggers": det.get("triggers") or {},
                    "structured": det.get("structured") or {},
                    "provenance": det.get("provenance") or {},
                    "model": det.get("model", ""),
                }

    # When the shallow pass didn't run, the research epilogue supplies the
    # banner stance and the watch/thesis panels — same fields, same renderer,
    # one analyst voice. Existing shallow keys always win (dict-merge order).
    if research.get("raw_response"):
        st = research.get("structured") or {}
        derived = {"research": research}
        if st.get("stance"):
            derived["recommendation"] = st["stance"]
            derived["stance_source"] = "research_verdict"
        if research.get("confidence") is not None:
            derived["confidence"] = research["confidence"]
        if st.get("sentiment_alignment"):
            derived["sentiment_explanation"] = st["sentiment_alignment"]
        if st.get("watch_items"):
            derived["watch_items"] = st["watch_items"]
        if st.get("company_thesis"):
            derived["company_thesis"] = st["company_thesis"]
        if research.get("provenance"):
            derived["provenance"] = research["provenance"]
        analysis = {**derived, **(analysis or {})}

    # -- Model Signal Cards (per-symbol tabs only) --
    if model_signals and not is_overall:
        symbol = symbols[0] if symbols else ""
        signal_cards = create_signal_cards(model_signals, symbol=symbol)
        children.append(signal_cards)

    # -- Per-symbol Recommendations (Full Analysis) --
    if recommendations and not is_overall:
        rec_section = create_recommendations_section(recommendations, symbol=symbols[0] if symbols else None)
        if rec_section:
            children.append(rec_section)

    # -- Strategy Performance (per-symbol tabs only) --
    if not is_overall and (strategy_metrics or strategy_evaluations):
        strategy_section = create_strategy_section(
            strategy_metrics or [], strategy_evaluations or []
        )
        children.append(strategy_section)

    # -- Recommendation Banner --
    if analysis and analysis.get("recommendation"):
        rec_banner = create_recommendation_banner(
            recommendation=analysis.get("recommendation", "NEUTRAL"),
            confidence=analysis.get("confidence"),
            article_count=len(articles),
            date_range=get_date_range(articles),
        )
    elif articles:
        # Show loading state while waiting for AI analysis
        rec_banner = create_recommendation_banner(recommendation="LOADING")
    else:
        rec_banner = None

    if rec_banner:
        children.append(rec_banner)

    # -- Research Report — the verdict-first deep dive (either source) --
    if research.get("raw_response"):
        from models.single_agent import strip_epilogue
        r_dec = research.get("decision", "HOLD")
        r_cls = ("positive" if r_dec == "BUY"
                 else "negative" if r_dec == "SELL" else "neutral")
        body = strip_epilogue(research["raw_response"])
        # New reports end with their own "Compiled by …" sources footer; only
        # older persisted reports need the model named in the footnote.
        footnote = "Saved to History → TradingAgents Reports (PDF available there)."
        if "Compiled by" not in body and research.get("model"):
            footnote = (f"Model: {research['model']}. " + footnote)
        children.append(html.Div(
            [
                html.Div(
                    [
                        html.I(className="bi bi-journal-richtext me-2"),
                        html.Span("Research Report", className="section-title mb-0"),
                        html.Span(
                            f"{r_dec} · {research.get('confidence', 0):.0%}",
                            className=f"research-verdict-badge {r_cls}",
                        ),
                    ],
                    className="research-report-header",
                ),
                dcc.Markdown(
                    body,
                    className="ta-report-body",
                    style={"maxHeight": "420px", "overflowY": "auto",
                           "fontSize": "0.82rem", "lineHeight": "1.55"},
                ),
                html.Div(footnote, className="research-report-footnote"),
            ],
            className="key-developments research-report-section",
        ))

    # -- Options Positioning & Quality Screen (per-symbol tabs only) --
    if not is_overall:
        pq_section = create_positioning_quality_section(analysis)
        if pq_section:
            children.append(pq_section)

    # -- Key Developments --
    if analysis and analysis.get("key_developments"):
        kd_children = [
            html.Div("Key Developments", className="section-title"),
            html.Div(
                analysis.get("key_developments", ""),
                className="key-developments-content",
            ),
        ]
        # v3 interpretation line — old cached payloads simply lack the key
        if analysis.get("developments_read"):
            kd_children.append(html.Div(
                [html.I(className="bi bi-arrow-return-right me-1"),
                 analysis["developments_read"]],
                className="analysis-read-line",
            ))
        children.append(html.Div(kd_children, className="key-developments"))

        if analysis.get("risk_factors"):
            risk_children = [
                html.Div(
                    [html.I(className="bi bi-exclamation-triangle me-2"), "Risk Factors"],
                    className="section-title",
                    style={"color": "var(--negative)"},
                ),
                html.Div(
                    analysis["risk_factors"],
                    className="key-developments-content",
                    style={"borderLeft": "3px solid var(--negative)", "paddingLeft": "10px"},
                ),
            ]
            if analysis.get("risks_read"):
                risk_children.append(html.Div(
                    [html.I(className="bi bi-arrow-return-right me-1"),
                     analysis["risks_read"]],
                    className="analysis-read-line",
                ))
            children.append(html.Div(risk_children, className="key-developments"))

        # Shallow-tier provenance: name the compiling model and its inputs so
        # the reader knows this analysis is news+metrics, not deep research.
        if analysis.get("model_used"):
            src = analysis.get("sources") or {}
            bits = [f"Compiled by {analysis['model_used']}"]
            if src.get("articles") is not None:
                bits.append(f"{src['articles']} articles")
            if src.get("validated_blocks"):
                bits.append("validated " + "/".join(src["validated_blocks"]))
            if src.get("analysis_tier") == "sentiment_fallback":
                bits.append("sentiment-count fallback (LLM parse failed)")
            children.append(html.Div(
                " · ".join(bits),
                className="research-report-footnote",
                style={"marginTop": "-4px", "marginBottom": "8px"},
            ))
    elif articles and not analysis:
        if ai_failed:
            children.append(create_ai_failure_indicator())
        else:
            children.append(create_ai_loading_indicator())

    # -- Watch Items / Company Thesis — from either tier (shallow JSON or the
    # research epilogue); rendered identically so there is ONE report style.
    if analysis:
        watch = analysis.get("watch_items") or []
        if isinstance(watch, list) and watch:
            children.append(html.Div(
                [
                    html.Div(
                        [html.I(className="bi bi-binoculars me-2"), "Watch Items"],
                        className="section-title",
                    ),
                    html.Ul(
                        [html.Li(str(w)) for w in watch[:4]],
                        className="watch-items-list",
                    ),
                ],
                className="key-developments",
            ))

        thesis = analysis.get("company_thesis") or {}
        if thesis:
            thesis_children = [
                html.Div(
                    [html.I(className="bi bi-building me-2"), "Company Thesis"],
                    className="section-title",
                ),
            ]
            if thesis.get("perception"):
                thesis_children.append(html.Div([
                    html.Strong("Market Perception: "), thesis["perception"],
                ], className="key-developments-content", style={"marginBottom": "8px"}))
            if thesis.get("goal_alignment"):
                thesis_children.append(html.Div([
                    html.Strong("Goal Alignment: "), thesis["goal_alignment"],
                ], className="key-developments-content", style={"marginBottom": "8px"}))
            catalysts = []
            for c in (thesis.get("positive_catalysts") or []):
                catalysts.append(html.Li(f"▲ {c}", style={"color": "var(--positive)"}))
            for c in (thesis.get("negative_catalysts") or []):
                catalysts.append(html.Li(f"▼ {c}", style={"color": "var(--negative)"}))
            if catalysts:
                thesis_children.append(html.Div([
                    html.Strong("Catalysts:"),
                    html.Ul(catalysts, style={"marginBottom": "8px", "paddingLeft": "18px",
                                              "fontSize": "0.85rem", "lineHeight": "1.5"}),
                ], className="key-developments-content"))
            if thesis.get("regime_risks"):
                thesis_children.append(html.Div([
                    html.Strong("Regime / Systematic Risks: "), thesis["regime_risks"],
                ], className="key-developments-content"))
            children.append(html.Div(thesis_children, className="key-developments"))

    # -- Top Headlines --
    if articles:
        top_headlines = create_top_headlines(articles, max_count=5)
        children.append(top_headlines)

    # -- Sentiment Breakdown --
    sentiment_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for a in articles:
        s = (a.get("sentiment") or "neutral").lower()
        if "bullish" in s:
            sentiment_counts["bullish"] += 1
        elif "bearish" in s:
            sentiment_counts["bearish"] += 1
        else:
            sentiment_counts["neutral"] += 1

    if any(sentiment_counts.values()):
        sentiment_breakdown = create_sentiment_breakdown(
            bullish=sentiment_counts["bullish"],
            neutral=sentiment_counts["neutral"],
            bearish=sentiment_counts["bearish"],
        )
        children.append(sentiment_breakdown)

    # -- Quick Stats --
    if articles:
        sources = list(set(a.get("source", "") for a in articles if a.get("source")))
        quick_stats = create_news_quick_stats(
            article_count=len(articles),
            source_count=len(sources),
            date_range=get_date_range(articles),
            symbols=symbols if is_overall else None,
        )
        children.append(quick_stats)

    # Handle empty state for this tab
    if not children:
        label = "all stocks" if is_overall else symbols[0] if symbols else "this stock"
        children.append(
            html.Div(
                [
                    html.I(className="bi bi-newspaper", style={"fontSize": "24px", "opacity": "0.5", "marginBottom": "8px"}),
                    html.P(f"No news available for {label}", style={"color": "#6B7280", "margin": "0"}),
                ],
                className="tab-empty-state",
                style={"textAlign": "center", "padding": "32px 16px"},
            )
        )

    return html.Div(children, className="tab-content-inner")


def create_main_content() -> html.Div:
    """Create the main content area with charts.

    Returns:
        Main content div component.
    """
    return html.Div(
        [
            # Summary Cards Row
            html.Div(
                id="summary-cards",
                className="summary-cards-row",
            ),

            # Main Price Chart
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(id="chart-title", className="chart-title"),
                            html.Div(id="chart-subtitle", className="chart-subtitle"),
                        ],
                        className="chart-header",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="price-chart",
                            className="main-chart",
                            style={"height": "380px"},
                            config={
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": [
                                    "lasso2d",
                                    "select2d",
                                ],
                                "displaylogo": False,
                                "responsive": False,
                            },
                        ),
                        type="circle",
                        color="#00D4AA",
                    ),
                ],
                className="chart-container price-chart-container",
            ),

            # Technical Indicators Row
            html.Div(
                [
                    # MACD Chart
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("MACD", className="subplot-title"),
                                    html.I(
                                        className="bi bi-info-circle ms-2 info-icon",
                                        id="macd-info-icon",
                                    ),
                                    dbc.Tooltip(
                                        "Moving Average Convergence Divergence: "
                                        "MACD Line = EMA(12) - EMA(26), "
                                        "Signal Line = 9-day EMA of MACD. "
                                        "Bullish when MACD crosses above Signal.",
                                        target="macd-info-icon",
                                        placement="top",
                                    ),
                                ],
                                className="subplot-title-row",
                            ),
                            dcc.Loading(
                                dcc.Graph(
                                    id="macd-chart",
                                    className="subplot-chart",
                                    style={"height": "140px"},
                                    config={"displayModeBar": False, "responsive": False},
                                ),
                                type="circle",
                                color="#00D4AA",
                            ),
                        ],
                        className="subplot-container",
                    ),
                    # RSI Chart
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("RSI", className="subplot-title"),
                                    html.I(
                                        className="bi bi-info-circle ms-2 info-icon",
                                        id="rsi-info-icon",
                                    ),
                                    dbc.Tooltip(
                                        "Relative Strength Index (14-day): "
                                        "Measures momentum on a 0-100 scale. "
                                        "Overbought: >70, Oversold: <30.",
                                        target="rsi-info-icon",
                                        placement="top",
                                    ),
                                ],
                                className="subplot-title-row",
                            ),
                            dcc.Loading(
                                dcc.Graph(
                                    id="rsi-chart",
                                    className="subplot-chart",
                                    style={"height": "140px"},
                                    config={"displayModeBar": False, "responsive": False},
                                ),
                                type="circle",
                                color="#00D4AA",
                            ),
                        ],
                        className="subplot-container",
                    ),
                ],
                className="indicators-row",
            ),

            # Volume Chart
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Volume", className="subplot-title"),
                            html.I(
                                className="bi bi-info-circle ms-2 info-icon",
                                id="volume-info-icon",
                            ),
                            dbc.Tooltip(
                                "Trading Volume: Number of shares traded. "
                                "Green = up day, Red = down day. "
                                "Line shows 20-day moving average.",
                                target="volume-info-icon",
                                placement="top",
                            ),
                        ],
                        className="subplot-title-row",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="volume-chart",
                            className="volume-chart",
                            style={"height": "110px"},
                            config={"displayModeBar": False, "responsive": False},
                        ),
                        type="circle",
                        color="#00D4AA",
                    ),
                ],
                className="chart-container volume-container",
            ),
        ],
        className="main-content",
        id="main-content",
    )


def create_context_panel() -> html.Div:
    """Create the right context panel with dynamic per-symbol tabs.

    Returns:
        Context panel div component with:
        - Panel header with LLM status
        - Dynamic tabs container (populated by callback based on selected symbols)

    Tab structure: [ Overall ] [ AAPL ] [ GOOGL ] [ MSFT ] ...
    Each symbol gets its own tab with recommendation + news + sentiment.
    """
    return html.Div(
        [
            # Panel header (stays outside tabs)
            html.Div(
                [
                    html.H2("Analysis", className="panel-main-title"),
                    html.Span(id="llm-status", className="llm-status-badge"),
                    # Background prediction running indicator
                    html.Span(
                        [
                            html.I(className="bi bi-gear-fill spinning-icon"),
                            " Predicting...",
                        ],
                        id="prediction-running-indicator",
                        className="prediction-running-badge",
                        style={"display": "none"},
                    ),
                    # Hidden — keeps download_report_pdf callback wired
                    html.Button(id="download-report-btn", style={"display": "none"}),
                ],
                className="panel-header-row",
            ),

            # Dynamic tabs container - populated by update_symbol_tabs callback
            dcc.Loading(
                html.Div(id="symbol-tabs-container"),
                type="circle",
                color="#00D4AA",
            ),
        ],
        className="context-panel",
        id="context-panel",
    )


def layout(period: str = "1y") -> html.Div:
    """The working surface: charts on the left, per-symbol analysis on the right.

    The period selector and the data actions (refresh / export / view data)
    live here rather than in the global toolbar for the same reason the
    indicator toggles always did: they apply to these charts and this data,
    and nowhere else. `period` seeds the selector from the current-period
    store so navigating here reflects the persisted choice.
    """
    return html.Div(
        [
            html.Div(
                [
                    create_period_selector(period if period else "1y"),
                    create_data_actions(include_refresh=True),
                ],
                className="analyze-action-bar",
            ),
            html.Div(
                [
                    html.Span("Indicators", className="input-label"),
                    create_indicator_toggles(),
                ],
                className="analyze-indicator-bar",
            ),
            html.Div(
                [create_main_content(), create_context_panel()],
                className="analyze-grid",
            ),
        ],
        className="page page-analyze",
    )
