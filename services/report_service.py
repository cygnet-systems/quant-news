"""Report generation service. HTML-to-PDF trading analysis reports.

Generates professional analysis reports styled after TradingAgents output.
Reports include: executive summary, model predictions with timestamps,
AI sentiment analysis, recommendations, and conflict analysis.
"""

import html
import io
import logging
from datetime import datetime
from typing import Optional

from xhtml2pdf import pisa

logger = logging.getLogger(__name__)


def _esc(value) -> str:
    """HTML-escape any model/user-derived string before it enters the report.

    LLM output (summaries, reasoning, conflicts) can contain '<', '>' or '&',
    which break the xhtml2pdf layout and are an injection vector. Numeric/enum
    fields are safe but escaping them is harmless.
    """
    return html.escape(str(value)) if value is not None else ""

_MODEL_DESCRIPTIONS = {
    "kronos_mini": "Kronos Mini",
    "xgboost_shap": "XGBoost SHAP",
    "lightgbm": "LightGBM",
    "deberta_sentiment": "DeBERTa Sentiment",
    "trading_agents": "TradingAgents",
    "ensemble": "Ensemble",
    "recommendation_synthesis": "Recommendations (LLM)",
}

_MODEL_TYPES = {
    "kronos_mini": "Time-series foundation model",
    "xgboost_shap": "Gradient-boosted tree + SHAP",
    "lightgbm": "Gradient-boosted tree",
    "deberta_sentiment": "Transformer sentiment",
    "trading_agents": "Single-agent LLM research",
    "ensemble": "Weighted consensus",
    "recommendation_synthesis": "LLM synthesis (reports + signals)",
}


def generate_report_pdf(
    symbols: list[str],
    ai_analysis: dict | None = None,
    model_signals: dict | None = None,
    recommendations: dict | None = None,
    news_data: dict | None = None,
    target_date: str | None = None,
    data_through: str | None = None,
) -> Optional[bytes]:
    """Generate the unified PDF report from current analysis data.

    One template for every flow: compilation provenance up front, executive
    summary, model predictions, then per-symbol chapters that carry the full
    research report when one exists (falling back to the news-summary tier).

    target_date/data_through are the run's own dates. Passing them is
    preferred: without them the report can only infer dates from the
    predictions payload, which is not necessarily the run's.
    Returns PDF bytes or None on failure.
    """
    html = _build_report_html(symbols, ai_analysis, model_signals, recommendations,
                              news_data, target_date, data_through)
    return _html_to_pdf(html)


def _render_markdown_report(text: str) -> str:
    """Render a research-report markdown body to HTML for the PDF.

    Strips the machine-read epilogue, fixes the missing-blank-line-before-list
    habit of LLM output, and demotes headings so report sections nest under the
    per-symbol chapter headings instead of competing with them.
    """
    import re
    import markdown as md
    from models.single_agent import render_report_markdown

    # render_report_markdown, not strip_epilogue: it also turns the verdict's
    # field lines into list items, which is what stops the whole verdict block
    # collapsing into a single run-on paragraph here.
    body = render_report_markdown(text or "")
    body = re.sub(
        r"(?m)^(?![ \t]*[-*+][ \t])(.*\S.*)\n([ \t]*[-*+][ \t])",
        r"\1\n\n\2",
        body,
    )
    body = re.sub(r"(?m)^### ", "##### ", body)
    body = re.sub(r"(?m)^## ", "#### ", body)
    return md.markdown(body, extensions=["tables", "fenced_code"])


def _html_to_pdf(html: str) -> Optional[bytes]:
    buf = io.BytesIO()
    status = pisa.CreatePDF(io.StringIO(html), dest=buf)
    if status.err:
        logger.error(f"PDF generation failed: {status.err}")
        return None
    return buf.getvalue()


def _build_ta_predictions_section(predictions: list[dict]) -> str:
    """Render model prediction signals for the symbol as a table.

    Dates go through _esc, not html.escape: these fields arrive as `date`
    objects from some cache accessors, and html.escape() on a date raises
    (it calls date.replace("&", "&amp;")), which took the whole PDF down
    and silently degraded the download to Markdown.
    """
    rows = ["<h2>Model Predictions</h2>"]
    first = predictions[0] if predictions else {}
    pred_date = first.get("prediction_date", "")
    target_date = first.get("target_date", "")
    # Always rendered: a missing date shows as ", " rather than the whole
    # line vanishing, so a broken date is visible instead of silent.
    meta = f"<strong>Target date (close being predicted):</strong> {_esc(target_date) or ', '}"
    meta += f" &nbsp;&nbsp;<strong>Data through:</strong> {_esc(pred_date) or ', '}"
    rows.append(f"<p class='meta'>{meta}</p>")

    rows.append("<table><tr><th>Model</th><th>Signal</th><th>Confidence</th>"
                "<th>Up Prob</th><th>Target</th><th>Result</th></tr>")
    for p in predictions:
        conf = p.get("confidence")
        conf_str = f"{int(conf * 100)}%" if conf is not None else "n/a"
        up = p.get("up_probability")
        up_str = f"{up:.2f}" if up is not None else "n/a"
        pnl = p.get("pnl_dollars")
        correct = p.get("was_correct")
        if pnl is not None:
            result = f"${pnl:+.2f}"
        elif correct is not None:
            result = "Correct" if correct else "Wrong"
        else:
            result = "Pending"
        rows.append(
            f"<tr><td>{_esc(p.get('model_name', ''))}</td>"
            f"<td>{_esc(p.get('decision', ''))}</td>"
            f"<td>{conf_str}</td><td>{up_str}</td>"
            f"<td>{_esc(p.get('target_date', '')) or ', '}</td><td>{result}</td></tr>"
        )
    rows.append("</table><hr/>")
    return "\n".join(rows)


def _build_ta_recommendation_section(recommendation: dict) -> str:
    """Render the recommendation-model (Luna) reasoning for the symbol."""
    model_used = recommendation.get("model_used", "")
    created_at = str(recommendation.get("created_at", ""))[:19]
    sym_rec = recommendation.get("symbol_rec", {}) or {}
    overall = recommendation.get("overall", {}) or {}

    rows = ["<h2>Recommendation Synthesis</h2>"]
    meta_bits = []
    if model_used:
        meta_bits.append(f"<strong>Model:</strong> {_esc(model_used)}")
    if created_at:
        meta_bits.append(f"<strong>Generated:</strong> {_esc(created_at)}")
    if meta_bits:
        rows.append(f"<p class='meta'>{' &nbsp;&nbsp; '.join(meta_bits)}</p>")

    action = sym_rec.get("action", "")
    conviction = sym_rec.get("conviction")
    if action:
        # Conviction is a HIGH/MEDIUM/LOW label, not a number.
        conviction_str = (f" &nbsp;&nbsp;<strong>Conviction:</strong> {_esc(conviction)}"
                          if conviction is not None else "")
        rows.append(f"<div class='action-box'><strong>Action:</strong> {_esc(action)}{conviction_str}</div>")

    reasoning = sym_rec.get("reasoning", "")
    if reasoning:
        rows.append(f"<h3>Reasoning</h3><p>{_esc(reasoning)}</p>")

    conflicts = sym_rec.get("conflicts", [])
    if conflicts:
        rows.append("<h3>Identified Conflicts</h3><ol>")
        for c in conflicts:
            rows.append(f"<li>{_esc(c)}</li>")
        rows.append("</ol>")

    model_notes = sym_rec.get("model_notes", "")
    if model_notes:
        rows.append(f"<h3>Model Notes</h3><p>{_esc(model_notes)}</p>")

    risk = overall.get("risk_assessment", "")
    if risk:
        rows.append(f"<h3>Risk Assessment</h3><p>{_esc(risk)}</p>")

    rows.append("<hr/>")
    return "\n".join(rows)


def generate_ta_report_pdf(
    report: dict,
    predictions: list[dict] | None = None,
    recommendation: dict | None = None,
) -> Optional[bytes]:
    """Render a single TradingAgents report (Markdown text) as a PDF.

    Optionally includes the symbol's model prediction signals and the
    recommendation-model reasoning alongside the full agent analysis.
    Returns PDF bytes or None on failure.
    """
    from models.single_agent import extract_confidence, parse_epilogue

    symbol = report.get("symbol", "UNKNOWN")
    decision = report.get("decision", "HOLD")
    confidence = report.get("confidence", 0)
    trade_date = report.get("trade_date", "")
    report_text = report.get("report_text", "")
    model_name = report.get("model_name", "")
    created_at = str(report.get("created_at", ""))[:19]

    # Two different numbers, and the header used to conflate them: the stored
    # confidence is the track-record-grounded reliability weight (0.5 until a
    # record exists: it is NOT the model saying "50% sure"), while the LLM's
    # own conviction lives in the report text's CONFIDENCE line.
    conf_pct = int((confidence or 0) * 100)
    stated = extract_confidence(report_text)
    stated_pct = int(round(stated * 100)) if stated is not None else None

    # The machine-read epilogue is rendered as a proper panel, not raw JSON.
    structured = parse_epilogue(report_text) or {}
    body_html = _render_markdown_report(report_text)

    epilogue_html = ""
    # The two numbers lead the At a Glance panel, each named. A reader who
    # opens only this panel must not have to guess which percentage is which.
    ep_bits = [
        "<p><strong>Conviction (this report's own):</strong> "
        + (f"{stated:.2f}" if stated is not None else "not stated")
        + " &nbsp;&nbsp; <strong>Track-record weight (measured):</strong> "
        + ("unrated: not enough resolved calls yet"
           if confidence in (None, 0.5) else f"{conf_pct}%")
        + "</p><p class='meta'>Conviction is the report's own estimate and has "
          "never been scored; the track-record weight is this model's measured "
          "hit rate on resolved calls.</p>"
    ]
    stance = structured.get("stance")
    if stance:
        ep_bits.append(f"<p><strong>Stance:</strong> {_esc(stance)}</p>")
    if structured.get("sentiment_alignment"):
        ep_bits.append(f"<p><strong>News vs. Technicals:</strong> "
                       f"{_esc(structured['sentiment_alignment'])}</p>")
    watch = structured.get("watch_items") or []
    if isinstance(watch, list) and watch:
        ep_bits.append("<p><strong>Watch Items:</strong></p><ul>"
                       + "".join(f"<li>{_esc(w)}</li>" for w in watch[:4])
                       + "</ul>")
    thesis = structured.get("company_thesis") or {}
    if isinstance(thesis, dict) and thesis:
        ep_bits.extend(_sym_thesis_html(thesis))
    if ep_bits:
        epilogue_html = "<h2>At a Glance</h2>" + "\n".join(ep_bits) + "<hr/>"

    sections = []
    if predictions:
        sections.append(_build_ta_predictions_section(predictions))
    if recommendation:
        sections.append(_build_ta_recommendation_section(recommendation))

    extra_sections = "\n".join(sections)
    analysis_heading = ("<h2>Research Analysis</h2>"
                        if (sections or epilogue_html) else "")

    model_line = f"&nbsp;&nbsp;<strong>Model:</strong> {model_name}" if model_name else ""
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
{_get_pdf_css()}
</style>
</head>
<body>
<div class="header">
    <div class="header-meta">
        <span class="filename">{symbol}_trading_agents_{trade_date}.pdf</span>
        <span class="date">{trade_date}</span>
    </div>
    <h1>{symbol} TradingAgents Report</h1>
    <div class="subtitle">
        <strong>Decision:</strong> {decision}
        {f'&nbsp;&nbsp;<strong>Conviction (report&#39;s own):</strong> {stated_pct}%' if stated_pct is not None else ''}
        &nbsp;&nbsp;<strong>Track-record weight (measured):</strong> {'unrated' if confidence in (None, 0.5) else f'{conf_pct}%'}
        &nbsp;&nbsp;<strong>Trade Date:</strong> {trade_date}
        {model_line}
    </div>
    <hr/>
</div>
{epilogue_html}
{extra_sections}
{analysis_heading}
{body_html}
<div class="footer">
    <em>Generated {created_at} by QuantNews TradingAgents</em>
</div>
</body>
</html>"""
    return _html_to_pdf(html)


def _build_report_html(
    symbols: list[str],
    ai_analysis: dict | None,
    model_signals: dict | None,
    recommendations: dict | None,
    news_data: dict | None,
    target_date: str | None = None,
    data_through: str | None = None,
) -> str:
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    symbols_str = ", ".join(symbols)

    sections = []

    # --- Header ---
    sections.append(f"""
    <div class="header">
        <div class="header-meta">
            <span class="filename">QuantNews_Report.pdf</span>
            <span class="date">{now.strftime('%Y-%m-%d')}</span>
        </div>
        <h1>{symbols_str} Trading Analysis Report</h1>
        <div class="subtitle">
            <strong>Date:</strong> {date_str}
            &nbsp;&nbsp;<strong>Symbols:</strong> {symbols_str}
            {_fmt_final_rec(recommendations)}
        </div>
        <hr/>
    </div>
    """)

    # --- How This Report Was Compiled (transparency first) ---
    sections.append(_build_provenance_box(
        symbols, ai_analysis, model_signals, recommendations,
    ))

    # --- I. Executive Summary ---
    sections.append(_build_executive_summary(recommendations, ai_analysis, symbols))

    # --- II. Model Predictions ---
    sections.append(_build_model_predictions_section(model_signals, symbols,
                                                     target_date, data_through))

    # --- III. Per-Symbol Analysis (research chapters or news-summary tier) ---
    sections.append(_build_ai_analysis_section(ai_analysis, symbols))

    # --- IV. Recommendations (if available) ---
    if recommendations and recommendations.get("overall"):
        sections.append(_build_recommendations_section(recommendations))

    # --- V. Analysis Metadata ---
    sections.append(_build_metadata_section(
        ai_analysis, model_signals, recommendations, timestamp,
    ))

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
{_get_pdf_css()}
</style>
</head>
<body>
{body}
<div class="footer">
    <em>Report generated by QuantNews</em>
</div>
</body>
</html>"""


def _fmt_final_rec(recommendations: dict | None) -> str:
    if not recommendations or not recommendations.get("overall"):
        return ""
    action = recommendations["overall"].get("portfolio_action", "")
    if not action:
        return ""
    short = action[:80] + "..." if len(action) > 80 else action
    return f'&nbsp;&nbsp;<strong>Portfolio Action:</strong> {_esc(short)}'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_provenance_box(
    symbols: list[str],
    ai_analysis: dict | None,
    model_signals: dict | None,
    recommendations: dict | None,
) -> str:
    """Render the 'how this report was compiled' box.

    Transparency contract: every layer of the report names the model that
    wrote it and the data it was given. All values here are computed from the
    payloads (stamped at generation time), never asserted by an LLM.
    """
    rows = ["<h2>How This Report Was Compiled</h2>"]
    items: list[tuple[str, str, str]] = []  # (layer, model, inputs)

    by_symbol = (ai_analysis or {}).get("by_symbol", {}) or {}

    # Per-symbol research reports (deep tier)
    # One row per symbol: the inputs differ per symbol (article counts,
    # bars), and the old single string generalised the first symbol's
    # numbers to every other row.
    for sym in symbols:
        research = (by_symbol.get(sym) or {}).get("research") or {}
        if not research.get("raw_response"):
            continue
        prov = research.get("provenance") or {}
        model = research.get("model") or prov.get("model") or "unknown model"
        if prov.get("news_enabled", True):
            read = prov.get("news_prompt_articles")
            news = (f"{read if read is not None else '?'} of "
                    f"{prov.get('news_count', '?')} news articles read "
                    f"({prov.get('news_window_days', '?')}d point-in-time window)")
        else:
            news = "news disabled"
        bars = prov.get("ohlcv_bars")
        through = prov.get("ohlcv_through")
        ohlcv = (f"{bars} OHLCV bars through {through}" if bars else "OHLCV")
        items.append((f"Research report ({sym})", model,
                      f"{ohlcv}, SPY & sector-ETF context, fundamentals "
                      f"(as-of filtered), {news}, precomputed "
                      f"metrics/events/peers"))

    # Shallow news-summary tier (per-symbol entries without research)
    shallow_syms = [s for s in symbols
                    if (by_symbol.get(s) or {}).get("key_developments")
                    and not (by_symbol.get(s) or {}).get("research")]
    if shallow_syms:
        first = by_symbol.get(shallow_syms[0]) or {}
        src = first.get("sources") or {}
        blocks = ", ".join(src.get("validated_blocks") or []) or "news only"
        items.append((f"News analysis ({', '.join(shallow_syms)})",
                      first.get("model_used") or "unknown model",
                      f"news articles + validated blocks ({blocks})"))

    overall = (ai_analysis or {}).get("overall") or {}
    if overall:
        src = overall.get("sources") or {}
        n_art = src.get("articles")
        items.append(("Portfolio overview",
                      overall.get("model_used") or "unknown model",
                      f"{n_art if n_art is not None else '?'} articles across "
                      f"{len(symbols)} symbols"))

    if model_signals:
        pred_models = sorted({
            name for sym, models in model_signals.items()
            if sym != "_meta" and isinstance(models, dict)
            for name, r in models.items()
            if isinstance(r, dict) and not r.get("error")
        })
        if pred_models:
            labels = [_MODEL_DESCRIPTIONS.get(m, m) for m in pred_models]
            items.append(("Model predictions", ", ".join(labels),
                          "point-in-time OHLCV (+ SPY / news features where used)"))

    if recommendations and recommendations.get("overall"):
        basis = {
            "research+signals": "research reports + model predictions",
            "news+signals": "news analysis + model predictions",
            "signals": "model predictions only",
        }.get(recommendations.get("basis"), "text analysis + model predictions")
        items.append(("Recommendation synthesis",
                      recommendations.get("model_used") or "unknown model", basis))

    if not items:
        return ""

    rows.append("<table><thead><tr><th>Layer</th><th>Model</th>"
                "<th>Inputs</th></tr></thead><tbody>")
    for layer, model, inputs in items:
        rows.append(f"<tr><td><strong>{_esc(layer)}</strong></td>"
                    f"<td>{_esc(model)}</td><td>{_esc(inputs)}</td></tr>")
    rows.append("</tbody></table>")
    rows.append(
        "<p class='meta'>All figures in the analyses below come from the data "
        "blocks listed here (OHLCV via market data vendor, point-in-time news "
        "windows, computed Wilder RSI/ATR indicators, as-of-filtered "
        "fundamentals). Analyses flag any value as \"not in data\" rather than "
        "estimate it. Model predictions and LLM analyses are research aids, "
        "not investment advice.</p>"
    )
    rows.append("<hr/>")
    return "\n".join(rows)


def _build_executive_summary(
    recommendations: dict | None,
    ai_analysis: dict | None,
    symbols: list[str],
) -> str:
    rows = []

    if recommendations and recommendations.get("overall"):
        overall = recommendations["overall"]
        summary = overall.get("summary", "No summary available.")
        action = overall.get("portfolio_action", "N/A")
        risk = overall.get("risk_assessment", "")
        model_used = recommendations.get("model_used", "N/A")

        rows.append(f"""
        <h2>I. Executive Summary</h2>
        <h3>Portfolio Recommendation</h3>
        <p><strong>Action:</strong> {_esc(action)}</p>
        <p><strong>Synthesized by:</strong> {_esc(model_used)}</p>
        <p>{_esc(summary)}</p>
        """)

        if risk:
            rows.append(f"<p><strong>Risk Assessment:</strong> {_esc(risk)}</p>")

        conflicts = overall.get("key_conflicts", [])
        if conflicts:
            rows.append("<h3>Key Conflicts</h3><ol>")
            for c in conflicts:
                rows.append(f"<li>{_esc(c)}</li>")
            rows.append("</ol>")

        by_symbol = recommendations.get("by_symbol", {})
        if by_symbol:
            rows.append("""
            <h3>Per-Symbol Recommendations</h3>
            <table>
            <thead><tr>
                <th>Symbol</th><th>Action</th><th>Conviction</th><th>Reasoning</th>
            </tr></thead><tbody>
            """)
            for sym in symbols:
                sr = by_symbol.get(sym, {})
                rows.append(f"""<tr>
                    <td><strong>{_esc(sym)}</strong></td>
                    <td>{_esc(sr.get('action', 'N/A'))}</td>
                    <td>{_esc(sr.get('conviction', 'N/A'))}</td>
                    <td>{_esc(sr.get('reasoning', 'N/A'))}</td>
                </tr>""")
            rows.append("</tbody></table>")

    elif ai_analysis and ai_analysis.get("overall"):
        overall = ai_analysis["overall"]
        rec = overall.get("recommendation", "N/A")
        conf = overall.get("confidence", "")
        summary_text = ""
        devs = overall.get("key_developments", [])
        if devs:
            summary_text = devs[0] if isinstance(devs[0], str) else str(devs[0])

        rows.append(f"""
        <h2>I. Executive Summary</h2>
        <p><strong>AI Sentiment Recommendation:</strong> {_esc(rec)}
        {f' ({_esc(conf)}% confidence)' if conf else ''}</p>
        <p>{_esc(summary_text)}</p>
        """)
    else:
        rows.append("<h2>I. Executive Summary</h2><p>No analysis data available.</p>")

    rows.append("<hr/>")
    return "\n".join(rows)


def _build_model_predictions_section(
    model_signals: dict | None,
    symbols: list[str],
    target_date: str | None = None,
    data_through: str | None = None,
) -> str:
    rows = ["<h2>II. Model Predictions</h2>"]

    if not model_signals:
        rows.append("<p>No model predictions available. Run predictions first.</p>")
        rows.append("<hr/>")
        return "\n".join(rows)

    filtered_signals = {
        sym: models for sym, models in model_signals.items()
        if sym != "_meta" and isinstance(models, dict)
    }
    meta = model_signals.get("_meta", {})

    if not filtered_signals:
        rows.append("<p>No model predictions available.</p><hr/>")
        return "\n".join(rows)

    for symbol in symbols:
        sym_models = filtered_signals.get(symbol, {})
        if not sym_models:
            continue

        rows.append(f"<h3>{symbol}: Model Signals</h3>")
        rows.append("""
        <table>
        <thead><tr>
            <th>Model</th><th>Type</th><th>Decision</th>
            <th>Confidence</th><th>Up Prob.</th>
        </tr></thead><tbody>
        """)

        for model_name, result in sym_models.items():
            if not isinstance(result, dict) or result.get("error"):
                continue

            decision = result.get("decision", "N/A")
            confidence = result.get("confidence")
            up_prob = result.get("up_probability")

            conf_str = f"{confidence:.0%}" if confidence is not None else "N/A"
            up_str = f"{up_prob:.0%}" if up_prob is not None else "N/A"

            label = _MODEL_DESCRIPTIONS.get(model_name, model_name)
            mtype = _MODEL_TYPES.get(model_name, "")

            rows.append(f"""<tr>
                <td><strong>{label}</strong></td>
                <td>{mtype}</td>
                <td><strong>{decision}</strong></td>
                <td>{conf_str}</td>
                <td>{up_str}</td>
            </tr>""")

        rows.append("</tbody></table>")

    # Date provenance. The run's own dates win; the payload's are only a
    # fallback, and are labelled when they are merely "latest in the DB"
    # rather than this run's. Always rendered, so a missing date shows as
    # "N/A" instead of the line silently disappearing.
    tgt = target_date or meta.get("target_date") or "N/A"
    through = data_through or meta.get("predict_date") or "N/A"
    stale = not (target_date or data_through) and meta.get("dates_are_latest_available")
    note = " (latest available, not this run)" if stale else ""
    rows.append(f"<p class='meta'><strong>Target date (close being predicted):</strong> "
                f"{_esc(tgt)}{note} &nbsp; <strong>Data through:</strong> {_esc(through)}"
                f" &nbsp; <strong>Generated at:</strong> "
                f"{datetime.now().strftime('%H:%M:%S')}</p>")

    rows.append("<hr/>")
    return "\n".join(rows)


def _sym_thesis_html(thesis: dict) -> list[str]:
    rows = ["<h4>Company Thesis</h4>"]
    if thesis.get("perception"):
        rows.append(f"<p><strong>Market Perception:</strong> {_esc(thesis['perception'])}</p>")
    if thesis.get("goal_alignment"):
        rows.append(f"<p><strong>Goal Alignment:</strong> {_esc(thesis['goal_alignment'])}</p>")
    pos = thesis.get("positive_catalysts") or []
    neg = thesis.get("negative_catalysts") or []
    if pos:
        rows.append("<p><strong>Positive Catalysts:</strong></p><ul>")
        rows.extend(f"<li>{_esc(c)}</li>" for c in pos)
        rows.append("</ul>")
    if neg:
        rows.append("<p><strong>Negative Catalysts:</strong></p><ul>")
        rows.extend(f"<li>{_esc(c)}</li>" for c in neg)
        rows.append("</ul>")
    if thesis.get("regime_risks"):
        rows.append(f"<p><strong>Regime / Systematic Risks:</strong> {_esc(thesis['regime_risks'])}</p>")
    return rows


def _sym_positioning_quality_html(sym: dict) -> list[str]:
    """Options positioning + Bad Apples quality screen for one symbol's
    chapter. Empty when the payload predates these sections."""
    rows: list[str] = []
    pos = sym.get("positioning") or {}
    if pos.get("pc_volume") is not None or pos.get("pc_oi") is not None:
        pcv = f"{pos['pc_volume']:.2f}" if pos.get("pc_volume") is not None else "n/a"
        pcoi = f"{pos['pc_oi']:.2f}" if pos.get("pc_oi") is not None else "n/a"
        rows.append(
            f"<p><strong>Options Positioning</strong> (chain as of "
            f"{_esc(pos.get('as_of', '?'))}): P/C volume {pcv} "
            f"({pos.get('put_volume', 0):,} puts / {pos.get('call_volume', 0):,} calls), "
            f"P/C open interest {pcoi}: {_esc(pos.get('read', ''))}</p>"
        )
    quality = sym.get("quality") or {}
    if quality.get("total_checks"):
        flag = str(quality.get("flag", "")).replace("_", " ").upper()
        rows.append(
            f"<p><strong>Quality Screen (Bad Apples)</strong> as of "
            f"{_esc(quality.get('as_of', '?'))}: {_esc(flag)}: "
            f"{quality['total_fails']}/{quality['total_checks']} checks failed</p>"
        )
        failed = quality.get("failed_checks") or []
        if failed:
            rows.append("<ul>")
            rows.extend(
                f"<li>{_esc(f['check'])}: {_esc(f['value'])}"
                + (f" ({_esc(f['note'])})" if f.get("note") else "") + "</li>"
                for f in failed[:8]
            )
            rows.append("</ul>")
        red_flags = quality.get("red_flags") or []
        if red_flags:
            rows.append("<p><strong>News red flags</strong> "
                        "(keyword-matched headlines):</p><ul>")
            rows.extend(
                f"<li>[{_esc(h['category'])}] {_esc(h['headline'])}"
                + (f" ({_esc(h['date'])})" if h.get("date") else "") + "</li>"
                for h in red_flags[:6]
            )
            rows.append("</ul>")
    if rows:
        rows.append("<p class='meta'>Positioning and quality shade conviction "
                    "and sizing: neither is a standalone timing signal.</p>")
    return rows


def _build_ai_analysis_section(
    ai_analysis: dict | None,
    symbols: list[str],
) -> str:
    rows = ["<h2>III. Per-Symbol Analysis</h2>"]

    if not ai_analysis or ai_analysis.get("failed"):
        rows.append("<p>No AI analysis available.</p><hr/>")
        return "\n".join(rows)

    generated_at = ai_analysis.get("generated_at", "")
    if generated_at:
        rows.append(f"<p class='meta'><strong>Generated:</strong> {_esc(generated_at)}</p>")

    overall = ai_analysis.get("overall", {})
    if overall:
        rec = overall.get("recommendation", "")
        conf = overall.get("confidence", "")
        model_used = overall.get("model_used", "")
        if rec:
            rows.append(f"<p><strong>Portfolio Overview:</strong> {_esc(rec)}"
                        f"{f' ({_esc(conf)}% confidence)' if conf else ''}"
                        f"{f': compiled by {_esc(model_used)}' if model_used else ''}</p>")

        devs = overall.get("key_developments", [])
        if devs:
            rows.append("<h3>Key Developments</h3><ol>")
            for d in (devs if isinstance(devs, list) else [devs]):
                rows.append(f"<li>{_esc(d)}</li>")
            rows.append("</ol>")

        risks = overall.get("risk_factors", [])
        if risks:
            rows.append("<h3>Risk Factors</h3><ol>")
            for r in (risks if isinstance(risks, list) else [risks]):
                rows.append(f"<li>{_esc(r)}</li>")
            rows.append("</ol>")

    # Per-symbol chapters, the full research report when one exists, else
    # the news-summary tier. One template either way.
    by_symbol = ai_analysis.get("by_symbol", {})
    for symbol in symbols:
        sym = by_symbol.get(symbol, {})
        if not sym:
            continue

        research = sym.get("research") or {}
        if research.get("raw_response"):
            decision = research.get("decision", "")
            model_used = research.get("model") or (research.get("provenance") or {}).get("model", "")
            rows.append(f"<h3>{_esc(symbol)}: Research Report</h3>")
            meta_bits = []
            if decision:
                meta_bits.append(f"<strong>Verdict:</strong> {_esc(decision)}")
            stance = (research.get("structured") or {}).get("stance")
            if stance:
                meta_bits.append(f"<strong>Stance:</strong> {_esc(stance)}")
            if model_used:
                meta_bits.append(f"<strong>Model:</strong> {_esc(model_used)}")
            if meta_bits:
                rows.append(f"<p class='meta'>{' &nbsp;&nbsp; '.join(meta_bits)}</p>")
            # The body carries its own "Compiled by … Sources: …" footer.
            rows.append(_render_markdown_report(research["raw_response"]))
        else:
            rec = sym.get("recommendation", "N/A")
            conf = sym.get("confidence", "")
            model_used = sym.get("model_used", "")
            src = sym.get("sources") or {}
            rows.append(f"<h3>{_esc(symbol)}: News Analysis</h3>")
            rows.append(f"<p><strong>Recommendation:</strong> {_esc(rec)}"
                        f"{f' ({_esc(conf)}% confidence)' if conf else ''}</p>")

            devs = sym.get("key_developments", [])
            if devs:
                rows.append("<p><strong>Key Developments:</strong></p><ol>")
                for d in (devs if isinstance(devs, list) else [devs]):
                    rows.append(f"<li>{_esc(d)}</li>")
                rows.append("</ol>")
            if sym.get("developments_read"):
                rows.append(f"<p><em>Read:</em> {_esc(sym['developments_read'])}</p>")

            risks = sym.get("risk_factors")
            if risks:
                rows.append(f"<p><strong>Risk Factors:</strong> {_esc(risks)}</p>")
            if sym.get("risks_read"):
                rows.append(f"<p><em>Read:</em> {_esc(sym['risks_read'])}</p>")

            if model_used:
                blocks = ", ".join(src.get("validated_blocks") or []) or "news only"
                n_art = src.get("articles")
                rows.append(
                    f"<p class='meta'>Compiled by {_esc(model_used)} from "
                    f"{_esc(n_art if n_art is not None else '?')} articles + "
                    f"validated blocks ({_esc(blocks)}).</p>"
                )

        # Shared panels: identical for both tiers.
        rows.extend(_sym_positioning_quality_html(sym))
        if sym.get("sentiment_explanation"):
            rows.append(f"<p><strong>News vs. Technicals:</strong> "
                        f"{_esc(sym['sentiment_explanation'])}</p>")
        watch = sym.get("watch_items") or []
        if isinstance(watch, list) and watch:
            rows.append("<p><strong>Watch Items:</strong></p><ul>")
            rows.extend(f"<li>{_esc(w)}</li>" for w in watch[:4])
            rows.append("</ul>")
        thesis = sym.get("company_thesis") or {}
        if thesis:
            rows.extend(_sym_thesis_html(thesis))

    rows.append("<hr/>")
    return "\n".join(rows)


def _build_recommendations_section(recommendations: dict) -> str:
    rows = ["<h2>IV. Recommendation Synthesis</h2>"]

    overall = recommendations.get("overall", {})
    model_used = recommendations.get("model_used", "N/A")

    rows.append(f"<p class='meta'><strong>Model:</strong> {_esc(model_used)}</p>")

    action = overall.get("portfolio_action", "")
    if action:
        rows.append(f"<div class='action-box'><strong>Portfolio Action:</strong> {_esc(action)}</div>")

    summary = overall.get("summary", "")
    if summary:
        rows.append(f"<p>{_esc(summary)}</p>")

    conflicts = overall.get("key_conflicts", [])
    if conflicts:
        rows.append("<h3>Identified Conflicts</h3><ol>")
        for c in conflicts:
            rows.append(f"<li>{_esc(c)}</li>")
        rows.append("</ol>")

    risk = overall.get("risk_assessment", "")
    if risk:
        rows.append(f"<p><strong>Risk Assessment:</strong> {_esc(risk)}</p>")

    rows.append("<hr/>")
    return "\n".join(rows)


def _build_metadata_section(
    ai_analysis: dict | None,
    model_signals: dict | None,
    recommendations: dict | None,
    timestamp: str,
) -> str:
    rows = ["<h2>Analysis Metadata</h2>"]

    model_count = 0
    if model_signals:
        for sym, models in model_signals.items():
            if sym == "_meta" or not isinstance(models, dict):
                continue
            model_count += sum(
                1 for m in models.values()
                if isinstance(m, dict) and not m.get("error")
            )

    symbol_count = 0
    if model_signals:
        symbol_count = sum(
            1 for k, v in model_signals.items()
            if k != "_meta" and isinstance(v, dict)
        )

    items = [
        ("Report Generated", timestamp),
        ("Symbols Analyzed", str(symbol_count) if symbol_count else "0"),
        ("Model Predictions", str(model_count)),
        ("AI Analysis", "Yes" if ai_analysis and not ai_analysis.get("failed") else "No"),
        ("Recommendations", "Yes" if recommendations and recommendations.get("overall") else "No"),
    ]

    if recommendations:
        items.append(("Recommendations Model", recommendations.get("model_used", "N/A")))

    if ai_analysis and ai_analysis.get("generated_at"):
        items.append(("AI Analysis Generated", ai_analysis["generated_at"]))

    # Models used
    if model_signals:
        models_used = set()
        for sym, models in model_signals.items():
            if sym == "_meta" or not isinstance(models, dict):
                continue
            for name, result in models.items():
                if isinstance(result, dict) and not result.get("error"):
                    models_used.add(name)
        if models_used:
            items.append(("Models Used", ", ".join(sorted(models_used))))

    rows.append("<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>")
    for label, value in items:
        rows.append(f"<tr><td><strong>{label}</strong></td><td>{value}</td></tr>")
    rows.append("</tbody></table>")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def _get_pdf_css() -> str:
    return """
    @page {
        size: A4;
        margin: 2cm 2.5cm;
    }
    body {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #222;
    }
    .header-meta {
        font-size: 8pt;
        color: #888;
        margin-bottom: 8pt;
    }
    .header-meta .filename { float: left; }
    .header-meta .date { float: right; }
    .header-meta::after { content: ""; display: table; clear: both; }
    h1 {
        font-size: 22pt;
        margin: 16pt 0 8pt 0;
        color: #111;
    }
    h2 {
        font-size: 16pt;
        margin: 20pt 0 8pt 0;
        color: #222;
        border-bottom: 1px solid #ccc;
        padding-bottom: 4pt;
    }
    h3 {
        font-size: 12pt;
        margin: 12pt 0 6pt 0;
        color: #333;
    }
    h4 {
        font-size: 11pt;
        margin: 10pt 0 5pt 0;
        color: #333;
    }
    h5 {
        font-size: 10pt;
        margin: 8pt 0 4pt 0;
        color: #444;
    }
    .subtitle {
        font-size: 10pt;
        margin-bottom: 8pt;
    }
    hr {
        border: none;
        border-top: 1px solid #ccc;
        margin: 16pt 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 8pt 0 12pt 0;
        font-size: 9pt;
    }
    th {
        background-color: #f5f5f5;
        border-bottom: 2px solid #ddd;
        text-align: left;
        padding: 6pt 8pt;
        font-weight: bold;
    }
    td {
        border-bottom: 1px solid #eee;
        padding: 5pt 8pt;
        vertical-align: top;
    }
    tr:nth-child(even) td {
        background-color: #fafafa;
    }
    ol, ul {
        margin: 4pt 0 8pt 0;
        padding-left: 20pt;
    }
    li {
        margin-bottom: 3pt;
    }
    p {
        margin: 4pt 0 8pt 0;
    }
    .meta {
        font-size: 9pt;
        color: #666;
    }
    .action-box {
        background-color: #f0f0f8;
        border-left: 3px solid #5b4fc7;
        padding: 8pt 12pt;
        margin: 8pt 0 12pt 0;
        font-size: 10pt;
    }
    .footer {
        margin-top: 24pt;
        font-size: 9pt;
        color: #888;
    }
    """
