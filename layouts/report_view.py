"""The research report as a page: verdict term sheet, numbered sections with
their takeaways, Read lines pulled out, provenance footer.

One builder for every place a report is read (reader modal, Home reading
pane, Analyze tab). It renders from ``services.report_parse.ParsedReport``
so the on-screen page and the PDF share a structure; anything the parser
could not structure (older free-form reports) falls back to plain markdown.
"""

from dash import dcc, html

from services.report_parse import ParsedReport, parse_research_report, tone_for


def _weight_text(weight) -> str:
    if weight in (None, 0.5):
        return "track-record weight unrated (fewer than 20 scored calls)"
    try:
        return f"track-record weight {float(weight):.0%} on resolved calls"
    except (TypeError, ValueError):
        return "track-record weight n/a"


def build_report_view(report_text: str, *, symbol: str = "", decision: str = "",
                      weight=None, trade_date: str = "", model_name: str = "",
                      generated: str = "", compact: bool = False) -> html.Div:
    parsed: ParsedReport = parse_research_report(report_text)
    if not parsed.structured:
        return html.Div(
            dcc.Markdown(parsed.legacy_body or report_text, className="ta-report-body"),
            className="rr rr-legacy" + (" rr-compact" if compact else ""),
        )

    call = parsed.call or decision or "?"
    tone = tone_for(call)
    conv = f"{parsed.conviction:.2f}" if parsed.conviction is not None else "n/a"

    masthead = html.Div(
        [
            html.Span(f"{symbol} · research note" if symbol else "Research note"),
            html.Span(" · ".join(x for x in (
                f"as of {trade_date}" if trade_date else "",
                model_name or "",
                f"generated {generated}" if generated else "") if x)),
        ],
        className="rr-masthead",
    )

    fields = [
        ("Since last report", parsed.since),
        ("Reassess to buy", parsed.reassess),
        ("Move to sell", parsed.move),
    ]
    verdict = html.Section(
        [
            html.Div([html.Span("Call", className="rr-label"),
                      html.B(call, className=f"rr-call rr-{tone}")],
                     className="rr-call-cell"),
            html.Div([
                html.B(conv, className="rr-conv-num"),
                html.Span(
                    parsed.conviction_note or
                    "conviction, the report's own probability that the direction is right",
                    className="rr-conv-note"),
                html.Span(_weight_text(weight), className="rr-conv-note"),
                html.Span(parsed.measured, className="rr-measured") if parsed.measured else "",
            ], className="rr-conv-cell"),
            html.Dl(
                [el for label, value in fields if value
                 for el in (html.Dt(label), html.Dd(dcc.Markdown(value, className="rr-inline")))],
                className="rr-fields",
            ),
            html.Ul([html.Li(dcc.Markdown(w, className="rr-inline")) for w in parsed.why],
                    className="rr-why") if parsed.why else "",
        ],
        className=f"rr-verdict rr-verdict-{tone}",
    )

    toc = html.Ul(
        [html.Li(html.A([html.Span(s.num, className="rr-toc-num"), s.name],
                        href=f"#rr-s{s.num}" if s.num else None))
         for s in parsed.sections if s.name],
        className="rr-toc",
    ) if not compact else ""

    sections = []
    for s in parsed.sections:
        sections.append(html.Section(
            [
                html.H2(
                    [html.Span(s.num, className="rr-num") if s.num else "",
                     html.Span(s.name, className="rr-name"),
                     html.Span(s.takeaway, className="rr-take") if s.takeaway else ""],
                    id=f"rr-s{s.num}" if s.num else None,
                ),
                dcc.Markdown(s.body_md, className="rr-body") if s.body_md else "",
                html.P([html.Span("Read", className="rr-read-label"), s.read],
                       className="rr-read") if s.read else "",
            ],
            className="rr-section",
        ))

    footer = html.Div(dcc.Markdown(parsed.footer, className="rr-inline"),
                      className="rr-footer") if parsed.footer else ""

    return html.Div([masthead, verdict, toc, html.Div(sections, className="rr-sections"), footer],
                    className="rr" + (" rr-compact" if compact else ""))
