"""Model explainers for the Run Analysis dialog.

Written for equity researchers who don't know the models: what each one is,
what it reads, why it's on the panel, and what our own backtests say about
trusting it. The honesty is deliberate — measured limitations from this
platform's benchmarks are part of the explanation, not a footnote.
"""

from config import MODEL
from dash import dcc, html

# model_id -> (title, what it is, what it reads, why it's useful, caveats)
MODEL_EXPLAINERS = {
    "kronos_mini": {
        "title": "Kronos — price-path forecaster",
        "what": (
            "A small time-series foundation model (transformer) that reads "
            "the last ~90 daily OHLCV bars and Monte-Carlo samples possible "
            "next-day price paths. The BUY/SELL/HOLD call is the direction "
            "most sampled paths agree on."
        ),
        "reads": "Price history only — no news, no fundamentals.",
        "why": (
            "The one voice on the panel that is pure price action. When "
            "Kronos disagrees with the news-driven models, that split is "
            "itself information: the tape and the narrative are not aligned."
        ),
        "caveats": (
            "Our 200-day, 5-symbol backtest measured ~49% directional "
            "accuracy on active calls — statistically a coin flip. It "
            "carries a structural SELL tilt that flatters it in down or "
            "flat tapes. Its self-reported confidence is the worst "
            "calibrated of the panel. Treat it as a tape-reading opinion, "
            "never a standalone signal."
        ),
    },
    "xgboost_shap": {
        "title": "XGBoost SHAP — engineered-feature classifier",
        "what": (
            "Gradient-boosted decision trees retrained from scratch for "
            "each symbol on every run, using only data available up to the "
            "as-of date. It classifies the next session as up or down from "
            "18 engineered features (trend vs SMAs, RSI, ATR, volume, SPY "
            "and sector correlation, aggregated news sentiment)."
        ),
        "reads": "1 year of OHLCV + SPY/sector context + news-sentiment aggregates.",
        "why": (
            "Fully transparent attribution: SHAP values show exactly which "
            "features drove each call (e.g. 'RSI and SPY correlation "
            "pushed this to BUY'). Good for asking WHY a quant signal fired."
        ),
        "caveats": (
            "Behaves like a momentum-long strategy: it is right on "
            "uptrending names and wrong on downtrends. Measured 52% "
            "accuracy at scale — inside the noise band of 50%. Its edge, "
            "if any, is thinner than realistic transaction costs."
        ),
    },
    "lightgbm": {
        "title": "LightGBM — second opinion on the same features",
        "what": (
            "A different gradient-boosting algorithm (leaf-wise growth) "
            "trained on the identical 18 features as XGBoost. Same inputs, "
            "different learning bias."
        ),
        "reads": "Same as XGBoost SHAP.",
        "why": (
            "Agreement between XGBoost and LightGBM means the FEATURES "
            "carry the signal; disagreement means the call is an artifact "
            "of one algorithm's fitting quirks. It exists to separate "
            "those two cases."
        ),
        "caveats": (
            "Shares XGBoost's momentum-long bias and its coin-flip-at-scale "
            "track record — two agreeing weak learners are not one strong "
            "one."
        ),
    },
    "deberta_sentiment": {
        "title": "DeBERTa — news-sentiment reader",
        "what": (
            "A transformer NLP model that scores each article in the "
            "point-in-time news window (relevance-filtered) and votes "
            "BUY/SELL/HOLD from the aggregate sentiment balance."
        ),
        "reads": "News headlines and summaries only — no price data at all.",
        "why": (
            "Quantifies the news narrative without technical bias. Useful "
            "as a cross-check: bullish tape + bearish news flow is a "
            "divergence worth explaining before acting."
        ),
        "caveats": (
            "Blind to valuation and price. Abstains when the window is "
            "empty, and a throttled news source can silently starve it — "
            "check the news-status note on the run. Sentiment is measured "
            "coin-flip as a 1-day direction signal here, like the others."
        ),
    },
    "trading_agents": {
        "title": "TradingAgents — LLM research analyst",
        "what": (
            "A large language model given a strict research procedure and "
            "ONLY validated, lookahead-safe data blocks (computed by our "
            "code, not asserted by the model). It writes the verdict-first "
            "research report you see on each symbol: regime, technicals, "
            "fundamentals, news, bull vs bear, risk, and a trade plan with "
            "explicit reassess triggers."
        ),
        "reads": (
            "Validated metric blocks: price action, technicals, "
            "fundamentals (as-of filtered), point-in-time news, event "
            "calendar, peer relative strength, SPY regime — plus the "
            "optional evidence blocks selected below (options positioning, "
            "quality screen). The exact prompt is shown under 'View "
            "research prompt'."
        ),
        "why": (
            "The only panel member that can weigh qualitative evidence "
            "(a regulatory probe, a guidance cut) against the numbers and "
            "explain its reasoning in prose a researcher can audit line "
            "by line. Every figure it cites must come from a supplied "
            "block — fabrication is checkable."
        ),
        "caveats": (
            "Its stated confidence carries no calibration signal (we "
            "measured it), so scoring uses its realized hit-rate instead. "
            "Report-layer backtests showed ~60% on small active-call "
            "samples with a SELL tilt that fit the regime — not proven "
            "alpha. Read it for the reasoning, not the verdict alone."
        ),
    },
    "ensemble": {
        "title": "Ensemble — weighted vote",
        "what": (
            "A configurable weighted majority vote across the enabled "
            "member models, with an optional minimum-agreement gate."
        ),
        "reads": "The other models' decisions and confidences. No market data.",
        "why": (
            "One line that summarizes the panel. Useful as a tie-break "
            "summary when members split."
        ),
        "caveats": (
            "Measured on our production data: 52.3% hit rate, "
            "anti-calibrated, and days when all members AGREED were "
            "slightly WORSE than average — the members are correlated "
            "momentum voters, so consensus mostly means 'the trend is "
            "obvious', not 'the call is safe'."
        ),
    },
}

_EVIDENCE_BLOCK_ROWS = [
    ("metrics", "Validated metrics", True,
     "ATR, support/resistance, reward:risk, volume vs average, drawdown — "
     "computed from OHLCV truncated to the as-of date."),
    ("events", "Event calendar", True,
     "Upcoming earnings/ex-dividend dates; gates the hold window."),
    ("peers", "Peer relative strength", True,
     "1-month and 1-week returns vs named peers and the sector ETF — "
     "separates company-specific moves from sector-wide repricing."),
    ("spy", "SPY regime", True,
     "Bull/bear/mixed by 50- and 200-day SMA rule, computed, not eyeballed."),
    ("options", "Options positioning (put/call)", None,
     "Put/call volume and open-interest ratios from the point-in-time "
     "chain. Presented as positioning context — hedging vs speculation — "
     "never as a standalone timing signal (our 1-day test showed only a "
     "weak, non-significant directional lean)."),
    ("quality", "Quality screen — Bad Apples + news red flags", None,
     "20 pass/fail checks (performance vs benchmark, fundamentals, "
     "valuation, short interest, analyst-revision momentum, insider "
     "selling) plus a news scan for leadership departures, layoffs, "
     "investigations, guidance cuts, short-seller reports and dilution. "
     "A high fail count argues for smaller size and skepticism toward "
     "bullish theses; it does not predict next-day direction."),
    ("investigation", "Situation & investigation (web research)", None,
     "A tool-using research call first classifies the situation (pending "
     "acquisition, legal/regulatory overhang, earnings event, leadership "
     "change, distress, momentum only) and then researches it on the open "
     "web with citations: deal terms and spread, regulators and milestones, "
     "the key figures and their track records and affiliations. Live runs "
     "only — on a backtest the classifier runs without web access, because "
     "web results cannot be bounded to a past as-of date."),
    ("political", "Political & institutional flows (Alpha Vantage)", None,
     "Congressional trades disclosed under the STOCK Act (filed on or "
     "before the as-of date, with party, chamber and amount band) and 13F "
     "holder flows (holders adding vs cutting, largest movers). Disclosure "
     "lags weeks; positioning context, not a timing signal."),
]


def build_model_info_body(model_id: str, evidence: list | None = None,
                          include_thesis: bool = True,
                          tools: list | None = None) -> list:
    """Explainer content for one model; TradingAgents also gets the live
    block list (reflecting the Evidence checkboxes) and the actual prompt."""
    info = MODEL_EXPLAINERS.get(model_id)
    if not info:
        return [html.P("No description available.")]

    children = [
        html.P(info["what"], className="model-info-what"),
        html.P([html.Strong("Reads: "), info["reads"]]),
        html.P([html.Strong("Why it's here: "), info["why"]]),
        html.P([html.Strong("Trust it how far? "), info["caveats"]],
               className="model-info-caveats"),
    ]

    if model_id != "trading_agents":
        return children

    evidence = set(evidence) if evidence is not None else set(MODEL.DEFAULT_EVIDENCE)
    rows = []
    for key, label, always, desc in _EVIDENCE_BLOCK_ROWS:
        included = always if always is not None else key in evidence
        icon = "✓" if included else "✗"
        color = "var(--positive)" if included else "var(--text-muted)"
        rows.append(html.Li(
            [
                html.Span(f"{icon} ", style={"color": color,
                                             "fontWeight": "700"}),
                html.Strong(label + ": "),
                html.Span(desc + ("" if included else
                                  " — excluded by the Evidence checkboxes.")),
            ],
            style={} if included else {"opacity": 0.6},
        ))
    web_on = "web_research" in set(tools or [])
    rows.append(html.Li(
        [
            html.Span(("✓ " if web_on else "✗ "),
                      style={"color": ("var(--positive)" if web_on
                                       else "var(--text-muted)"),
                             "fontWeight": "700"}),
            html.Strong("Tool — web research: "),
            html.Span("the investigation searches the open web with "
                      "citations (on by default for next-day runs)"
                      if web_on else
                      "off — the investigation classifies from the supplied "
                      "evidence only (the default for backtest dates, where "
                      "the open web would leak the future)."),
        ],
        style={} if web_on else {"opacity": 0.6},
    ))
    children.append(html.Hr())
    children.append(html.H6("Context blocks in THIS run"))
    children.append(html.P(
        "Every block is computed by our code from data available on or "
        "before the as-of date. The model may only cite figures that appear "
        "in these blocks.", className="run-model-req"))
    children.append(html.Ul(rows, className="model-info-blocks"))

    try:
        from models.single_agent import SINGLE_AGENT_PROMPT
        prompt_text = SINGLE_AGENT_PROMPT
        if include_thesis:
            from models.single_agent import EPILOGUE_INSTRUCTIONS
            prompt_text = prompt_text + "\n\n" + EPILOGUE_INSTRUCTIONS
        children.append(html.Hr())
        children.append(html.H6("The research prompt (verbatim)"))
        children.append(html.P(
            "This is the actual instruction template sent to the model. "
            "The {placeholders} are filled with the blocks above at run "
            "time; the selected evidence blocks arrive under 'PRECOMPUTED "
            "METRICS & EVENTS'.", className="run-model-req"))
        children.append(html.Pre(
            prompt_text,
            className="model-info-prompt",
            style={"maxHeight": "320px", "overflowY": "auto",
                   "fontSize": "0.72rem", "lineHeight": "1.45",
                   "whiteSpace": "pre-wrap",
                   "background": "var(--surface-2, rgba(128,128,128,0.08))",
                   "padding": "10px", "borderRadius": "6px"},
        ))
    except Exception:
        children.append(html.P("Prompt template unavailable.",
                               className="run-model-req"))
    return children
