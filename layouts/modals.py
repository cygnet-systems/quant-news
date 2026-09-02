"""Modal dialogs mounted at the app root.

These sit outside the routed page content so a modal opened from one section
survives navigation, and so their callbacks always have their Inputs mounted.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html



def create_data_modal() -> dbc.Modal:
    """Create the raw data view modal.

    Returns:
        Modal component for viewing data.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Raw Data"),
                close_button=True,
            ),
            dbc.ModalBody(
                [
                    html.Div(id="data-table-container"),
                ],
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Export Parquet",
                        id="modal-export-btn",
                        color="primary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Close",
                        id="modal-close-btn",
                        color="secondary",
                    ),
                ],
            ),
        ],
        id="data-modal",
        size="xl",
        is_open=False,
    )


def _report_param_selects(prefix: str, values: dict | None = None) -> dict:
    """Parameter controls for the Run dialog (prefix "run").

    The news window and article cap govern EVERY consumer of the run's news
: the sentiment and research models as well as the report, which is
    why they render in their own always-visible section, not under Report.
    """
    from config import MODEL
    v = values or {}

    return {
        "lookback": dbc.Select(
            id=f"{prefix}-lookback",
            options=[
                {"label": "3 days", "value": "3"},
                {"label": "7 days", "value": "7"},
                {"label": "14 days", "value": "14"},
                {"label": "1 month (30 days)", "value": "30"},
                {"label": "2 months (60 days)", "value": "60"},
                {"label": "3 months (90 days)", "value": "90"},
                {"label": "6 months (180 days)", "value": "180"},
                {"label": "1 year (365 days)", "value": "365"},
                {"label": "Overnight (close → open)", "value": "overnight"},
            ],
            value=str(v.get("lookback") or MODEL.NEWS_LOOKBACK_DAYS),
            size="sm",
        ),
        # Per-symbol cap on the window: keep the newest N, 0 = all. Applied
        # at fetch time and reported in the trace when it bites; the prompts
        # then sample ACROSS what is kept rather than reading the newest few.
        "max_articles": dbc.Input(
            id=f"{prefix}-max-articles",
            type="number",
            min=0,
            step=1,
            value=(v["max_articles"] if v.get("max_articles") is not None
                   else MODEL.NEWS_MAX_ARTICLES),
            size="sm",
        ),
        "model": dbc.Select(
            id=f"{prefix}-model",
            options=[
                {"label": "GPT-5.6 Luna (default)", "value": "gpt-5.6-luna"},
                {"label": "Claude Sonnet 5", "value": "claude-sonnet-5"},
                {"label": "Claude Sonnet 4.6", "value": "claude-sonnet-4-6"},
                {"label": "Claude Haiku 4.5 (fast)", "value": "claude-haiku-4-5"},
            ],
            value=v.get("report_model") or "gpt-5.6-luna",
            size="sm",
        ),
        "type": dbc.Select(
            id=f"{prefix}-type",
            options=[
                {"label": "Deep (company thesis)", "value": "thesis"},
                {"label": "Standard (faster)", "value": "standard"},
            ],
            value=v.get("depth") or "thesis",
            size="sm",
        ),
        "recs": dbc.Select(
            id=f"{prefix}-recs",
            options=[
                {"label": "From text analysis + predictions", "value": "auto"},
                {"label": "From predictions only (no text)", "value": "signals"},
                {"label": "Off", "value": "off"},
            ],
            value=v.get("recs") or "auto",
            size="sm",
        ),
        "recs-model": dbc.Select(
            id=f"{prefix}-recs-model",
            options=[
                {"label": "Claude Sonnet 5 (default)", "value": "claude-sonnet-5"},
                {"label": "GPT-5.6 Luna (reasoning)", "value": "gpt-5.6-luna"},
                {"label": "Claude Sonnet 4.6", "value": "claude-sonnet-4-6"},
            ],
            value=v.get("recs_model") or "claude-sonnet-5",
            size="sm",
        ),
    }


# =============================================================================
# UNIFIED RUN MODAL
# =============================================================================

RUN_MODELS = [
    ("kronos_mini", "Kronos", "90 bars OHLCV (min 30)"),
    ("xgboost_shap", "XGBoost SHAP", "1Y OHLCV + SPY + news (SMA-200)"),
    ("lightgbm", "LightGBM", "1Y OHLCV + SPY + news (SMA-200)"),
    ("deberta_sentiment", "DeBERTa Sentiment", "News articles (relevance >= 0.7)"),
    ("trading_agents", "TradingAgents",
     "Full research: technicals + fundamentals + news (~60s)"),
]

RUN_SCOPES = [
    ("models", "Models only",
     "Numerical predictions. No LLM report, no recommendations."),
    ("report", "Report only",
     "News and research analysis. No model predictions."),
    ("full", "Full pipeline",
     "Report, then all models, then the synthesis recommendation."),
]

# Wall-clock components for the pre-flight estimate, in seconds. These are the
# numbers already recorded elsewhere in the codebase rather than guesses: a
# research call is ~35s (app.py's fan-out comment) and RUN_MODELS advertises
# ~60s for TradingAgents end to end; the synthesis is one call over the whole
# batch and the run dialog already warns it "may take a minute". Everything
# here is presented to the user as approximate.
RESEARCH_SECONDS = 35
RESEARCH_CONCURRENCY = 8      # app.py caps the async fan-out at 8 workers
SYNTHESIS_SECONDS = 15        # one call for the batch, not per symbol
NEWS_FETCH_SECONDS = 3        # per symbol, Alpha Vantage is rate-limited
ML_MODEL_SECONDS = 4          # per symbol per numerical model

_LLM_MODELS = {"trading_agents"}
_NEWS_MODELS = {"deberta_sentiment", "xgboost_shap", "lightgbm"}


def _provider_for(model_id: str) -> str:
    return "openai" if (model_id or "").startswith("gpt-") else "anthropic"


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"~{int(round(seconds / 5.0)) * 5}s"
    minutes = seconds / 60.0
    return f"~{minutes:.0f} min" if minutes >= 2 else "~1-2 min"


def estimate_run_seconds(scope: str, n_symbols: int, models: list,
                         recs_on: bool) -> float:
    """Rough wall-clock for the selected scope. Deliberately coarse."""
    n = max(1, int(n_symbols or 0))
    models = list(models or [])
    total = 0.0
    does_report = scope in ("report", "full")
    does_models = scope in ("models", "full")

    if does_report or (does_models and any(m in _NEWS_MODELS for m in models)):
        total += n * NEWS_FETCH_SECONDS
    if does_report:
        # Research fans out concurrently, capped at RESEARCH_CONCURRENCY.
        batches = -(-n // RESEARCH_CONCURRENCY)
        total += batches * RESEARCH_SECONDS
    if does_models:
        ml = [m for m in models if m not in _LLM_MODELS]
        total += n * len(ml) * ML_MODEL_SECONDS
        if "trading_agents" in models and not does_report:
            # In the models-only path the research call runs inside the
            # prediction subprocess, one symbol at a time.
            total += n * RESEARCH_SECONDS
    if recs_on and scope == "full":
        total += SYNTHESIS_SECONDS
    return total


def run_preflight_children(scope: str, symbols: list, models: list,
                           report_model: str, recs_basis: str,
                           recs_model: str) -> list:
    """Warnings about missing keys plus a duration estimate for this run.

    A run with no usable API key used to look identical to a healthy one until
    it failed several stages in, and nothing anywhere said how long to wait.
    """
    import os

    from config import API

    def _key(name: str) -> bool:
        return bool(getattr(API, name, "") or os.getenv(name, ""))

    models = list(models or [])
    n = len(symbols or [])
    does_report = scope in ("report", "full")
    does_models = scope in ("models", "full")
    recs_on = scope == "full" and (recs_basis or "auto") != "off"

    # Which LLM providers this run will actually call, and for what.
    needs: dict[str, list[str]] = {}
    if does_report:
        needs.setdefault(_provider_for(report_model), []).append(
            "the research report")
    if does_models and "trading_agents" in models:
        needs.setdefault(_provider_for(report_model), []).append(
            "the TradingAgents model")
    if recs_on:
        needs.setdefault(_provider_for(recs_model), []).append(
            "the recommendation synthesis")

    warnings = []
    env_for = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
    for provider, uses in needs.items():
        env = env_for[provider]
        if not _key(env):
            warnings.append(
                f"{env} is not set. {', '.join(sorted(set(uses)))} will fail.")

    news_needed = does_report or (does_models
                                  and any(m in _NEWS_MODELS for m in models))
    if news_needed and not _key("ALPHA_VANTAGE_API_KEY"):
        display = {mid: label for mid, label, _ in RUN_MODELS}
        which = ["the report's news window"] if does_report else []
        which += [display.get(m, m) for m in models if m in _NEWS_MODELS]
        warnings.append(
            "ALPHA_VANTAGE_API_KEY is not set. No articles will be fetched, so "
            + ", ".join(which) + " will run on empty news.")

    if n == 0:
        # estimate_run_seconds floors at one symbol so a fixed cost is never
        # divided away; printing that as "~70s for 0 symbols" advertised work
        # for a run that cannot start.
        estimate = html.Div(
            [
                html.I(className="bi bi-clock me-1"),
                html.Span("No symbols selected. Nothing will run. Add at "
                          "least one above for a duration estimate."),
            ],
            className="run-preflight-estimate",
        )
    else:
        seconds = estimate_run_seconds(scope, n, models, recs_on)
        estimate = html.Div(
            [
                html.I(className="bi bi-clock me-1"),
                html.Span(f"Estimated {_fmt_duration(seconds)} for {n} symbol"
                          f"{'' if n == 1 else 's'}: approximate, and slower "
                          f"when a provider is rate-limiting."),
            ],
            className="run-preflight-estimate",
        )
    if not warnings:
        return [estimate]
    return [
        html.Div(
            [html.I(className="bi bi-exclamation-triangle-fill me-2")]
            + [html.Div(w) for w in warnings],
            className="run-preflight-warn",
        ),
        estimate,
    ]

# (value, label, what-it-does hint). The hints are honest about what the
# platform's own evals found, the selector is only useful if it tells you
# when each method actually differs.
ENSEMBLE_METHODS = [
    ("confidence_weighted", "Confidence-weighted vote",
     "Each vote counts weight x the model's stated confidence. Evals here "
     "found member confidence carries little calibration signal, so this "
     "mostly tracks the majority vote."),
    ("majority", "Weighted majority vote",
     "One model, one vote, scaled by its weight. Confidence ignored: the "
     "transparent baseline."),
    ("prob_mean", "Mean up-probability",
     "Weighted mean of the members' up-probabilities, thresholded. The "
     "best-calibrated signal the members publish (drives the ensemble's "
     "confidence in every method)."),
    ("agreement", "Consensus gate",
     "BUY/SELL only when at least N members back one direction and none "
     "back the other; otherwise HOLD. Weights ignored. Trades far less, but "
     "replaying it over stored history did NOT show the surviving trades "
     "being better: at the default gate of 3 it traded 40 times and hit "
     "42.5%. Raise the gate only if you want fewer positions, not because "
     "it is known to be more accurate."),
]

# One scenario, run through every method, so the choice can be made by seeing
# the answers diverge rather than by reading four descriptions. These four
# members are deliberately split: two BUY, one SELL, one HOLD. The numbers are
# not illustrative-but-approximate, tests/test_ensemble_methods.py replays
# this exact scenario through EnsembleModel and fails if any documented figure
# drifts from what the combiner really produces.
ENSEMBLE_EXAMPLE_MEMBERS = [
    # (model, decision, confidence, up_probability, weight)
    ("Kronos",   "BUY",  0.55, 0.55, 1.0),
    ("XGBoost",  "BUY",  0.80, 0.68, 1.5),
    ("LightGBM", "SELL", 0.60, 0.42, 1.0),
    ("DeBERTa",  "HOLD", 0.30, 0.50, 0.5),
]

# Direction is all that changes between methods. Confidence and up-probability
# always come from the weighted mean member probability, which is why the
# example resolves to the same 56% either way. Worth seeing, because it means
# switching method changes WHAT you trade, never how sure the ensemble claims
# to be.
ENSEMBLE_EXAMPLE_PROB = 0.56

ENSEMBLE_METHOD_DETAIL = {
    "confidence_weighted": {
        "formula": "score = Σ (weight × confidence × dir) ÷ Σ (weight × confidence)",
        "uses": "weights + each model's confidence",
        "ignores": "up-probabilities",
        "steps": [
            "Kronos   1.0 × 0.55 × (+1) = +0.55",
            "XGBoost  1.5 × 0.80 × (+1) = +1.20",
            "LightGBM 1.0 × 0.60 × (−1) = −0.60",
            "DeBERTa  0.5 × 0.30 × ( 0) =  0.00   (HOLD adds weight, no direction)",
            "score = +1.15 ÷ 2.50 = +0.46",
        ],
        "verdict": "BUY",
        "why": "+0.46 clears the +0.15 buy threshold. XGBoost's high weight "
               "and high confidence outrun LightGBM's dissent.",
    },
    "majority": {
        "formula": "score = Σ (weight × dir) ÷ Σ weight",
        "uses": "weights only",
        "ignores": "confidence and up-probabilities",
        "steps": [
            "Kronos   1.0 × (+1) = +1.00",
            "XGBoost  1.5 × (+1) = +1.50",
            "LightGBM 1.0 × (−1) = −1.00",
            "DeBERTa  0.5 × ( 0) =  0.00",
            "score = +1.50 ÷ 4.00 = +0.375",
        ],
        "verdict": "BUY",
        "why": "Same call as confidence-weighting, but by a narrower margin: "
               "dropping confidence costs XGBoost its extra pull.",
    },
    "prob_mean": {
        "formula": "p = Σ (weight × p_up) ÷ Σ weight,   score = (p − 0.5) × 2",
        "uses": "weights + each model's published up-probability",
        "ignores": "the BUY/SELL labels themselves",
        "steps": [
            "Kronos   1.0 × 0.55 = 0.550",
            "XGBoost  1.5 × 0.68 = 1.020",
            "LightGBM 1.0 × 0.42 = 0.420",
            "DeBERTa  0.5 × 0.50 = 0.250",
            "p = 2.24 ÷ 4.00 = 0.56  →  score = (0.56 − 0.5) × 2 = +0.12",
        ],
        "verdict": "HOLD",
        "why": "+0.12 falls short of +0.15, so the same inputs that produced "
               "a BUY above produce no trade. The labels said buy; the "
               "probabilities behind them were only mildly bullish.",
    },
    "agreement": {
        "formula": "BUY if buy_votes ≥ N and sell_votes = 0   "
                   "(SELL mirrored); else HOLD",
        "uses": "the vote count only",
        "ignores": "weights, confidence and up-probabilities",
        "steps": [
            "BUY votes  = 2   (Kronos, XGBoost)",
            "SELL votes = 1   (LightGBM)",
            "HOLD votes = 1   (DeBERTa, counts toward neither side)",
            "Needs ≥ 3 on one side AND 0 on the other → gate stays shut",
        ],
        "verdict": "HOLD",
        "why": "One dissenter is enough to veto, regardless of its weight. "
               "This trades far less often by design. Run "
               "scripts/replay_ensemble_methods.py to see how that has "
               "actually scored on your own history before relying on it.",
    },
}


def ensemble_method_detail(method: str, min_agree: int = 3) -> html.Div:
    """Formula, worked example and outcome for one combination method.

    The selector used to carry a sentence each, which said what a method was
    called but not what it would do to your run. Showing one scenario resolved
    four ways makes the choice concrete: the same four members produce BUY
    under both vote methods and HOLD under the other two.
    """
    d = ENSEMBLE_METHOD_DETAIL.get(method)
    if not d:
        return html.Div()

    verdict_cls = {"BUY": "positive", "SELL": "negative"}.get(d["verdict"], "neutral")

    members = html.Table(
        [
            html.Thead(html.Tr([
                html.Th("Model"), html.Th("Call"), html.Th("Conf"),
                html.Th("P(up)"), html.Th("Weight"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(name),
                    html.Td(dec, className=f"ens-ex-call ens-ex-{dec.lower()}"),
                    html.Td(f"{conf:.0%}", className="num"),
                    html.Td(f"{p:.2f}", className="num"),
                    html.Td(f"{w:g}", className="num"),
                ])
                for name, dec, conf, p, w in ENSEMBLE_EXAMPLE_MEMBERS
            ]),
        ],
        className="ens-ex-table",
    )

    steps = d["steps"]
    if method == "agreement":
        # The gate's threshold is user-set, so echo the live value rather than
        # the default baked into the example text.
        steps = [st.replace("≥ 3", f"≥ {min_agree}") for st in steps]

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Formula", className="ens-ex-label"),
                    html.Code(d["formula"], className="ens-ex-formula"),
                ],
                className="ens-ex-row",
            ),
            html.Div(
                [
                    html.Span("Uses", className="ens-ex-label"),
                    html.Span(d["uses"], className="ens-ex-uses"),
                    html.Span("Ignores", className="ens-ex-label ms-3"),
                    html.Span(d["ignores"], className="ens-ex-ignores"),
                ],
                className="ens-ex-row",
            ),
            html.Details(
                [
                    html.Summary("Worked example", className="ens-ex-summary"),
                    html.Div(
                        [
                            members,
                            html.Pre("\n".join(steps), className="ens-ex-steps"),
                            html.Div(
                                [
                                    html.Span("Ensemble says", className="ens-ex-label"),
                                    html.Span(d["verdict"],
                                              className=f"ens-ex-verdict {verdict_cls}"),
                                    html.Span(f"at {ENSEMBLE_EXAMPLE_PROB:.0%} confidence",
                                              className="ens-ex-conf"),
                                ],
                                className="ens-ex-verdict-row",
                            ),
                            html.Div(d["why"], className="ens-ex-why"),
                            html.Div(
                                "Confidence is the weighted mean up-probability "
                                "in every method, so switching method changes "
                                "which trades you take, not how sure the "
                                "ensemble claims to be.",
                                className="ens-ex-note",
                            ),
                        ],
                        className="ens-ex-body",
                    ),
                ],
                className="ens-ex",
            ),
        ],
    )


def create_model_info_modal() -> dbc.Modal:
    """Explainer modal opened by the ⓘ icons in the Run dialog's model list.

    Body is filled by the open callback from layouts.model_info: for
    TradingAgents it includes the live prompt preview reflecting the
    Evidence-blocks checklist at the moment it was opened.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id="model-info-title")),
            dbc.ModalBody(id="model-info-body", className="model-info-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="model-info-close-btn",
                           color="secondary", size="sm"),
            ),
        ],
        id="model-info-modal",
        size="lg",
        scrollable=True,
        is_open=False,
    )


def run_model_controls(prefix: str, with_info: bool = False,
                       values: dict | None = None):
    """The Models + Ensemble controls, for any dialog that configures a run.

    ``prefix`` namespaces every id ("run" for the Run dialog, "sj" for the
    Schedule modal); the controls are otherwise identical, so a scheduled
    job is configured with exactly the Run dialog's vocabulary. ``values``
    seeds the controls (models, run_ensemble, ensemble{method, min_agree,
    enabled_models, weights}); None = the config defaults.
    """
    from config import MODEL
    v = values or {}
    ens = v.get("ensemble") or {}
    selected_models = set(v.get("models") or [mid for mid, _, _ in RUN_MODELS])

    model_checks = [
        html.Div(
            [
                dbc.Checkbox(id={"type": f"{prefix}-model-check", "model": mid},
                             value=mid in selected_models, className="me-2"),
                html.Span(display, className="run-model-name"),
                html.Span(requirement, className="run-model-req"),
                # Explainer for researchers (Run dialog only; the info
                # modal's callback is bound to the run ids).
                html.I(
                    className="bi bi-info-circle run-model-info-btn ms-auto",
                    id={"type": "run-model-info", "model": mid},
                    title=f"What is {display}?",
                    style={"cursor": "pointer",
                           "color": "var(--text-secondary)"},
                ) if with_info else None,
            ],
            className="d-flex align-items-center mb-2",
        )
        for mid, display, requirement in RUN_MODELS
    ]

    default_enabled = set(ens.get("enabled_models") or MODEL.ENSEMBLE_DEFAULT_ENABLED)
    default_weights = dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)
    default_weights.update(ens.get("weights") or {})
    member_rows = [
        html.Div(
            [
                dbc.Checkbox(id={"type": f"{prefix}-ens-member", "model": mid},
                             value=mid in default_enabled, className="me-2"),
                html.Span(display, className="run-model-name"),
                dcc.Input(
                    id={"type": f"{prefix}-ens-weight", "model": mid},
                    type="number", min=0.1, max=2.0, step=0.1,
                    value=default_weights.get(mid, 1.0),
                    className="ensemble-weight-input",
                    style={"width": "70px"},
                ),
            ],
            className="d-flex align-items-center mb-1",
        )
        for mid, display, _ in RUN_MODELS
    ]

    ensemble_section = html.Div(
        [
            html.Div(
                [
                    dbc.Checkbox(id=f"{prefix}-ensemble-check",
                                 value=v.get("run_ensemble", True),
                                 className="me-2"),
                    html.Span("Ensemble", className="run-model-name"),
                    html.Span("combines the member models configured below",
                              className="run-model-req"),
                    html.I(
                        className="bi bi-info-circle run-model-info-btn ms-auto",
                        id={"type": "run-model-info", "model": "ensemble"},
                        title="What is the Ensemble?",
                        style={"cursor": "pointer",
                               "color": "var(--text-secondary)"},
                    ) if with_info else None,
                ],
                className="d-flex align-items-center mb-2",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Combination method",
                                               className="input-label"),
                                    dbc.Select(
                                        id=f"{prefix}-ensemble-method",
                                        options=[{"label": lbl, "value": val}
                                                 for val, lbl, _ in ENSEMBLE_METHODS],
                                        value=ens.get("method") or MODEL.ENSEMBLE_DEFAULT_METHOD,
                                        size="sm",
                                    ),
                                ],
                                className="run-field",
                            ),
                            html.Div(
                                [
                                    html.Label("Min. agreeing models",
                                               className="input-label"),
                                    dcc.Input(
                                        id=f"{prefix}-ensemble-min-agree",
                                        type="number", min=2, max=len(RUN_MODELS),
                                        step=1,
                                        value=ens.get("min_agree") or MODEL.ENSEMBLE_MIN_AGREE,
                                        className="ensemble-weight-input",
                                        style={"width": "70px"},
                                    ),
                                ],
                                id=f"{prefix}-ensemble-min-agree-wrap",
                                className="run-field",
                            ),
                        ],
                        className="run-field-grid",
                    ),
                    html.Div(id=f"{prefix}-ensemble-method-hint",
                             className="run-field-hint mb-2"),
                    html.Div(
                        [
                            html.Label("Members and weights",
                                       className="input-label"),
                            html.Div(member_rows),
                        ],
                        className="run-field",
                    ),
                    html.Div(id=f"{prefix}-ensemble-summary"),
                ],
                id=f"{prefix}-ensemble-body",
                className="ms-4",
            ),
        ],
    )
    return model_checks, ensemble_section


EVIDENCE_OPTIONS = [
    {"label": "Options positioning (put/call)", "value": "options"},
    {"label": "Quality screen (Bad Apples + news red flags)", "value": "quality"},
    {"label": "Situation & investigation (web research, live runs)",
     "value": "investigation"},
    {"label": "Political & institutional flows", "value": "political"},
]

TOOL_OPTIONS = [
    {"label": "Web research. The situation investigation searches the open "
              "web with citations (on for next-day runs, off for backtests)",
     "value": "web_research"},
]


def run_evidence_tools(prefix: str, values: dict | None = None):
    """The Evidence-blocks and Tools checklists for any run-configuring dialog."""
    from config import MODEL
    v = values or {}
    evidence = dbc.Checklist(
        id=f"{prefix}-evidence", options=EVIDENCE_OPTIONS,
        value=(list(v["evidence"]) if v.get("evidence") is not None
               else list(MODEL.DEFAULT_EVIDENCE)),
        inline=True, className="run-evidence-checklist",
    )
    tools = dbc.Checklist(
        id=f"{prefix}-tools", options=TOOL_OPTIONS,
        value=(list(v["tools"]) if v.get("tools") is not None else ["web_research"]),
        inline=True, className="run-evidence-checklist",
    )
    return evidence, tools


def run_settings_sections(prefix: str, values: dict | None = None,
                          with_info: bool = False) -> list:
    """News / Models / Report / Tools / Recommendations sections with
    ``prefix``-namespaced ids, seeded from ``values`` (a job's params) or the
    config defaults. The Schedule modal renders exactly these, so a scheduled
    job can be configured with everything the Run dialog offers."""
    v = values or {}
    sel = _report_param_selects(prefix, v)
    model_checks, ensemble_section = run_model_controls(prefix, with_info, v)
    evidence, tools = run_evidence_tools(prefix, v)

    def field(label, control, hint=None):
        return html.Div(
            [html.Label(label, className="input-label"), control,
             html.Div(hint, className="run-field-hint") if hint else None],
            className="run-field",
        )

    return [
        html.Div([
            html.Hr(), html.H6("News", className="mb-2"),
            html.Div([
                field("Window", sel["lookback"],
                      "Point-in-time: articles published in this many days up "
                      "to the data cutoff."),
                field("Article cap per symbol", sel["max_articles"],
                      "Keep the newest N of the window; 0 = all. The trace "
                      "reports when this drops articles."),
            ], className="run-field-grid"),
        ], id=f"{prefix}-news-section"),
        html.Div([
            html.Hr(), html.H6("Models", className="mb-2"),
            html.Div(model_checks), ensemble_section,
        ], id=f"{prefix}-models-section"),
        html.Div([
            html.Hr(), html.H6("Report", className="mb-2"),
            html.Div([field("Report model", sel["model"]),
                      field("Depth", sel["type"])], className="run-field-grid"),
            html.Div([html.Div("Evidence blocks", className="run-field-label"),
                      evidence], className="mt-2"),
        ], id=f"{prefix}-report-section"),
        html.Div([
            html.Hr(), html.H6("Tools", className="mb-2"), tools,
        ], id=f"{prefix}-tools-section"),
        html.Div([
            html.Hr(), html.H6("Recommendations", className="mb-2"),
            html.Div([field("Basis", sel["recs"]),
                      field("Synthesis model", sel["recs-model"])],
                     className="run-field-grid"),
        ], id=f"{prefix}-recs-section"),
    ]


def create_run_modal() -> dbc.Modal:
    """One dialog for every way of starting a run.

    Replaces three near-identical modals (Predict, AI Report, Full Analysis)
    whose parameter panels were duplicated: the report selects existed twice,
    once as ai-report-* and once as fa-*, and Full Analysis silently
    overrode the Predict modal's model checkboxes. Scope is now an explicit
    choice rather than something inferred from which button was pressed, and
    there is exactly one set of controls behind it.
    """
    from datetime import date

    # The target is a market session, so weekends and holidays are not
    # choosable. Greying them out states the rule in the calendar rather than
    # letting a run be configured against a date that cannot resolve.
    picker_min = "2020-01-01"
    try:
        from utils.trading_calendar import get_default_target_day, non_trading_days
        default_target = get_default_target_day()
        closed_days = [d.isoformat()
                       for d in non_trading_days(picker_min, default_target)]
    except Exception:
        # A calendar failure must not cost the dialog: an open picker with
        # every day selectable still works, and the runner resolves the
        # session itself.
        default_target = date.today()
        closed_days = []

    from config import MODEL

    sel = _report_param_selects("run")

    model_checks, ensemble_section = run_model_controls("run", with_info=True)

    def field(label, control, hint=None):
        return html.Div(
            [
                html.Label(label, className="input-label"),
                control,
                html.Div(hint, className="run-field-hint") if hint else None,
            ],
            className="run-field",
        )

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle([html.I(className="bi bi-play-circle me-2"),
                                "Run analysis"]),
                close_button=True,
            ),
            dbc.ModalBody([
                # Pre-flight: what this run needs that is not configured, and
                # roughly how long it will take. Both were previously only
                # discoverable by starting the run and watching it fail.
                html.Div(id="run-preflight", className="run-preflight"),

                # --- Symbols ---
                # The set a run acts on used to be an invisible rule (always
                # the watchlist). It is now the dialog's first question, preset
                # by whichever control opened it: the toolbar button starts
                # from the watchlist, a per-symbol "New report" starts from
                # that one symbol. Edits apply to this run only.
                html.Div(
                    [
                        html.Label("Symbols", className="input-label"),
                        dbc.RadioItems(
                            id="run-symbols-source",
                            options=[],  # filled with live counts on open
                            inline=True,
                            className="run-scope-radio",
                        ),
                        html.Div(id="run-symbols-chips",
                                 className="run-symbols-chips"),
                        dbc.Input(
                            id="run-symbol-add",
                            type="text",
                            placeholder="Add a symbol to this run… (Enter)",
                            size="sm",
                            className="run-symbol-add",
                            autoComplete="off",
                        ),
                        html.Div(
                            "Applies to this run only. The watchlist is "
                            "not changed.",
                            className="run-field-hint",
                        ),
                    ],
                    className="run-field",
                ),
                html.Hr(),

                html.Div(
                    [
                        html.Label("What to run", className="input-label"),
                        dbc.RadioItems(
                            id="run-scope",
                            options=[{"label": lbl, "value": val}
                                     for val, lbl, _ in RUN_SCOPES],
                            value="full",
                            inline=True,
                            className="run-scope-radio",
                        ),
                        html.Div(id="run-scope-hint", className="run-field-hint"),
                    ],
                    className="run-field",
                ),
                html.Hr(),

                field(
                    "Target session",
                    html.Div(
                        [
                            dcc.DatePickerSingle(
                                id="run-date-picker",
                                date=default_target.isoformat(),
                                max_date_allowed=default_target.isoformat(),
                                min_date_allowed=picker_min,
                                disabled_days=closed_days,
                                display_format="YYYY-MM-DD",
                                className="predict-date-picker",
                            ),
                            html.Span(id="run-date-mode-label", className="ms-2"),
                        ],
                        className="d-flex align-items-center",
                    ),
                    "The close being predicted. Data is cut off at the previous "
                    "trading day, so a Monday target sees nothing after Friday.",
                ),
                html.Div(id="run-data-summary"),

                # --- News (feeds models AND report, every scope) ---
                html.Div(
                    [
                        html.Hr(),
                        html.H6("News", className="mb-2"),
                        html.Div(
                            [
                                field("Window", sel["lookback"],
                                      "Point-in-time: articles published in "
                                      "this many days up to the data cutoff."),
                                field("Article cap per symbol",
                                      sel["max_articles"],
                                      "Keep the newest N of the window; "
                                      "0 = all. The trace reports when this "
                                      "drops articles."),
                            ],
                            className="run-field-grid",
                        ),
                        html.Div(id="run-article-preview",
                                 className="run-article-preview"),
                    ],
                    id="run-news-section",
                ),

                # --- Models ---
                html.Div(
                    [
                        html.Hr(),
                        html.H6("Models", className="mb-2"),
                        html.Div(model_checks),
                        ensemble_section,
                    ],
                    id="run-models-section",
                ),

                # --- Report ---
                html.Div(
                    [
                        html.Hr(),
                        html.H6("Report", className="mb-2"),
                        html.Div(
                            [
                                field("Report model", sel["model"]),
                                field("Depth", sel["type"]),
                            ],
                            className="run-field-grid",
                        ),
                        # Terminal-derived evidence blocks, selectable per
                        # run. Both default on; the research prompt, the
                        # synthesis evidence and the rendered report sections
                        # follow this choice.
                        html.Div(
                            [
                                html.Div("Evidence blocks",
                                         className="run-field-label"),
                                run_evidence_tools("run")[0],
                            ],
                            className="mt-2",
                        ),
                    ],
                    id="run-report-section",
                ),

                # --- Tools ---
                # Run tools are opt-in switches the backend never flips on
                # its own. Web research defaults ON here for a next-day run
                # and is switched off by the date callback for a backtest
                # date (the open web cannot be bounded to a past as-of).
                html.Div(
                    [
                        html.Hr(),
                        html.H6("Tools", className="mb-2"),
                        run_evidence_tools("run")[1],
                    ],
                    id="run-tools-section",
                ),

                # --- Recommendations ---
                html.Div(
                    [
                        html.Hr(),
                        html.H6("Recommendations", className="mb-2"),
                        html.Div(
                            [
                                field("Basis", sel["recs"]),
                                field("Synthesis model", sel["recs-model"]),
                            ],
                            className="run-field-grid",
                        ),
                    ],
                    id="run-recs-section",
                ),
            ]),
            dbc.ModalFooter([
                # Inline validation (e.g. an empty symbol set), me-auto keeps
                # it left of the buttons in the footer's flex row.
                html.Div(id="run-validation-msg",
                         className="run-validation-msg text-danger me-auto"),
                dbc.Button("Cancel", id="run-cancel-btn", color="secondary",
                           className="me-2"),
                dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run"],
                           id="run-confirm-btn", color="success"),
            ]),
        ],
        id="run-modal",
        is_open=False,
        size="lg",
        scrollable=True,
    )
