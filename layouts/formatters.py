"""Display helpers shared across pages: model naming and report rendering."""

MODEL_DISPLAY = {
    "kronos_mini": "Kronos",
    "xgboost_shap": "XGBoost SHAP",
    "lightgbm": "LightGBM",
    "deberta_sentiment": "DeBERTa Sentiment",
    "trading_agents": "TradingAgents",
    "ensemble": "Ensemble",
    "recommendation_synthesis": "Recommendations (LLM)",
}


# Two unrelated numbers used to sit unlabeled inches apart on the same report:
# a modal titled "MRAM — SELL (50%)" beside body text reading "CONFIDENCE:
# 0.68". The 50% is the model's measured directional hit rate (0.5 = it has
# not earned one yet — NOT the model saying it is 50% sure); the 0.68 is the
# report's own stated conviction. Every surface that shows either one goes
# through these so the wording can never drift apart again.

def weight_label(confidence) -> str:
    """Reader-facing label for the stored reliability weight."""
    if confidence is None:
        return "track-record weight — unrated"
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "track-record weight — unrated"
    if value == 0.5:
        return "track-record weight — unrated"
    return f"track-record weight {value:.0%}"


def conviction_label(stated) -> str:
    """Reader-facing label for the report's own stated conviction."""
    if stated is None:
        return "conviction — not stated"
    try:
        return f"conviction {float(stated):.2f}"
    except (TypeError, ValueError):
        return "conviction — not stated"


def confidence_tooltip() -> str:
    """One sentence explaining the pair, for `title=` attributes."""
    return ("Track-record weight is this model's measured hit rate on resolved "
            "calls (unrated until it has enough of them); conviction is what "
            "the report itself claimed, which has never been scored.")


def json_report_to_markdown(data: dict) -> str:
    """Convert an AI report JSON dict into a readable Markdown document."""
    lines = []
    generated = data.get("generated_at", "")
    if generated:
        lines.append(f"*Generated: {generated}*\n")

    overall = data.get("overall", {})
    if overall:
        lines.append("# Portfolio Summary\n")
        rec = overall.get("recommendation", "—")
        conf = overall.get("confidence", 0)
        lines.append(f"**Recommendation:** {rec}  ")
        lines.append(f"**Confidence:** {int(conf * 100) if isinstance(conf, float) and conf <= 1 else conf}%  ")
        lines.append(f"**Sentiment:** {overall.get('market_sentiment', '—')}\n")
        if overall.get("sentiment_explanation"):
            lines.append(f"{overall['sentiment_explanation']}\n")
        if overall.get("key_developments"):
            lines.append(f"**Key Developments:** {overall['key_developments']}\n")

    by_symbol = data.get("by_symbol", {})
    if by_symbol:
        lines.append("---\n")
        for sym, info in by_symbol.items():
            lines.append(f"# {sym}\n")
            rec = info.get("recommendation", "—")
            conf = info.get("confidence", 0)
            lines.append(f"**Recommendation:** {rec}  ")
            lines.append(f"**Confidence:** {int(conf * 100) if isinstance(conf, float) and conf <= 1 else conf}%  ")
            lines.append(f"**Sentiment:** {info.get('market_sentiment', '—')}\n")
            if info.get("key_developments"):
                lines.append(f"### Key Developments\n{info['key_developments']}\n")
            if info.get("sentiment_explanation"):
                lines.append(f"### Sentiment Analysis\n{info['sentiment_explanation']}\n")
            if info.get("risk_factors"):
                lines.append(f"### Risk Factors\n{info['risk_factors']}\n")

            pos = info.get("positioning") or {}
            quality = info.get("quality") or {}
            if pos.get("pc_volume") is not None or quality.get("total_checks"):
                lines.append("### Positioning & Quality")
                if pos.get("pc_volume") is not None:
                    pcoi = (f"{pos['pc_oi']:.2f}"
                            if pos.get("pc_oi") is not None else "n/a")
                    lines.append(
                        f"**Options flow** (chain as of {pos.get('as_of', '?')}): "
                        f"P/C volume {pos['pc_volume']:.2f} "
                        f"({pos.get('put_volume', 0):,} puts / "
                        f"{pos.get('call_volume', 0):,} calls), "
                        f"P/C open interest {pcoi} — {pos.get('read', '')}\n")
                if quality.get("total_checks"):
                    flag = str(quality.get("flag", "")).replace("_", " ").upper()
                    lines.append(
                        f"**Quality screen (Bad Apples)** as of "
                        f"{quality.get('as_of', '?')}: {flag} — "
                        f"{quality['total_fails']}/{quality['total_checks']} "
                        f"checks failed")
                    for f in (quality.get("failed_checks") or [])[:8]:
                        note = f" ({f['note']})" if f.get("note") else ""
                        lines.append(f"- {f['check']}: {f['value']}{note}")
                    for h in (quality.get("red_flags") or [])[:6]:
                        when = f" ({h['date']})" if h.get("date") else ""
                        lines.append(f"- red flag [{h['category']}]: "
                                     f"{h['headline']}{when}")
                    lines.append("")

            thesis = info.get("company_thesis") or {}
            if thesis:
                lines.append("### Company Thesis")
                if thesis.get("perception"):
                    lines.append(f"**Market Perception:** {thesis['perception']}\n")
                if thesis.get("goal_alignment"):
                    lines.append(f"**Goal Alignment:** {thesis['goal_alignment']}\n")
                if thesis.get("positive_catalysts"):
                    lines.append("**Positive Catalysts:**")
                    for c in thesis["positive_catalysts"]:
                        lines.append(f"- {c}")
                    lines.append("")
                if thesis.get("negative_catalysts"):
                    lines.append("**Negative Catalysts:**")
                    for c in thesis["negative_catalysts"]:
                        lines.append(f"- {c}")
                    lines.append("")
                if thesis.get("regime_risks"):
                    lines.append(f"**Regime / Systematic Risks:** {thesis['regime_risks']}\n")
            lines.append("")

    return "\n".join(lines)
