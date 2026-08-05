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
