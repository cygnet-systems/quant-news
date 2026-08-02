"""Swappable research-agent backend — the seam for ingesting future TradingAgents.

The `trading_agents` model calls a `ResearchAgent` (the `analyze(...)` contract)
rather than a concrete class, so the research strategy is one interchangeable
implementation. Today the default is our in-tree `SingleAgentResearch`; a future
TradingAgents release can be adopted as a second implementation without touching
`models/trading_agents_model.py` or any caller.

Selection (first match wins):
    kwargs `backend=` on predict()  >  env `RESEARCH_BACKEND`  >  MODEL.RESEARCH_BACKEND  >  "single_agent"

To ingest a new TradingAgents version when one ships that is worth adopting:
  1. Pin it:  pip install 'tradingagents==X.Y.Z'  (never depend on an unpinned
     sibling checkout again — that is exactly what broke us at v0.3.1).
  2. Implement `TradingAgentsAdapter.analyze()` to map our (symbol, as_of, ...)
     call onto that version's *public* API and return the standard result dict.
  3. Set RESEARCH_BACKEND=tradingagents.
Callers, the ensemble, persistence, and reports are all unchanged.
"""

import os
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd

from config import MODEL


@runtime_checkable
class ResearchAgent(Protocol):
    """The contract every research backend must satisfy.

    Returns a dict with at least: decision (BUY/SELL/HOLD), confidence (0-1,
    the backend's self-report — the model layer decides how much to trust it),
    raw_response (full text), and optionally triggers / news_count / sector_etf.
    """

    def analyze(
        self,
        symbol: str,
        as_of: str,
        ohlcv_df: Optional[pd.DataFrame] = None,
        news: Optional[list] = None,
        extra_context: str = "",
        use_news: bool = True,
        include_thesis: bool = False,
    ) -> dict[str, Any]:
        ...


def resolve_backend(backend: Optional[str] = None) -> str:
    return (
        backend
        or os.getenv("RESEARCH_BACKEND")
        or getattr(MODEL, "RESEARCH_BACKEND", "single_agent")
    ).lower()


def get_research_agent(
    model: Optional[str] = None, backend: Optional[str] = None
) -> ResearchAgent:
    """Return the configured research backend."""
    name = resolve_backend(backend)
    if name in ("single_agent", "singleagent", "native", "in_tree"):
        from models.single_agent import SingleAgentResearch
        return SingleAgentResearch(model=model)
    if name in ("tradingagents", "ta", "external"):
        return TradingAgentsAdapter(model=model)
    raise ValueError(
        f"Unknown RESEARCH_BACKEND '{name}'. Valid: single_agent, tradingagents."
    )


class TradingAgentsAdapter:
    """Adapter over a pinned external TradingAgents release.

    Intentionally a stub until a version worth adopting ships. It raises a clear,
    actionable error rather than silently degrading — so choosing this backend
    without wiring it fails loudly instead of returning fake HOLDs (the failure
    mode we just removed).
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model

    def analyze(
        self,
        symbol: str,
        as_of: str,
        ohlcv_df: Optional[pd.DataFrame] = None,
        news: Optional[list] = None,
        extra_context: str = "",
        use_news: bool = True,
    ) -> dict[str, Any]:
        # When adopting a release, map onto its PUBLIC API here, e.g.:
        #   from tradingagents.graph.trading_graph import TradingAgentsGraph
        #   from tradingagents.default_config import DEFAULT_CONFIG
        #   cfg = {**DEFAULT_CONFIG, "llm_provider": "anthropic",
        #          "deep_think_llm": self.model, ...}
        #   final_state, rating = TradingAgentsGraph(config=cfg).propagate(symbol, as_of)
        #   return {
        #       "decision": _rating_to_decision(rating),
        #       "confidence": <backend self-report or 0.5>,
        #       "raw_response": final_state["final_trade_decision"],
        #       "triggers": {}, "news_count": 0,
        #       "sector_etf": "",
        #   }
        raise NotImplementedError(
            "TradingAgentsAdapter is a stub. To ingest a TradingAgents release: "
            "pin the version, map its public API in analyze(), then set "
            "RESEARCH_BACKEND=tradingagents. See models/research_backend.py."
        )
