"""Model prediction infrastructure for quant-news.

Provides base classes, prediction results, and model registry for
Kronos, XGBoost, and LLM agent models.
"""

from models.base import BaseModel, PredictionResult, compute_pnl
from models.registry import ModelRegistry

__all__ = ["BaseModel", "PredictionResult", "compute_pnl", "ModelRegistry"]
