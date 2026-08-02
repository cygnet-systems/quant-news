"""Model registry for managing prediction models.

Provides centralized registration, lookup, and iteration over models.
"""

import logging
from typing import Optional

from models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for prediction models.

    Supports registration, lookup by name, and iteration over all models.
    """

    def __init__(self) -> None:
        self._models: dict[str, BaseModel] = {}

    def register(self, model: BaseModel) -> None:
        """Register a model instance.

        Args:
            model: Model instance to register.

        Raises:
            ValueError: If a model with the same name is already registered.
        """
        if model.name in self._models:
            raise ValueError(f"Model '{model.name}' already registered")
        self._models[model.name] = model
        logger.info(f"Registered model: {model.name} (ready={model.is_ready()})")

    def get(self, name: str) -> Optional[BaseModel]:
        """Get a model by name.

        Args:
            name: Model identifier.

        Returns:
            Model instance or None if not found.
        """
        return self._models.get(name)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_ready_models(self) -> list[str]:
        """List names of models that are ready to predict."""
        return [name for name, model in self._models.items() if model.is_ready()]

    def __iter__(self):
        """Iterate over (name, model) pairs."""
        return iter(self._models.items())

    def __len__(self) -> int:
        return len(self._models)
