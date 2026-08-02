"""Strategy registry with auto-discovery.

Scans the strategies/ package for BaseStrategy subclasses on init.
Drop a new .py file with a BaseStrategy subclass — it registers automatically.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Optional

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_SKIP_MODULES = {"base", "registry", "__init__"}


class StrategyRegistry:
    """Auto-discovers and manages strategy instances."""

    def __init__(self) -> None:
        self._strategies: dict[str, BaseStrategy] = {}
        self._auto_discover()

    def _auto_discover(self) -> None:
        """Import all modules in strategies/ and register BaseStrategy subclasses."""
        package_dir = Path(__file__).parent
        for module_info in pkgutil.iter_modules([str(package_dir)]):
            if module_info.name in _SKIP_MODULES:
                continue
            try:
                module = importlib.import_module(f"strategies.{module_info.name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseStrategy)
                        and attr is not BaseStrategy
                    ):
                        instance = attr()
                        self.register(instance)
            except Exception as e:
                logger.warning(f"Failed to load strategy '{module_info.name}': {e}")

    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy instance."""
        if strategy.name in self._strategies:
            logger.warning(f"Strategy '{strategy.name}' already registered, skipping")
            return
        self._strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name} v{strategy.version}")

    def get(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)

    def list_strategies(self) -> list[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())

    def __iter__(self):
        """Iterate over (name, strategy) pairs."""
        return iter(self._strategies.items())

    def __len__(self) -> int:
        return len(self._strategies)
