"""Adapter registry with setuptools entry-point discovery.

Built-in adapters register via the ``@register_adapter`` decorator.
Third-party adapters register via setuptools entry points::

    [project.entry-points."agentique.adapters"]
    openai = "agentique_openai:OpenAIAdapterFactory"

Discovery is automatic: ``pip install agentique-openai`` makes the
adapter available without explicit configuration.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from .protocols import AdapterFactory, AgentAdapter
from .types import AgentInfo

logger = logging.getLogger(__name__)

# Internal registry populated by @register_adapter
_REGISTRY: dict[str, type] = {}


def register_adapter(protocol_name: str):
    """Decorator to register a built-in adapter factory.

    Usage::

        @register_adapter("a2a")
        class A2AAdapterFactory:
            protocol_name = "a2a"

            def create(self, agents, **kwargs):
                return A2AAgentAdapter(agents, **kwargs)
    """
    def decorator(cls: type) -> type:
        _REGISTRY[protocol_name] = cls
        return cls
    return decorator


def discover_adapters() -> dict[str, type]:
    """Return all known adapter factories (built-in + entry points).

    Returns:
        Mapping from protocol name to factory class.
    """
    result = dict(_REGISTRY)

    # Discover third-party adapters via entry points
    try:
        eps = entry_points(group="agentique.adapters")
        for ep in eps:
            if ep.name in result:
                logger.debug(
                    "Entry point '%s' shadows built-in adapter", ep.name
                )
            try:
                factory_cls = ep.load()
                result[ep.name] = factory_cls
                logger.info(
                    "Discovered adapter '%s' from %s", ep.name, ep.value
                )
            except Exception:
                logger.warning(
                    "Failed to load adapter entry point '%s'", ep.name,
                    exc_info=True,
                )
    except Exception:
        logger.debug("Entry point discovery unavailable", exc_info=True)

    return result


def create_adapter(
    protocol: str,
    agents: dict[str, AgentInfo],
    **kwargs: Any,
) -> AgentAdapter:
    """Create an adapter by protocol name.

    Args:
        protocol: Protocol identifier (e.g. ``"a2a"``, ``"http"``).
        agents: Agent descriptors to pass to the factory.
        **kwargs: Additional configuration forwarded to the factory.

    Returns:
        A configured adapter instance.

    Raises:
        ValueError: If no factory is registered for *protocol*.
    """
    factories = discover_adapters()
    factory_cls = factories.get(protocol)
    if factory_cls is None:
        available = sorted(factories.keys()) or ["(none)"]
        raise ValueError(
            f"No adapter registered for protocol '{protocol}'. "
            f"Available: {', '.join(available)}"
        )
    factory = factory_cls()
    return factory.create(agents, **kwargs)


def list_protocols() -> list[str]:
    """Return sorted list of all known protocol names."""
    return sorted(discover_adapters().keys())
