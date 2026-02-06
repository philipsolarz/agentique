"""Protocol-specific adapters and built-in adapter registration."""

from __future__ import annotations

from typing import Any

from agentique.core.registry import register_adapter
from agentique.core.types import AgentInfo

from .a2a import A2AAgentAdapter
from .http import HttpAgentAdapter


@register_adapter("a2a")
class A2AAdapterFactory:
    """Factory for creating A2A adapters."""

    protocol_name = "a2a"

    def create(
        self,
        agents: dict[str, AgentInfo],
        **kwargs: Any,
    ) -> A2AAgentAdapter:
        return A2AAgentAdapter(agents, **kwargs)


@register_adapter("http")
class HttpAdapterFactory:
    """Factory for creating generic HTTP adapters."""

    protocol_name = "http"

    def create(
        self,
        agents: dict[str, AgentInfo],
        **kwargs: Any,
    ) -> HttpAgentAdapter:
        return HttpAgentAdapter(agents, **kwargs)


__all__ = [
    "A2AAgentAdapter",
    "HttpAgentAdapter",
    "A2AAdapterFactory",
    "HttpAdapterFactory",
]
