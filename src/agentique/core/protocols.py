"""Protocol definitions for agentique's pluggable architecture.

All public interfaces are ``typing.Protocol`` classes so that adapters,
routers, and middleware satisfy them via structural subtyping — no
inheritance from a base class is ever required.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Awaitable, Protocol, runtime_checkable

from .types import AgentEvent, AgentInfo, AgentResponse, BridgeContext


# ---------------------------------------------------------------------------
# Adapter protocol — the contract every backend must satisfy
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentAdapter(Protocol):
    """Protocol for agent backend adapters (A2A, OpenAI, HTTP, …).

    Any class that implements these four async methods is a valid adapter.
    """

    async def discover_agents(self) -> list[AgentInfo]:
        """Return the agents reachable through this adapter."""
        ...

    async def send_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AgentResponse:
        """Send a message and wait for the full response."""
        ...

    async def stream_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AsyncIterator[AgentEvent]:
        """Send a message and yield events as they arrive."""
        ...  # pragma: no cover
        # yield is needed so the type-checker sees this as AsyncIterator
        if False:
            yield  # type: ignore[misc]

    async def close(self) -> None:
        """Release resources held by the adapter."""
        ...


# ---------------------------------------------------------------------------
# Tool mapper protocol — controls how agent capabilities become MCP tools
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolMapper(Protocol):
    """Protocol for mapping agent info into MCP component definitions.

    A default implementation creates one tool per agent. Users may
    provide mappers that create one tool per skill, flatten sub-agent
    hierarchies, or apply custom naming conventions.
    """

    def map_tools(self, agent: AgentInfo) -> list[dict[str, Any]]:
        """Return MCP tool definitions for *agent*."""
        ...

    def map_resources(self, agent: AgentInfo) -> list[dict[str, Any]]:
        """Return MCP resource definitions for *agent*."""
        ...

    def map_prompts(self, agent: AgentInfo) -> list[dict[str, Any]]:
        """Return MCP prompt definitions for *agent*."""
        ...


# ---------------------------------------------------------------------------
# Middleware protocol — chain-of-responsibility processing
# ---------------------------------------------------------------------------


@runtime_checkable
class BridgeMiddleware(Protocol):
    """Middleware that wraps bridge request/response processing.

    Follows the ASGI / Starlette middleware pattern: each middleware
    calls ``call_next`` to invoke the rest of the chain.
    """

    async def process(
        self,
        request: dict[str, Any],
        call_next: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        """Process a bridge request, optionally delegating to *call_next*."""
        ...


# ---------------------------------------------------------------------------
# Adapter registry protocol — pluggable adapter discovery
# ---------------------------------------------------------------------------


@runtime_checkable
class AdapterFactory(Protocol):
    """Protocol for adapter factories that create adapters from config.

    Entry-point based discovery enables ``pip install agentique-openai``
    to make the adapter available without explicit imports.
    """

    def create(
        self,
        agents: dict[str, AgentInfo],
        **kwargs: Any,
    ) -> AgentAdapter:
        """Create an adapter instance from agent descriptors and config."""
        ...

    @property
    def protocol_name(self) -> str:
        """Short identifier for the protocol (e.g. 'a2a', 'openai', 'http')."""
        ...
