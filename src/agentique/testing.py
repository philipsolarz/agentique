"""Testing utilities for adapters and gateways.

This module provides ready-made test doubles and assertion helpers so
users can write unit and integration tests for their adapters, middleware,
and routing configurations without spinning up real A2A agents.

Classes
-------
MockAdapter
    In-memory ``AgentAdapter`` implementation with configurable responses
    and a call-log for assertions.

InMemoryBridge
    Thin wrapper around ``create_server()`` backed by ``MockAdapter``.
    Returns a live ``FastMCP`` instance for in-process testing.

Functions
---------
assert_adapter_protocol
    Validates that an object satisfies the ``AgentAdapter``
    ``@runtime_checkable`` Protocol via ``isinstance`` check.

Usage
-----
Unit test — adapter protocol compliance::

    from agentique.testing import MockAdapter, assert_adapter_protocol

    def test_custom_adapter_satisfies_protocol():
        adapter = MyCustomAdapter(config)
        assert_adapter_protocol(adapter)

Unit test — routing with MockAdapter::

    from agentique.testing import MockAdapter
    from agentique.core.types import AgentInfo, BridgeContext
    from agentique.bridge.router import AgentRouter

    async def test_routes_to_single_agent():
        agent = AgentInfo(name="bot", base_url="http://bot")
        adapter = MockAdapter([agent], responses={"bot": "pong"})
        ctx = BridgeContext()
        response = await adapter.send_message("bot", "ping", ctx)
        assert response.text == "pong"
        assert adapter.calls[0]["message"] == "ping"

Integration test — InMemoryBridge::

    from agentique.testing import InMemoryBridge
    from agentique.core.types import AgentInfo

    def test_bridge_exposes_agent_tool():
        agent = AgentInfo(name="assistant", base_url="http://assistant")
        bridge = InMemoryBridge(agents=[agent])
        tools = bridge.server._tool_manager.list_tools()
        assert any(t.name == "agent" for t in tools)
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from .core.protocols import AgentAdapter
from .core.types import AgentEvent, AgentInfo, AgentResponse, BridgeContext


# ---------------------------------------------------------------------------
# MockAdapter
# ---------------------------------------------------------------------------


class MockAdapter:
    """In-memory ``AgentAdapter`` for unit tests.

    Implements the full ``AgentAdapter`` protocol.  Responses are
    configurable per agent and the call log records every interaction
    so tests can assert on routing and message content.

    Args:
        agents: Agents to expose via ``discover_agents()``.
        default_response: Text returned for agents not in *responses*.
        responses: Per-agent response strings keyed by agent name.
        events: Per-agent list of ``AgentEvent`` instances to yield from
            ``stream_message()`` instead of the default single text event.

    Attributes:
        calls: List of ``{"agent_id": str, "message": str, "kind": str}``
            dicts appended on every ``send_message`` / ``stream_message``
            call.  ``"kind"`` is ``"send"`` or ``"stream"``.
    """

    def __init__(
        self,
        agents: list[AgentInfo] | None = None,
        *,
        default_response: str = "mock response",
        responses: dict[str, str] | None = None,
        events: dict[str, list[AgentEvent]] | None = None,
    ) -> None:
        self._agents: list[AgentInfo] = agents or []
        self._default_response = default_response
        self._responses: dict[str, str] = responses or {}
        self._events: dict[str, list[AgentEvent]] = events or {}
        self.calls: list[dict[str, Any]] = []

    # -- AgentAdapter protocol --

    async def discover_agents(self) -> list[AgentInfo]:
        """Return the configured agent list."""
        return list(self._agents)

    async def send_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AgentResponse:
        """Return a pre-configured text response, or the default."""
        self.calls.append({"agent_id": agent_id, "message": message, "kind": "send"})
        text = self._responses.get(agent_id, self._default_response)
        return AgentResponse(agent=agent_id, text=text)

    async def stream_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AsyncIterator[AgentEvent]:
        """Yield pre-configured events, or a single default text event."""
        self.calls.append(
            {"agent_id": agent_id, "message": message, "kind": "stream"}
        )
        agent_events = self._events.get(agent_id)
        if agent_events:
            for event in agent_events:
                yield event
        else:
            text = self._responses.get(agent_id, self._default_response)
            yield AgentEvent(kind="message", text=text)

    async def close(self) -> None:
        """No-op: no real connections to clean up."""

    # -- Test helpers --

    def reset_calls(self) -> None:
        """Clear the call log between test assertions."""
        self.calls.clear()

    def call_count(self, agent_id: str | None = None) -> int:
        """Return the number of calls, optionally filtered by agent."""
        if agent_id is None:
            return len(self.calls)
        return sum(1 for c in self.calls if c["agent_id"] == agent_id)

    def last_message(self, agent_id: str | None = None) -> str | None:
        """Return the most recent message sent, optionally for a specific agent."""
        matching = (
            [c for c in self.calls if c["agent_id"] == agent_id]
            if agent_id
            else self.calls
        )
        return matching[-1]["message"] if matching else None

    def set_response(self, agent_id: str, text: str) -> None:
        """Update the response for *agent_id* at runtime."""
        self._responses[agent_id] = text

    def set_events(self, agent_id: str, events: list[AgentEvent]) -> None:
        """Override the event stream for *agent_id* at runtime."""
        self._events[agent_id] = list(events)


# ---------------------------------------------------------------------------
# Protocol assertion
# ---------------------------------------------------------------------------


def assert_adapter_protocol(adapter: Any) -> None:
    """Assert that *adapter* satisfies the ``AgentAdapter`` protocol.

    Uses Python's ``@runtime_checkable`` Protocol mechanism.  Any object
    that structurally implements ``discover_agents``, ``send_message``,
    ``stream_message``, and ``close`` will pass — no inheritance required.

    Args:
        adapter: The adapter instance to validate.

    Raises:
        AssertionError: If *adapter* does not satisfy ``AgentAdapter``.

    Example::

        def test_my_adapter():
            adapter = MyAdapter(config)
            assert_adapter_protocol(adapter)  # raises if protocol missing
    """
    assert isinstance(adapter, AgentAdapter), (
        f"{type(adapter).__name__!r} does not satisfy the AgentAdapter protocol.\n"
        f"Required async methods: discover_agents(), send_message(), "
        f"stream_message(), close().\n"
        f"Actual methods: {[m for m in dir(adapter) if not m.startswith('_')]}"
    )


# ---------------------------------------------------------------------------
# InMemoryBridge
# ---------------------------------------------------------------------------


class InMemoryBridge:
    """FastMCP server backed by a ``MockAdapter`` for integration tests.

    Combines ``create_server()`` with ``MockAdapter`` so tests can verify
    routing decisions, middleware behaviour, and artifact capture end-to-end
    without real agent processes.

    Args:
        agents: Agents to register with the gateway.
        responses: Per-agent text responses for ``MockAdapter``.
        default_response: Fallback response when no per-agent text is set.
        config: Optional ``AgentiqueConfig`` override.
        tool_mapper: Optional ``ToolMapper`` to pass to ``create_server()``.

    Attributes:
        server: The underlying ``FastMCP`` instance.
        adapter: The ``MockAdapter`` backing the bridge.

    Example::

        bridge = InMemoryBridge(
            agents=[AgentInfo("assistant", "http://assistant")],
            responses={"assistant": "Hello from mock!"},
        )
        # Inspect registered tools
        tools = bridge.server._tool_manager.list_tools()
        assert any(t.name == "agent" for t in tools)
    """

    def __init__(
        self,
        agents: list[AgentInfo] | None = None,
        *,
        responses: dict[str, str] | None = None,
        default_response: str = "mock response",
        config: Any | None = None,
        tool_mapper: Any | None = None,
    ) -> None:
        from .core.config import AgentiqueConfig
        from .server import create_server

        _agents = agents or []
        self._mock_adapter = MockAdapter(
            _agents,
            default_response=default_response,
            responses=responses,
        )

        _config = config or AgentiqueConfig(
            name="InMemoryBridge",
            prefetch_cards=False,
            enable_background_tasks=False,
        )

        self._server = create_server(
            agents=_agents,
            adapter=self._mock_adapter,
            config=_config,
            tool_mapper=tool_mapper,
        )

    @property
    def server(self) -> Any:
        """The underlying ``FastMCP`` server instance."""
        return self._server

    @property
    def adapter(self) -> MockAdapter:
        """The ``MockAdapter`` backing the bridge."""
        return self._mock_adapter

    def reset_calls(self) -> None:
        """Convenience: reset the adapter call log."""
        self._mock_adapter.reset_calls()
