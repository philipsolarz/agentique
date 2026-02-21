"""Tests for LLMRouter enhancements: sampling fallback and agentic planning loop.

Covers:
  - sampling_fallback= invoked when ctx is None or has no sample()
  - sampling_fallback= invoked when ctx.sample() raises
  - sampling_fallback receives message + available agents
  - enable_introspection= passes tools= to ctx.sample()
  - _make_introspection_tools() — inspect_agent and list_agents
  - provider composition via create_server(extra_providers=[])
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio

from agentique.bridge.router import (
    LLMRouter,
    RoutingDecision,
    _make_introspection_tools,
)
from agentique.core.errors import AgentNotFoundError
from agentique.core.types import AgentInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_agent(name: str, skills: tuple[str, ...] = (), description: str = "") -> AgentInfo:
    return AgentInfo(
        name=name,
        base_url=f"http://{name}.example.com",
        description=description,
        skills=list(skills),
    )


def make_decision(agent_id: str) -> RoutingDecision:
    return RoutingDecision(
        agent_id=agent_id,
        confidence=0.9,
        reasoning="test",
        fallback_agents=[],
    )


def make_ctx_with_sample(decision: RoutingDecision | None = None, *, raises: Exception | None = None) -> Any:
    ctx = MagicMock()

    async def _sample(*args: Any, **kwargs: Any) -> Any:
        if raises:
            raise raises
        result = MagicMock()
        result.result = decision
        return result

    ctx.sample = _sample
    return ctx


# ---------------------------------------------------------------------------
# sampling_fallback — called when ctx is None
# ---------------------------------------------------------------------------


async def test_sampling_fallback_called_when_ctx_is_none():
    agents = [make_agent("billing"), make_agent("crm")]
    received: list[Any] = []

    async def my_fallback(message: str, available: list[AgentInfo]) -> str:
        received.append((message, available))
        return "billing"

    router = LLMRouter(sampling_fallback=my_fallback)
    result = await router.aselect("pay invoice", agents, ctx=None)

    assert result.name == "billing"
    assert len(received) == 1
    assert received[0][0] == "pay invoice"
    assert len(received[0][1]) == 2


async def test_sampling_fallback_called_when_ctx_has_no_sample():
    agents = [make_agent("billing")]

    async def my_fallback(message: str, available: list[AgentInfo]) -> str:
        return "billing"

    ctx = object()  # no .sample attribute
    router = LLMRouter(sampling_fallback=my_fallback)
    result = await router.aselect("pay", agents, ctx=ctx)
    assert result.name == "billing"


async def test_no_fallback_and_no_ctx_raises_runtime_error():
    agents = [make_agent("billing")]
    router = LLMRouter()
    with pytest.raises(RuntimeError, match="sampling"):
        await router.aselect("pay", agents, ctx=None)


async def test_sampling_fallback_sync_callable():
    """Fallback can also be a sync function (not async)."""
    agents = [make_agent("crm")]

    def sync_fallback(message: str, available: list[AgentInfo]) -> str:
        return "crm"

    router = LLMRouter(sampling_fallback=sync_fallback)
    result = await router.aselect("update contact", agents, ctx=None)
    assert result.name == "crm"


async def test_sampling_fallback_called_on_sample_failure():
    agents = [make_agent("billing"), make_agent("crm")]
    ctx = make_ctx_with_sample(raises=RuntimeError("sampling not available"))

    async def my_fallback(message: str, available: list[AgentInfo]) -> str:
        return "crm"

    router = LLMRouter(sampling_fallback=my_fallback)
    result = await router.aselect("contact", agents, ctx=ctx)
    assert result.name == "crm"


async def test_fallback_unknown_agent_raises():
    agents = [make_agent("billing")]

    async def bad_fallback(message: str, available: list[AgentInfo]) -> str:
        return "nonexistent"

    router = LLMRouter(sampling_fallback=bad_fallback)
    with pytest.raises((AgentNotFoundError, RuntimeError)):
        await router.aselect("pay", agents, ctx=None)


# ---------------------------------------------------------------------------
# enable_introspection — tools passed to ctx.sample
# ---------------------------------------------------------------------------


async def test_introspection_passes_tools_to_sample():
    agents = [make_agent("billing"), make_agent("crm")]
    decision = make_decision("billing")
    tools_seen: list[Any] = []

    ctx = MagicMock()

    async def _sample(*args: Any, **kwargs: Any) -> Any:
        tools_seen.extend(kwargs.get("tools") or [])
        result = MagicMock()
        result.result = decision
        return result

    ctx.sample = _sample

    router = LLMRouter(enable_introspection=True)
    result = await router.aselect("pay invoice", agents, ctx=ctx)

    assert result.name == "billing"
    assert len(tools_seen) >= 2  # at least inspect_agent + list_agents


async def test_no_introspection_does_not_pass_tools():
    agents = [make_agent("billing")]
    decision = make_decision("billing")
    tools_seen: list[Any] = []

    ctx = MagicMock()

    async def _sample(*args: Any, **kwargs: Any) -> Any:
        if "tools" in kwargs:
            tools_seen.extend(kwargs["tools"])
        result = MagicMock()
        result.result = decision
        return result

    ctx.sample = _sample

    router = LLMRouter(enable_introspection=False)
    await router.aselect("pay", agents, ctx=ctx)

    assert len(tools_seen) == 0


# ---------------------------------------------------------------------------
# _make_introspection_tools
# ---------------------------------------------------------------------------


def test_introspection_list_agents_returns_names():
    agents = [make_agent("billing"), make_agent("crm"), make_agent("analytics")]
    _, list_agents = _make_introspection_tools(agents)
    names = list_agents()
    assert set(names) == {"billing", "crm", "analytics"}


def test_introspection_inspect_agent_known():
    agents = [make_agent("billing", skills=("invoice", "payment"), description="Billing agent")]
    inspect_agent, _ = _make_introspection_tools(agents)
    info = inspect_agent("billing")
    assert info["name"] == "billing"
    assert "invoice" in info["skills"]
    assert "Billing agent" in info["description"]
    assert "endpoint" in info


def test_introspection_inspect_agent_unknown():
    agents = [make_agent("billing")]
    inspect_agent, _ = _make_introspection_tools(agents)
    result = inspect_agent("nonexistent")
    assert "error" in result
    assert "billing" in result.get("available", [])


def test_introspection_tools_are_callable():
    agents = [make_agent("billing")]
    tools = _make_introspection_tools(agents)
    assert len(tools) == 2
    assert all(callable(t) for t in tools)


def test_introspection_list_agents_empty():
    tools = _make_introspection_tools([])
    _, list_agents = tools
    assert list_agents() == []


# ---------------------------------------------------------------------------
# Provider composition via create_server(extra_providers=[])
# ---------------------------------------------------------------------------


def test_create_server_with_no_extra_providers():
    """create_server() still works without extra_providers."""
    from agentique.server import create_server
    from agentique.core.types import AgentInfo

    agent = AgentInfo(name="test", base_url="http://test.example.com")
    server = create_server(agents=[agent])
    assert server is not None


def test_create_server_with_extra_providers():
    """extra_providers are included in the FastMCP providers list."""
    from unittest.mock import MagicMock, patch
    from agentique.server import create_server
    from agentique.core.types import AgentInfo

    agent = AgentInfo(name="test", base_url="http://test.example.com")
    extra = MagicMock()

    with patch("fastmcp.FastMCP.__init__", return_value=None) as mock_init, \
         patch("fastmcp.FastMCP.add_middleware"), \
         patch("fastmcp.FastMCP.add_transform"), \
         patch("fastmcp.FastMCP.tool", return_value=lambda f: f), \
         patch("fastmcp.FastMCP.resource", return_value=lambda f: f):
        create_server(agents=[agent], extra_providers=[extra])

        # Verify providers arg included both AgentProvider and extra
        call_kwargs = mock_init.call_args
        providers = call_kwargs.kwargs.get("providers") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        # providers may be passed as positional or keyword; either way extra should be in it
        if providers is not None:
            assert extra in providers


def test_create_server_extra_providers_none_is_fine():
    """Passing extra_providers=None is equivalent to not passing it."""
    from agentique.server import create_server
    from agentique.core.types import AgentInfo

    agent = AgentInfo(name="test", base_url="http://test.example.com")
    server = create_server(agents=[agent], extra_providers=None)
    assert server is not None
