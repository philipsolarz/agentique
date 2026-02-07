"""Tests for the FastMCP-level middleware (AgentiqueMiddleware)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import pytest

from agentique.bridge.fastmcp_middleware import AgentiqueMiddleware
from agentique.core.events import AsyncEventEmitter


# Mock the FastMCP middleware context
@dataclass(frozen=True)
class MockMiddlewareContext:
    message: Any = None
    method: str | None = None
    source: str = "client"
    type: str = "request"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fastmcp_context: Any = None


@dataclass
class MockToolResult:
    content: str = "ok"


class MockCallToolMessage:
    name = "test_tool"


# ---- Tests ----


@pytest.mark.asyncio
async def test_agentique_middleware_emits_events():
    """AgentiqueMiddleware should emit tool.called and tool.completed events."""
    emitter = AsyncEventEmitter()
    events_received: list[str] = []

    emitter.on("tool.called", lambda **kw: events_received.append("called"))
    emitter.on("tool.completed", lambda **kw: events_received.append("completed"))

    mw = AgentiqueMiddleware(emitter=emitter)

    context = MockMiddlewareContext(message=MockCallToolMessage())
    result = MockToolResult()

    async def call_next(ctx: Any) -> MockToolResult:
        return result

    got = await mw.on_call_tool(context, call_next)

    assert got is result
    assert "called" in events_received
    assert "completed" in events_received


@pytest.mark.asyncio
async def test_agentique_middleware_emits_failed_on_error():
    """AgentiqueMiddleware should emit tool.failed when the tool raises."""
    emitter = AsyncEventEmitter()
    events_received: list[str] = []

    emitter.on("tool.failed", lambda **kw: events_received.append("failed"))

    mw = AgentiqueMiddleware(emitter=emitter)

    context = MockMiddlewareContext(message=MockCallToolMessage())

    async def call_next(ctx: Any) -> Any:
        raise RuntimeError("test error")

    with pytest.raises(RuntimeError):
        await mw.on_call_tool(context, call_next)

    assert "failed" in events_received


@pytest.mark.asyncio
async def test_agentique_middleware_list_tools():
    """on_list_tools should pass through the tools list."""
    mw = AgentiqueMiddleware()
    context = MockMiddlewareContext()
    tools = ["tool_a", "tool_b"]

    async def call_next(ctx: Any) -> list[str]:
        return tools

    result = await mw.on_list_tools(context, call_next)
    assert result == tools


@pytest.mark.asyncio
async def test_agentique_middleware_extract_tool_name():
    """Should extract tool name from message object."""
    from agentique.bridge.fastmcp_middleware import _extract_tool_name

    # Object with name attribute
    class Named:
        name = "my_tool"

    ctx = MockMiddlewareContext(message=Named())
    assert _extract_tool_name(ctx) == "my_tool"

    # Dict with name key
    ctx2 = MockMiddlewareContext(message={"name": "dict_tool"})
    assert _extract_tool_name(ctx2) == "dict_tool"

    # Unknown
    ctx3 = MockMiddlewareContext(message=42)
    assert _extract_tool_name(ctx3) == "unknown"
