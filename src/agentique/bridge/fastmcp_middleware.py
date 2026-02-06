"""FastMCP server-level middleware for agentique.

Bridges agentique's bridge-layer middleware into FastMCP 3.0's native
``Middleware`` class, providing cross-cutting concerns at the MCP
protocol level (tool calls, listing, resources).

Usage::

    from agentique.bridge.fastmcp_middleware import AgentiqueMiddleware

    mcp = FastMCP("server")
    mcp.add_middleware(AgentiqueMiddleware(emitter=emitter))
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from fastmcp.server.middleware.middleware import (
    CallNext,
    Middleware,
    MiddlewareContext,
)
from fastmcp.tools.tool import Tool, ToolResult

from agentique.core.events import AsyncEventEmitter

logger = logging.getLogger(__name__)


class AgentiqueMiddleware(Middleware):
    """FastMCP-level middleware for agentique cross-cutting concerns.

    Provides:
        - Event emission on tool calls (``tool.called`` / ``tool.completed`` / ``tool.failed``)
        - Request timing and logging
        - OpenTelemetry span attribute injection (when tracing is active)
    """

    def __init__(
        self,
        *,
        emitter: AsyncEventEmitter | None = None,
        enable_tracing: bool = True,
    ) -> None:
        self._emitter = emitter or AsyncEventEmitter()
        self._enable_tracing = enable_tracing

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, ToolResult],
    ) -> ToolResult:
        """Intercept tool calls for logging, events, and tracing."""
        tool_name = _extract_tool_name(context)
        start = time.monotonic()

        await self._emitter.emit(
            "tool.called", tool_name=tool_name, timestamp=start,
        )

        if self._enable_tracing:
            _add_trace_attributes(tool_name)

        try:
            result = await call_next(context)
            elapsed = time.monotonic() - start
            logger.info(
                "[agentique] tool '%s' completed (%.2fs)", tool_name, elapsed,
            )
            await self._emitter.emit(
                "tool.completed",
                tool_name=tool_name,
                elapsed=elapsed,
            )
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "[agentique] tool '%s' failed: %s (%.2fs)",
                tool_name, type(exc).__name__, elapsed,
            )
            await self._emitter.emit(
                "tool.failed",
                tool_name=tool_name,
                error=str(exc),
                elapsed=elapsed,
            )
            raise

    async def on_list_tools(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Sequence[Tool]],
    ) -> Sequence[Tool]:
        """Log tool listing requests."""
        tools = await call_next(context)
        logger.debug("[agentique] listed %d tools", len(tools))
        return tools


def _extract_tool_name(context: MiddlewareContext[Any]) -> str:
    """Extract the tool name from a call_tool middleware context."""
    msg = context.message
    if hasattr(msg, "name"):
        return str(msg.name)
    if isinstance(msg, dict):
        return str(msg.get("name", "unknown"))
    return "unknown"


def _add_trace_attributes(tool_name: str) -> None:
    """Add agentique-specific attributes to the current OTel span."""
    try:
        from fastmcp.telemetry import get_tracer

        tracer = get_tracer()
        # The span is typically already created by FastMCP;
        # we add attributes to the current span if available.
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute("agentique.tool_name", tool_name)
                span.set_attribute("agentique.protocol", "bridge")
        except ImportError:
            pass  # OTel SDK not installed — no-op
    except ImportError:
        pass  # FastMCP telemetry not available
