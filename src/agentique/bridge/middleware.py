"""Middleware chain for the bridge layer.

Implements a chain-of-responsibility pattern mirroring ASGI / Starlette
middleware. Each middleware wraps the next, creating a composable stack.

Built-in middleware:

- ``LoggingMiddleware`` — logs tool calls and responses
- ``ErrorMappingMiddleware`` — maps A2A errors to MCP error codes
- ``RateLimitMiddleware`` — simple per-agent rate limiting
- ``MetricsMiddleware`` — collects timing and count metrics
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

try:
    from fastmcp.server.middleware import Middleware as FastMCPMiddleware
except Exception:  # pragma: no cover - fastmcp is a required runtime dependency
    FastMCPMiddleware = object  # type: ignore[assignment]

# Type alias for the call_next function
CallNext = Callable[[dict[str, Any]], Awaitable[Any]]


class MiddlewareChain:
    """Executes a chain of middleware in order.

    Usage::

        chain = MiddlewareChain()
        chain.add(LoggingMiddleware())
        chain.add(ErrorMappingMiddleware())

        result = await chain.execute(request, final_handler)
    """

    def __init__(self) -> None:
        self._middleware: list[Any] = []

    def add(self, middleware: Any) -> "MiddlewareChain":
        """Add a middleware to the chain. Returns self for chaining."""
        self._middleware.append(middleware)
        return self

    async def execute(
        self,
        request: dict[str, Any],
        handler: CallNext,
    ) -> Any:
        """Execute the middleware chain, ending with *handler*."""
        chain = handler
        # Build the chain in reverse so the first-added middleware runs first
        for mw in reversed(self._middleware):
            chain = _wrap(mw, chain)
        return await chain(request)

    @property
    def count(self) -> int:
        return len(self._middleware)


def _wrap(middleware: Any, next_fn: CallNext) -> CallNext:
    """Wrap a middleware around the next handler."""
    async def wrapped(request: dict[str, Any]) -> Any:
        return await middleware.process(request, next_fn)
    return wrapped


# ---------------------------------------------------------------------------
# Built-in middleware implementations
# ---------------------------------------------------------------------------


class LoggingMiddleware:
    """Logs request entry, exit, and errors with timing information."""

    def __init__(self, *, level: int = logging.INFO) -> None:
        self._level = level

    async def process(
        self,
        request: dict[str, Any],
        call_next: CallNext,
    ) -> Any:
        agent = request.get("agent", "unknown")
        action = request.get("action", "message")
        start = time.monotonic()

        logger.log(
            self._level,
            "[middleware] %s → agent=%s",
            action, agent,
        )

        try:
            result = await call_next(request)
            elapsed = time.monotonic() - start
            logger.log(
                self._level,
                "[middleware] %s ← agent=%s (%.2fs)",
                action, agent, elapsed,
            )
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "[middleware] %s ✗ agent=%s error=%s (%.2fs)",
                action, agent, type(exc).__name__, elapsed,
            )
            raise


class ErrorMappingMiddleware:
    """Maps adapter exceptions to agentique typed errors.

    Translates A2A error codes and HTTP errors into the appropriate
    ``AgentiqueError`` subclass with correct ``mcp_code`` and
    ``a2a_code`` fields.
    """

    # A2A JSON-RPC error codes → agentique error types
    A2A_ERROR_MAP: dict[int, str] = {
        -32001: "TaskNotFoundError",
        -32002: "TranslationError",      # ContentTypeNotSupported
        -32003: "AgentUnavailableError",  # UnsupportedOperation
    }

    async def process(
        self,
        request: dict[str, Any],
        call_next: CallNext,
    ) -> Any:
        try:
            return await call_next(request)
        except Exception as exc:
            mapped = self._map_error(exc)
            if mapped is not exc:
                raise mapped from exc
            raise

    def _map_error(self, exc: Exception) -> Exception:
        from agentique.core.errors import (
            AdapterError,
            AgentNotFoundError,
            AgentUnavailableError,
            TaskNotFoundError,
            TranslationError,
        )

        # Check for A2A SDK error codes
        error_code = getattr(exc, "code", None) or getattr(exc, "error_code", None)
        if isinstance(error_code, int) and error_code in self.A2A_ERROR_MAP:
            error_name = self.A2A_ERROR_MAP[error_code]
            error_cls = {
                "TaskNotFoundError": TaskNotFoundError,
                "TranslationError": TranslationError,
                "AgentUnavailableError": AgentUnavailableError,
            }.get(error_name, AdapterError)
            return error_cls(str(exc))

        # Map HTTP-like errors
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(status, int):
            if status == 404:
                return AgentNotFoundError(str(exc))
            if status in {502, 503, 504}:
                return AgentUnavailableError(str(exc))

        # Map connection errors
        exc_name = type(exc).__name__.lower()
        if any(k in exc_name for k in ("connect", "timeout", "refused")):
            return AgentUnavailableError(str(exc))

        return exc


class RateLimitMiddleware:
    """Simple in-memory per-agent rate limiter.

    Uses a sliding window counter. Requests exceeding the limit
    receive an ``AgentUnavailableError``.
    """

    def __init__(
        self,
        *,
        max_requests: int = 100,
        window_seconds: float = 60.0,
    ) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._counters: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def process(
        self,
        request: dict[str, Any],
        call_next: CallNext,
    ) -> Any:
        agent = request.get("agent", "__global__")
        now = time.monotonic()

        async with self._lock:
            timestamps = self._counters[agent]
            # Prune expired entries
            cutoff = now - self._window
            self._counters[agent] = [t for t in timestamps if t > cutoff]
            timestamps = self._counters[agent]

            if len(timestamps) >= self._max:
                from agentique.core.errors import AgentUnavailableError
                raise AgentUnavailableError(
                    f"Rate limit exceeded for agent '{agent}' "
                    f"({self._max} requests per {self._window}s)"
                )
            timestamps.append(now)

        return await call_next(request)


class MetricsMiddleware:
    """Collects simple timing and counter metrics.

    Stores metrics in memory; access via the ``.metrics`` property.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._total_time: dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    async def process(
        self,
        request: dict[str, Any],
        call_next: CallNext,
    ) -> Any:
        agent = request.get("agent", "__global__")
        start = time.monotonic()

        try:
            result = await call_next(request)
            elapsed = time.monotonic() - start
            async with self._lock:
                self._counts[agent] += 1
                self._total_time[agent] += elapsed
            return result
        except Exception:
            async with self._lock:
                self._errors[agent] += 1
            raise

    @property
    def metrics(self) -> dict[str, dict[str, Any]]:
        """Return current metrics snapshot."""
        agents = set(self._counts) | set(self._errors)
        result: dict[str, dict[str, Any]] = {}
        for agent in sorted(agents):
            count = self._counts.get(agent, 0)
            result[agent] = {
                "requests": count,
                "errors": self._errors.get(agent, 0),
                "total_time_s": round(self._total_time.get(agent, 0), 3),
                "avg_time_s": round(
                    self._total_time.get(agent, 0) / count, 3
                ) if count > 0 else 0.0,
            }
        return result


class FastMCPBridgeMiddleware(FastMCPMiddleware):
    """Adapter that runs the bridge ``MiddlewareChain`` inside FastMCP hooks."""

    def __init__(self, chain: MiddlewareChain) -> None:
        self._chain = chain

    async def on_call_tool(
        self,
        context: Any,
        call_next: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        message = getattr(context, "message", None)
        request = {
            "action": "tools/call",
            "agent": getattr(message, "name", "__unknown__"),
            "arguments": getattr(message, "arguments", {}) or {},
        }

        async def final_handler(payload: dict[str, Any]) -> Any:
            next_context = _copy_context_with_message(context, payload)
            return await call_next(next_context)

        return await self._chain.execute(request, final_handler)

    async def on_list_tools(
        self,
        context: Any,
        call_next: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        request = {"action": "tools/list", "agent": "__catalog__"}

        async def final_handler(payload: dict[str, Any]) -> Any:
            return await call_next(context)

        return await self._chain.execute(request, final_handler)

    async def on_read_resource(
        self,
        context: Any,
        call_next: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        message = getattr(context, "message", None)
        request = {
            "action": "resources/read",
            "agent": "__resource__",
            "uri": getattr(message, "uri", None),
        }

        async def final_handler(payload: dict[str, Any]) -> Any:
            return await call_next(context)

        return await self._chain.execute(request, final_handler)


def _copy_context_with_message(context: Any, payload: dict[str, Any]) -> Any:
    """Copy middleware context with updated call-tool message fields."""
    message = getattr(context, "message", None)
    if message is None:
        return context

    updates: dict[str, Any] = {}
    requested_name = payload.get("agent")
    if isinstance(requested_name, str) and requested_name:
        updates["name"] = requested_name

    arguments = payload.get("arguments")
    if isinstance(arguments, dict):
        updates["arguments"] = arguments

    if not updates:
        return context

    updated_message = message
    if hasattr(message, "model_copy"):
        try:
            updated_message = message.model_copy(update=updates)
        except Exception:
            updated_message = message
    else:
        try:
            for key, value in updates.items():
                setattr(updated_message, key, value)
        except Exception:
            return context

    if hasattr(context, "copy"):
        try:
            return context.copy(message=updated_message)
        except Exception:
            return context
    return context
