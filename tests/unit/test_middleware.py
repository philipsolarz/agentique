"""Tests for the middleware chain and built-in middleware."""

from __future__ import annotations

import asyncio
import pytest

from agentique.bridge.middleware import (
    ErrorMappingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareChain,
    RateLimitMiddleware,
)
from agentique.core.errors import AgentUnavailableError


# ---- MiddlewareChain ----


@pytest.mark.asyncio
async def test_chain_empty():
    """An empty chain should just call the handler."""
    chain = MiddlewareChain()
    result = await chain.execute({"x": 1}, _echo_handler)
    assert result == {"x": 1}


@pytest.mark.asyncio
async def test_chain_single_middleware():
    """A single middleware should wrap the handler."""
    chain = MiddlewareChain()
    chain.add(_AddFieldMiddleware("added", True))
    result = await chain.execute({"x": 1}, _echo_handler)
    assert result == {"x": 1, "added": True}


@pytest.mark.asyncio
async def test_chain_ordering():
    """Middleware should execute in add-order (first added runs first)."""
    order: list[str] = []

    class OrderMiddleware:
        def __init__(self, name: str) -> None:
            self._name = name

        async def process(self, request, call_next):
            order.append(f"{self._name}-before")
            result = await call_next(request)
            order.append(f"{self._name}-after")
            return result

    chain = MiddlewareChain()
    chain.add(OrderMiddleware("A"))
    chain.add(OrderMiddleware("B"))

    await chain.execute({}, _echo_handler)
    assert order == ["A-before", "B-before", "B-after", "A-after"]


@pytest.mark.asyncio
async def test_chain_fluent():
    """Chain.add() should return self for fluent chaining."""
    chain = MiddlewareChain()
    result = chain.add(LoggingMiddleware()).add(ErrorMappingMiddleware())
    assert result is chain
    assert chain.count == 2


# ---- LoggingMiddleware ----


@pytest.mark.asyncio
async def test_logging_middleware_passes_through():
    chain = MiddlewareChain()
    chain.add(LoggingMiddleware())
    result = await chain.execute({"agent": "test"}, _echo_handler)
    assert result == {"agent": "test"}


# ---- ErrorMappingMiddleware ----


@pytest.mark.asyncio
async def test_error_mapping_connection_error():
    """Connection errors should be mapped to AgentUnavailableError."""
    chain = MiddlewareChain()
    chain.add(ErrorMappingMiddleware())

    async def handler(req):
        raise ConnectionRefusedError("refused")

    with pytest.raises(AgentUnavailableError):
        await chain.execute({"agent": "test"}, handler)


@pytest.mark.asyncio
async def test_error_mapping_passthrough():
    """Non-mapped errors should pass through unchanged."""
    chain = MiddlewareChain()
    chain.add(ErrorMappingMiddleware())

    async def handler(req):
        raise ValueError("not mapped")

    with pytest.raises(ValueError, match="not mapped"):
        await chain.execute({"agent": "test"}, handler)


# ---- RateLimitMiddleware ----


@pytest.mark.asyncio
async def test_rate_limit_allows_normal_traffic():
    chain = MiddlewareChain()
    chain.add(RateLimitMiddleware(max_requests=5, window_seconds=10))

    for _ in range(5):
        result = await chain.execute({"agent": "test"}, _echo_handler)
        assert result == {"agent": "test"}


@pytest.mark.asyncio
async def test_rate_limit_blocks_excess():
    chain = MiddlewareChain()
    chain.add(RateLimitMiddleware(max_requests=2, window_seconds=60))

    await chain.execute({"agent": "test"}, _echo_handler)
    await chain.execute({"agent": "test"}, _echo_handler)

    with pytest.raises(AgentUnavailableError, match="Rate limit"):
        await chain.execute({"agent": "test"}, _echo_handler)


# ---- MetricsMiddleware ----


@pytest.mark.asyncio
async def test_metrics_collection():
    mw = MetricsMiddleware()
    chain = MiddlewareChain()
    chain.add(mw)

    await chain.execute({"agent": "a"}, _echo_handler)
    await chain.execute({"agent": "a"}, _echo_handler)
    await chain.execute({"agent": "b"}, _echo_handler)

    metrics = mw.metrics
    assert metrics["a"]["requests"] == 2
    assert metrics["b"]["requests"] == 1
    assert metrics["a"]["errors"] == 0


@pytest.mark.asyncio
async def test_metrics_tracks_errors():
    mw = MetricsMiddleware()
    chain = MiddlewareChain()
    chain.add(mw)

    async def failing(req):
        raise RuntimeError("oops")

    with pytest.raises(RuntimeError):
        await chain.execute({"agent": "x"}, failing)

    assert mw.metrics["x"]["errors"] == 1


# ---- Helpers ----


async def _echo_handler(request):
    return request


class _AddFieldMiddleware:
    def __init__(self, key: str, value) -> None:
        self._key = key
        self._value = value

    async def process(self, request, call_next):
        request[self._key] = self._value
        return await call_next(request)
