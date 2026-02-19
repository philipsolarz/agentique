"""Tests for agentique.bridge.lifespans — composable lifespan factories.

Covers:
  - make_cleanup_lifespan: pool.close() called on exit, clear_deps_fn called
  - make_cleanup_lifespan with pool=None: no error
  - make_health_monitor_lifespan: task started, cancelled on exit, on_degraded callback
  - compose_lifespans: empty, single, multiple, None entries skipped
  - | pipe operator composes two lifespans correctly (via compose)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anyio

import pytest

pytestmark = pytest.mark.anyio

from agentique.bridge.lifespans import (
    compose_lifespans,
    make_cleanup_lifespan,
    make_health_monitor_lifespan,
)


# ---------------------------------------------------------------------------
# make_cleanup_lifespan
# ---------------------------------------------------------------------------


async def _run_lifespan(lifespan_instance: Any, server: Any = None) -> None:
    """Drive a lifespan: enter it, then exit it immediately."""
    if server is None:
        server = MagicMock()
    async with lifespan_instance(server):
        pass  # yield point — lifespan body runs here


async def test_cleanup_lifespan_closes_pool():
    pool = AsyncMock()
    ls = make_cleanup_lifespan(pool)
    await _run_lifespan(ls)
    pool.close.assert_awaited_once()


async def test_cleanup_lifespan_none_pool_no_error():
    """pool=None should not raise."""
    ls = make_cleanup_lifespan(None)
    await _run_lifespan(ls)  # should complete without error


async def test_cleanup_lifespan_calls_clear_deps_fn():
    pool = AsyncMock()
    cleared: list[int] = []
    ls = make_cleanup_lifespan(pool, clear_deps_fn=lambda: cleared.append(1))
    await _run_lifespan(ls)
    assert len(cleared) == 1


async def test_cleanup_lifespan_clear_deps_fn_called_even_on_pool_error():
    pool = AsyncMock()
    pool.close.side_effect = RuntimeError("close failed")
    cleared: list[int] = []
    ls = make_cleanup_lifespan(pool, clear_deps_fn=lambda: cleared.append(1))
    await _run_lifespan(ls)
    # clear_deps_fn must still be called even if pool.close raises
    assert len(cleared) == 1


async def test_cleanup_lifespan_returns_lifespan_instance():
    pool = AsyncMock()
    ls = make_cleanup_lifespan(pool)
    assert ls is not None
    # FastMCP Lifespan instances are callable context managers
    assert callable(ls)


async def test_cleanup_lifespan_yields_empty_dict():
    pool = AsyncMock()
    ls = make_cleanup_lifespan(pool)
    server = MagicMock()
    async with ls(server) as ctx:
        assert isinstance(ctx, dict)


# ---------------------------------------------------------------------------
# make_health_monitor_lifespan
# ---------------------------------------------------------------------------


async def test_health_monitor_starts_and_stops():
    adapter = MagicMock()
    adapter.health_check = AsyncMock()
    ls = make_health_monitor_lifespan(adapter, interval=0.01)
    server = MagicMock()
    async with ls(server) as ctx:
        assert "health_monitor_task" in ctx
        await anyio.sleep(0.05)  # let at least one check run
    # After exit the task group / task is cancelled — no exception means success


async def test_health_monitor_no_health_check_method():
    """Adapters without health_check() should not cause errors."""
    adapter = MagicMock(spec=[])  # no health_check attribute
    ls = make_health_monitor_lifespan(adapter, interval=0.01)
    async with ls(MagicMock()) as ctx:
        await anyio.sleep(0.05)
    # No exception should have been raised


async def test_health_monitor_on_degraded_callback():
    failures: list[Exception] = []

    def on_degraded(name: str, exc: Exception) -> None:
        failures.append(exc)

    adapter = MagicMock()
    adapter.health_check = AsyncMock(side_effect=RuntimeError("down"))

    ls = make_health_monitor_lifespan(
        adapter, interval=0.01, on_degraded=on_degraded
    )
    async with ls(MagicMock()):
        await anyio.sleep(0.05)

    assert len(failures) > 0
    assert all(isinstance(e, RuntimeError) for e in failures)


async def test_health_monitor_returns_lifespan_instance():
    adapter = MagicMock()
    ls = make_health_monitor_lifespan(adapter)
    assert ls is not None
    assert callable(ls)


# ---------------------------------------------------------------------------
# compose_lifespans
# ---------------------------------------------------------------------------


def test_compose_empty_returns_none():
    result = compose_lifespans()
    assert result is None


def test_compose_none_only_returns_none():
    result = compose_lifespans(None, None)
    assert result is None


def test_compose_single_returns_same():
    pool = AsyncMock()
    ls = make_cleanup_lifespan(pool)
    result = compose_lifespans(ls)
    assert result is ls


def test_compose_two_returns_composed():
    pool = AsyncMock()
    ls1 = make_cleanup_lifespan(pool)
    adapter = MagicMock()
    ls2 = make_health_monitor_lifespan(adapter)
    result = compose_lifespans(ls1, ls2)
    assert result is not None
    assert result is not ls1
    assert result is not ls2


def test_compose_skips_none_entries():
    pool = AsyncMock()
    ls = make_cleanup_lifespan(pool)
    result = compose_lifespans(None, ls, None)
    assert result is ls


async def test_compose_two_lifespans_both_run():
    pool = AsyncMock()
    cleared: list[int] = []
    ls1 = make_cleanup_lifespan(pool, clear_deps_fn=lambda: cleared.append(1))

    adapter = MagicMock()
    # health_check not present — no-op health checks
    ls2 = make_health_monitor_lifespan(adapter, interval=9999.0)

    composed = compose_lifespans(ls1, ls2)
    async with composed(MagicMock()):
        pass  # exit immediately

    pool.close.assert_awaited_once()
    assert len(cleared) == 1


# ---------------------------------------------------------------------------
# Integration: compose_lifespans with |
# ---------------------------------------------------------------------------


def test_pipe_operator_gives_composed_lifespan():
    pool = AsyncMock()
    ls1 = make_cleanup_lifespan(pool)
    adapter = MagicMock()
    ls2 = make_health_monitor_lifespan(adapter)
    composed = ls1 | ls2
    assert composed is not None


async def test_pipe_composed_lifespan_runs_cleanup():
    pool = AsyncMock()
    ls1 = make_cleanup_lifespan(pool)
    adapter = MagicMock()
    ls2 = make_health_monitor_lifespan(adapter, interval=9999.0)
    composed = ls1 | ls2
    async with composed(MagicMock()):
        pass
    pool.close.assert_awaited_once()
