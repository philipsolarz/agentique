"""Composable lifespan factories for Agentique's resource lifecycle.

Uses FastMCP 3.0's ``@lifespan`` decorator and the ``|`` pipe operator to
compose independent resource lifecycles into a single server lifespan.  Each
factory produces a ``Lifespan`` instance that can be composed with ``|``:

    from agentique.bridge.lifespans import (
        make_cleanup_lifespan,
        make_health_monitor_lifespan,
    )

    pool_lifespan   = make_cleanup_lifespan(pool, clear_deps_fn=deps.clear)
    health_lifespan = make_health_monitor_lifespan(adapter, interval=30.0)

    server = FastMCP("name", lifespan=pool_lifespan | health_lifespan)

**Lifespan ordering**: lifespans enter left-to-right and exit right-to-left
(LIFO), so cleanup_lifespan should come first.

**Feature-flag-driven participation**: optional components (health monitor,
webhook receiver, metrics exporter) contribute to the lifespan chain only
when configured.  This keeps the default path lightweight.

**Backward compatibility**: ``configure_deps()`` is still called in
``create_server()`` for immediate availability in tests and one-shot
scripts that don't run the full server lifespan.  The lifespans here add
*cleanup* semantics, not initialization — the A2A client pool is closed on
server shutdown without any changes to the startup path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

try:
    import anyio as _anyio
except ImportError:
    _anyio = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def make_cleanup_lifespan(
    pool: Any | None,
    *,
    clear_deps_fn: Callable[[], None] | None = None,
) -> Any:
    """Create a lifespan that closes the A2A client pool on shutdown.

    This is the primary Agentique lifespan — it ensures the ``A2AClientPool``
    (and its underlying ``httpx.AsyncClient`` instances) are properly closed
    when the FastMCP server shuts down.

    Optionally calls *clear_deps_fn* (typically ``agentique.bridge.dependencies.clear``)
    to reset the module-level dependency registry on shutdown, which is useful
    in test environments where multiple servers are created in a single process.

    Args:
        pool: The ``A2AClientPool`` instance to close on server shutdown.
        clear_deps_fn: Optional callable invoked after pool close.
            Defaults to ``None`` (no registry reset).

    Returns:
        A FastMCP ``Lifespan`` instance.

    Example::

        from agentique.bridge.lifespans import make_cleanup_lifespan
        from agentique.bridge.dependencies import clear as clear_deps
        from fastmcp import FastMCP

        pool = A2AClientPool()
        server = FastMCP("bridge", lifespan=make_cleanup_lifespan(pool, clear_deps_fn=clear_deps))
    """
    from fastmcp.server.lifespan import lifespan as _lifespan

    async def _cleanup(server: Any) -> Any:
        try:
            yield {}
        finally:
            if pool is not None:
                try:
                    await pool.close()
                    logger.debug("A2AClientPool closed")
                except Exception as exc:
                    logger.warning("Error closing A2AClientPool: %s", exc)
            if clear_deps_fn is not None:
                try:
                    clear_deps_fn()
                except Exception as exc:
                    logger.warning("Error clearing dependency registry: %s", exc)

    return _lifespan(_cleanup)


def make_health_monitor_lifespan(
    adapter: Any,
    *,
    interval: float = 30.0,
    on_degraded: Callable[[str, Exception], None] | None = None,
) -> Any:
    """Create a lifespan that runs a periodic health check loop.

    Starts a background ``asyncio.Task`` that calls
    ``adapter.health_check()`` (if available) every *interval* seconds.
    The task is cancelled cleanly on server shutdown.

    The health monitor participates in the lifespan chain via ``|``:

        server_lifespan = cleanup | health_monitor

    This means the monitor starts after the pool is ready and stops before
    the pool closes (LIFO order).

    Args:
        adapter: The agent adapter exposing an optional ``health_check()``
            coroutine.  When the method is absent the loop logs periodically
            but performs no checks.
        interval: Seconds between health checks.  Defaults to ``30.0``.
        on_degraded: Optional callback invoked when a health check fails.
            Receives the agent name (or ``"adapter"``) and the exception.

    Returns:
        A FastMCP ``Lifespan`` instance.

    Example::

        from agentique.bridge.lifespans import (
            make_cleanup_lifespan,
            make_health_monitor_lifespan,
        )

        lifespan = (
            make_cleanup_lifespan(pool)
            | make_health_monitor_lifespan(adapter, interval=15.0)
        )
        server = FastMCP("bridge", lifespan=lifespan)
    """
    from fastmcp.server.lifespan import lifespan as _lifespan

    async def _health_loop() -> None:
        """Background task: periodically check adapter health."""
        while True:
            # Use anyio.sleep when available (works under both asyncio + trio),
            # fall back to asyncio.sleep for environments without anyio.
            if _anyio is not None:
                await _anyio.sleep(interval)
            else:
                await asyncio.sleep(interval)
            try:
                check_fn = getattr(adapter, "health_check", None)
                if callable(check_fn):
                    result = check_fn()
                    if asyncio.iscoroutine(result):
                        await result
                    logger.debug("Health check passed")
                else:
                    logger.debug("Adapter has no health_check(); skipping")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Health check failed: %s", exc)
                if on_degraded is not None:
                    try:
                        on_degraded("adapter", exc)
                    except Exception:
                        pass

    async def _monitor(server: Any) -> Any:
        logger.debug("Health monitor started (interval=%.1fs)", interval)
        if _anyio is not None:
            # anyio task group works under both asyncio and trio
            async with _anyio.create_task_group() as tg:
                tg.start_soon(_health_loop)
                yield {"health_monitor_task": tg}
                tg.cancel_scope.cancel()
        else:
            # Fallback: asyncio only
            task = asyncio.create_task(_health_loop(), name="agentique-health-monitor")
            try:
                yield {"health_monitor_task": task}
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.debug("Health monitor stopped")

    return _lifespan(_monitor)


def make_artifact_cleanup_lifespan(
    task_manager: Any,
    *,
    ttl: float,
    interval: float | None = None,
) -> Any:
    """Create a lifespan that periodically evicts expired artifacts.

    Runs a background task every *interval* seconds (defaults to ``ttl / 2``
    with a minimum of 60 seconds) that calls
    ``task_manager.evict_all_expired_artifacts()``.

    Args:
        task_manager: The ``TaskManager`` instance whose artifacts to clean up.
        ttl: Artifact time-to-live in seconds. Artifacts older than this
            are evicted.
        interval: Cleanup interval in seconds. Defaults to ``max(ttl / 2, 60)``.

    Returns:
        A FastMCP ``Lifespan`` instance.
    """
    from fastmcp.server.lifespan import lifespan as _lifespan

    _interval = interval if interval is not None else max(ttl / 2, 60.0)

    async def _cleanup_loop() -> None:
        while True:
            if _anyio is not None:
                await _anyio.sleep(_interval)
            else:
                await asyncio.sleep(_interval)
            try:
                evicted = task_manager.evict_all_expired_artifacts()
                if evicted:
                    logger.debug("Artifact cleanup: evicted %d expired artifacts", evicted)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Artifact cleanup failed: %s", exc)

    async def _artifact_cleanup(server: Any) -> Any:
        logger.debug(
            "Artifact cleanup started (ttl=%.1fs, interval=%.1fs)", ttl, _interval
        )
        if _anyio is not None:
            async with _anyio.create_task_group() as tg:
                tg.start_soon(_cleanup_loop)
                yield {}
                tg.cancel_scope.cancel()
        else:
            task = asyncio.create_task(
                _cleanup_loop(), name="agentique-artifact-cleanup"
            )
            try:
                yield {}
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.debug("Artifact cleanup stopped")

    return _lifespan(_artifact_cleanup)


def compose_lifespans(*lifespans: Any) -> Any | None:
    """Compose multiple lifespan instances with ``|``.

    Returns ``None`` when the list is empty (FastMCP interprets ``None``
    as "no lifespan").

    Args:
        *lifespans: ``Lifespan`` instances to compose in left-to-right order.
            ``None`` entries are silently skipped.

    Returns:
        A single composed ``Lifespan`` or ``None``.

    Example::

        lifespan = compose_lifespans(
            make_cleanup_lifespan(pool),
            make_health_monitor_lifespan(adapter) if config.health_check else None,
        )
        server = FastMCP("bridge", lifespan=lifespan)
    """
    active = [ls for ls in lifespans if ls is not None]
    if not active:
        return None
    result = active[0]
    for ls in active[1:]:
        result = result | ls
    return result
