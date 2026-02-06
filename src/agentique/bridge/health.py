"""Adapter health monitoring with periodic checks.

Monitors registered adapters for health, automatically marking them
as healthy/unhealthy, emitting events, and optionally removing failed
adapters from the router.

Usage::

    from agentique.bridge.health import HealthMonitor

    monitor = HealthMonitor(
        adapter=adapter,
        emitter=emitter,
        router=router,
        check_interval=30.0,
    )
    await monitor.start()
    # ... later ...
    await monitor.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agentique.core.events import AsyncEventEmitter

logger = logging.getLogger(__name__)


@dataclass
class AgentHealth:
    """Health status for a single agent."""

    agent_id: str
    healthy: bool = True
    last_check: float = 0.0
    last_success: float = 0.0
    consecutive_failures: int = 0
    last_error: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "healthy": self.healthy,
            "last_check": self.last_check,
            "last_success": self.last_success,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
        }


class HealthMonitor:
    """Periodic health checker for agent adapters.

    Probes each registered agent at a configurable interval and emits
    health events through the ``AsyncEventEmitter``.

    Events emitted:
        - ``adapter.healthy`` — agent transitioned to healthy
        - ``adapter.unhealthy`` — agent transitioned to unhealthy
        - ``adapter.health_check`` — every check result (for logging)

    Args:
        adapter: The agent adapter to monitor.
        emitter: Event emitter for health events.
        router: Optional router for automatic agent removal/re-add.
        check_interval: Seconds between health checks. Default 30.
        failure_threshold: Consecutive failures before marking unhealthy.
            Default 3.
        timeout: Timeout in seconds for each health check. Default 10.
    """

    def __init__(
        self,
        adapter: Any,
        emitter: AsyncEventEmitter,
        router: Any | None = None,
        check_interval: float = 30.0,
        failure_threshold: int = 3,
        timeout: float = 10.0,
    ) -> None:
        self._adapter = adapter
        self._emitter = emitter
        self._router = router
        self._check_interval = check_interval
        self._failure_threshold = failure_threshold
        self._timeout = timeout
        self._health: dict[str, AgentHealth] = {}
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._removed_agents: dict[str, Any] = {}  # agent_id -> AgentInfo

    async def start(self) -> None:
        """Start periodic health monitoring."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started (interval=%.1fs)", self._check_interval)

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Health monitor stopped")

    async def check_agent(self, agent_id: str) -> AgentHealth:
        """Run a health check for a single agent.

        The check tries to call ``get_agent_card()`` on the adapter
        as a lightweight probe. If the adapter doesn't support
        ``get_agent_card``, it falls back to ``discover_agents()``.

        Args:
            agent_id: The agent to check.

        Returns:
            Updated ``AgentHealth`` record.
        """
        if agent_id not in self._health:
            self._health[agent_id] = AgentHealth(agent_id=agent_id)

        health = self._health[agent_id]
        was_healthy = health.healthy
        start = time.monotonic()

        try:
            # Try get_agent_card as a lightweight probe
            get_card = getattr(self._adapter, "get_agent_card", None)
            if callable(get_card):
                await asyncio.wait_for(
                    get_card(agent_id),
                    timeout=self._timeout,
                )
            else:
                # Fall back to discover_agents
                await asyncio.wait_for(
                    self._adapter.discover_agents(),
                    timeout=self._timeout,
                )

            elapsed = (time.monotonic() - start) * 1000
            health.healthy = True
            health.last_check = time.time()
            health.last_success = time.time()
            health.consecutive_failures = 0
            health.last_error = None
            health.latency_ms = round(elapsed, 2)

            # Transition: unhealthy -> healthy
            if not was_healthy:
                await self._emitter.emit(
                    "adapter.healthy",
                    agent_id=agent_id,
                    health=health,
                )
                await self._try_re_register(agent_id)

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            health.last_check = time.time()
            health.consecutive_failures += 1
            health.last_error = str(exc)
            health.latency_ms = round(elapsed, 2)

            if health.consecutive_failures >= self._failure_threshold:
                health.healthy = False

                # Transition: healthy -> unhealthy
                if was_healthy:
                    await self._emitter.emit(
                        "adapter.unhealthy",
                        agent_id=agent_id,
                        health=health,
                        error=str(exc),
                    )
                    await self._try_remove(agent_id)

        # Always emit health check event
        await self._emitter.emit(
            "adapter.health_check",
            agent_id=agent_id,
            health=health,
        )

        return health

    async def check_all(self) -> dict[str, AgentHealth]:
        """Run health checks for all known agents.

        Returns:
            Dict of agent_id -> AgentHealth.
        """
        # Discover agents if we haven't seen any yet
        if not self._health:
            try:
                agents = await self._adapter.discover_agents()
                for agent in agents:
                    self._health[agent.name] = AgentHealth(agent_id=agent.name)
            except Exception:
                logger.debug("Failed to discover agents for health check", exc_info=True)

        # Check all known agents concurrently
        tasks = [
            self.check_agent(agent_id)
            for agent_id in list(self._health.keys())
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return dict(self._health)

    def get_health(self, agent_id: str) -> AgentHealth | None:
        """Get the current health status for an agent."""
        return self._health.get(agent_id)

    def get_all_health(self) -> dict[str, AgentHealth]:
        """Get all agent health records."""
        return dict(self._health)

    def get_healthy_agents(self) -> list[str]:
        """Return IDs of all currently healthy agents."""
        return [
            agent_id for agent_id, health in self._health.items()
            if health.healthy
        ]

    def get_unhealthy_agents(self) -> list[str]:
        """Return IDs of all currently unhealthy agents."""
        return [
            agent_id for agent_id, health in self._health.items()
            if not health.healthy
        ]

    def register_tools(self, mcp: Any) -> None:
        """Register health monitoring tools on the FastMCP server.

        Args:
            mcp: The FastMCP server instance.
        """
        import json
        monitor = self

        @mcp.tool(name="agent_health")
        async def agent_health_tool(agent_id: str | None = None) -> str:
            """Check agent health status.

            Args:
                agent_id: Specific agent to check, or omit for all
            """
            if agent_id:
                health = await monitor.check_agent(agent_id)
                return json.dumps(health.to_dict(), indent=2)
            else:
                all_health = await monitor.check_all()
                return json.dumps(
                    {k: v.to_dict() for k, v in all_health.items()},
                    indent=2,
                )

    # ---- Internal ----

    async def _monitor_loop(self) -> None:
        """Background loop that periodically checks all agents."""
        while self._running:
            try:
                await self.check_all()
            except Exception:
                logger.debug("Health check cycle failed", exc_info=True)
            await asyncio.sleep(self._check_interval)

    async def _try_remove(self, agent_id: str) -> None:
        """Try to remove an unhealthy agent from the router."""
        if self._router is None:
            return
        remove = getattr(self._router, "remove_agent", None)
        if callable(remove):
            try:
                # Store agent info before removal for re-registration
                agents = self._router.list_agents()
                for agent in agents:
                    if agent.name == agent_id:
                        self._removed_agents[agent_id] = agent
                        break
                remove(agent_id)
                logger.info("Removed unhealthy agent '%s' from router", agent_id)
            except Exception:
                logger.debug(
                    "Failed to remove agent '%s' from router",
                    agent_id, exc_info=True,
                )

    async def _try_re_register(self, agent_id: str) -> None:
        """Try to re-register a recovered agent in the router."""
        if self._router is None:
            return
        agent_info = self._removed_agents.pop(agent_id, None)
        if agent_info is None:
            return
        add = getattr(self._router, "add_agent", None)
        if callable(add):
            try:
                add(agent_info)
                logger.info("Re-registered recovered agent '%s'", agent_id)
            except Exception:
                logger.debug(
                    "Failed to re-register agent '%s'",
                    agent_id, exc_info=True,
                )
