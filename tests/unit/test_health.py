"""Tests for the adapter health monitoring module."""

from __future__ import annotations

import asyncio
import pytest

from agentique.core.events import AsyncEventEmitter
from agentique.core.types import AgentInfo
from agentique.bridge.health import AgentHealth, HealthMonitor


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class MockAdapter:
    """Adapter with controllable health behavior."""

    def __init__(self, healthy: bool = True):
        self._healthy = healthy
        self._agents = [
            AgentInfo(name="agent-a", base_url="http://localhost:9001"),
            AgentInfo(name="agent-b", base_url="http://localhost:9002"),
        ]
        self.check_count = 0

    async def discover_agents(self) -> list[AgentInfo]:
        return list(self._agents)

    async def get_agent_card(self, agent_id: str) -> dict | None:
        self.check_count += 1
        if not self._healthy:
            raise ConnectionError(f"Agent {agent_id} unreachable")
        return {"name": agent_id, "version": "1.0"}

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy


class MockRouter:
    """Router with agent add/remove support."""

    def __init__(self):
        self._agents: dict[str, AgentInfo] = {}

    def add_agent(self, agent: AgentInfo) -> None:
        self._agents[agent.name] = agent

    def remove_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def list_agents(self) -> list[AgentInfo]:
        return list(self._agents.values())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def emitter():
    return AsyncEventEmitter()


@pytest.fixture
def adapter():
    return MockAdapter()


@pytest.fixture
def router():
    r = MockRouter()
    r.add_agent(AgentInfo(name="agent-a", base_url="http://localhost:9001"))
    r.add_agent(AgentInfo(name="agent-b", base_url="http://localhost:9002"))
    return r


@pytest.fixture
def monitor(adapter, emitter, router):
    return HealthMonitor(
        adapter=adapter,
        emitter=emitter,
        router=router,
        check_interval=0.1,
        failure_threshold=2,
        timeout=5.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentHealth:
    def test_defaults(self):
        health = AgentHealth(agent_id="test")
        assert health.healthy is True
        assert health.consecutive_failures == 0

    def test_to_dict(self):
        health = AgentHealth(agent_id="test", healthy=False, consecutive_failures=3)
        d = health.to_dict()
        assert d["agent_id"] == "test"
        assert d["healthy"] is False
        assert d["consecutive_failures"] == 3


class TestHealthMonitor:
    @pytest.mark.anyio
    async def test_check_healthy_agent(self, monitor, adapter):
        health = await monitor.check_agent("agent-a")
        assert health.healthy is True
        assert health.latency_ms is not None
        assert adapter.check_count == 1

    @pytest.mark.anyio
    async def test_check_unhealthy_agent(self, monitor, adapter):
        adapter.set_healthy(False)

        # First failure — not yet unhealthy (threshold=2)
        health = await monitor.check_agent("agent-a")
        assert health.healthy is True
        assert health.consecutive_failures == 1

        # Second failure — now unhealthy
        health = await monitor.check_agent("agent-a")
        assert health.healthy is False
        assert health.consecutive_failures == 2

    @pytest.mark.anyio
    async def test_healthy_event_emitted(self, monitor, adapter, emitter):
        events = []
        emitter.on("adapter.healthy", lambda **kw: events.append(kw))

        # Make unhealthy first
        adapter.set_healthy(False)
        await monitor.check_agent("agent-a")
        await monitor.check_agent("agent-a")
        assert monitor.get_health("agent-a").healthy is False

        # Recover
        adapter.set_healthy(True)
        await monitor.check_agent("agent-a")
        assert len(events) == 1
        assert events[0]["agent_id"] == "agent-a"

    @pytest.mark.anyio
    async def test_unhealthy_event_emitted(self, monitor, adapter, emitter):
        events = []
        emitter.on("adapter.unhealthy", lambda **kw: events.append(kw))

        adapter.set_healthy(False)
        await monitor.check_agent("agent-a")
        await monitor.check_agent("agent-a")

        assert len(events) == 1
        assert events[0]["agent_id"] == "agent-a"
        assert "error" in events[0]

    @pytest.mark.anyio
    async def test_check_all(self, monitor, adapter):
        # Need to seed health dict
        monitor._health["agent-a"] = AgentHealth(agent_id="agent-a")
        monitor._health["agent-b"] = AgentHealth(agent_id="agent-b")

        results = await monitor.check_all()
        assert len(results) == 2
        assert results["agent-a"].healthy is True
        assert results["agent-b"].healthy is True

    @pytest.mark.anyio
    async def test_check_all_discovers_agents(self, monitor):
        # Should discover agents if health dict is empty
        results = await monitor.check_all()
        assert "agent-a" in results
        assert "agent-b" in results

    @pytest.mark.anyio
    async def test_get_healthy_unhealthy(self, monitor, adapter):
        monitor._health["agent-a"] = AgentHealth(agent_id="agent-a", healthy=True)
        monitor._health["agent-b"] = AgentHealth(agent_id="agent-b", healthy=False)

        assert monitor.get_healthy_agents() == ["agent-a"]
        assert monitor.get_unhealthy_agents() == ["agent-b"]

    @pytest.mark.anyio
    async def test_auto_remove_from_router(self, monitor, adapter, router):
        adapter.set_healthy(False)
        await monitor.check_agent("agent-a")
        await monitor.check_agent("agent-a")  # hits threshold

        # Agent should be removed from router
        agent_names = [a.name for a in router.list_agents()]
        assert "agent-a" not in agent_names

    @pytest.mark.anyio
    async def test_auto_re_register_in_router(self, monitor, adapter, router):
        # Remove agent via health failure
        adapter.set_healthy(False)
        await monitor.check_agent("agent-a")
        await monitor.check_agent("agent-a")

        # Recover
        adapter.set_healthy(True)
        await monitor.check_agent("agent-a")

        agent_names = [a.name for a in router.list_agents()]
        assert "agent-a" in agent_names

    @pytest.mark.anyio
    async def test_start_stop(self, monitor):
        await monitor.start()
        assert monitor._running is True
        # Let it run briefly
        await asyncio.sleep(0.05)
        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.anyio
    async def test_health_check_event_always_emitted(self, monitor, emitter):
        events = []
        emitter.on("adapter.health_check", lambda **kw: events.append(kw))

        await monitor.check_agent("agent-a")
        assert len(events) == 1
