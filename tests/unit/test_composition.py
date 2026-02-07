"""Tests for multi-bridge composition via mount_bridge()."""

from __future__ import annotations

import pytest

from agentique.core.config import AgentiqueConfig
from agentique.core.types import AgentInfo
from agentique.server import create_server, mount_bridge
from agentique.testing.mocks import MockAdapter

from fastmcp import FastMCP


# ---- Tests ----


def test_mount_bridge_creates_child_server():
    """mount_bridge should create and mount a child server."""
    parent = FastMCP("Parent")

    agents = [AgentInfo(name="agent1", base_url="http://localhost:9001")]
    adapter = MockAdapter(
        agents={"agent1": agents[0]},
        responses={"agent1": "hello"},
    )

    child = mount_bridge(
        parent,
        agents=agents,
        adapter=adapter,
        namespace="a2a",
    )

    assert child is not None
    assert isinstance(child, FastMCP)


def test_mount_bridge_multiple_namespaces():
    """mount_bridge should support mounting multiple bridges."""
    parent = FastMCP("Parent")

    a2a_agents = [AgentInfo(name="calc", base_url="http://localhost:9001")]
    http_agents = [AgentInfo(name="writer", base_url="http://localhost:9002")]

    a2a_adapter = MockAdapter(
        agents={"calc": a2a_agents[0]},
        responses={"calc": "42"},
    )
    http_adapter = MockAdapter(
        agents={"writer": http_agents[0]},
        responses={"writer": "once upon a time"},
    )

    child_a2a = mount_bridge(
        parent, agents=a2a_agents, adapter=a2a_adapter, namespace="a2a",
    )
    child_http = mount_bridge(
        parent, agents=http_agents, adapter=http_adapter, namespace="http",
    )

    assert child_a2a is not None
    assert child_http is not None
    assert child_a2a is not child_http


def test_mount_bridge_with_custom_config():
    """mount_bridge should accept a custom config."""
    parent = FastMCP("Parent")

    agents = [AgentInfo(name="agent1", base_url="http://localhost:9001")]
    adapter = MockAdapter(
        agents={"agent1": agents[0]},
        responses={"agent1": "hello"},
    )

    custom_config = AgentiqueConfig(
        name="Custom Bridge",
        enable_background_tasks=False,
        prefetch_cards=False,
    )

    child = mount_bridge(
        parent,
        agents=agents,
        adapter=adapter,
        namespace="custom",
        config=custom_config,
    )

    assert child is not None


def test_mount_bridge_returns_child():
    """mount_bridge should return the child server for further configuration."""
    parent = FastMCP("Parent")

    agents = [AgentInfo(name="a1", base_url="http://localhost:9001")]
    adapter = MockAdapter(
        agents={"a1": agents[0]},
        responses={"a1": "hi"},
    )

    child = mount_bridge(
        parent, agents=agents, adapter=adapter, namespace="ns",
    )

    # Child should be usable (e.g., adding more tools)
    @child.tool(name="extra")
    def extra_tool() -> str:
        return "extra"

    assert child is not None


def test_create_server_still_works_standalone():
    """create_server should still work without mount_bridge."""
    agents = [AgentInfo(name="solo", base_url="http://localhost:9001")]
    adapter = MockAdapter(
        agents={"solo": agents[0]},
        responses={"solo": "result"},
    )

    server = create_server(
        agents=agents,
        adapter=adapter,
        config=AgentiqueConfig(
            enable_background_tasks=False,
            prefetch_cards=False,
        ),
    )

    assert server is not None
    assert server.name == "Agentique"
