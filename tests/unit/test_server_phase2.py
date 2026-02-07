"""Tests for Phase 2 server factory features.

Tests namespace transforms, structured content (ToolResult), TaskConfig,
and create_server() new parameters.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentique.core.config import AgentiqueConfig
from agentique.core.events import AsyncEventEmitter
from agentique.core.types import AgentInfo, BridgeContext
from agentique.bridge.storage import InMemoryTaskStore
from agentique.testing.mocks import MockAdapter


# ---- create_server parameter tests ----


def test_create_server_accepts_new_params():
    """create_server should accept task_store, transforms, and namespace params."""
    from agentique.server import create_server

    agent = AgentInfo(name="test", base_url="http://localhost:9999")
    store = InMemoryTaskStore()
    adapter = MockAdapter(
        agents={"test": agent},
        responses={"test": "hello"},
    )

    server = create_server(
        agents=[agent],
        adapter=adapter,
        task_store=store,
        config=AgentiqueConfig(
            enable_background_tasks=False,
            prefetch_cards=False,
        ),
    )

    assert server is not None
    assert server.name == "Agentique"


def test_create_server_with_namespace():
    """create_server with namespace should apply Namespace transform."""
    from agentique.server import create_server

    agent = AgentInfo(name="test", base_url="http://localhost:9999")
    adapter = MockAdapter(
        agents={"test": agent},
        responses={"test": "hello"},
    )

    server = create_server(
        agents=[agent],
        adapter=adapter,
        namespace="myns",
        config=AgentiqueConfig(
            enable_background_tasks=False,
            prefetch_cards=False,
        ),
    )

    assert server is not None


def test_create_server_with_custom_transforms():
    """create_server should accept and apply custom transforms."""
    from agentique.server import create_server

    agent = AgentInfo(name="test", base_url="http://localhost:9999")
    adapter = MockAdapter(
        agents={"test": agent},
        responses={"test": "hello"},
    )

    # Use a Visibility transform to test
    try:
        from fastmcp.server.transforms import Visibility

        transform = Visibility(enabled=True, names={"agent"})
        server = create_server(
            agents=[agent],
            adapter=adapter,
            transforms=[transform],
            config=AgentiqueConfig(
                enable_background_tasks=False,
                prefetch_cards=False,
            ),
        )
        assert server is not None
    except ImportError:
        pytest.skip("FastMCP transforms not available")


def test_create_server_with_events():
    """create_server should accept a custom event emitter."""
    from agentique.server import create_server

    agent = AgentInfo(name="test", base_url="http://localhost:9999")
    adapter = MockAdapter(
        agents={"test": agent},
        responses={"test": "hello"},
    )
    emitter = AsyncEventEmitter()
    events_received: list[str] = []
    emitter.on("task.created", lambda **kw: events_received.append("created"))

    server = create_server(
        agents=[agent],
        adapter=adapter,
        events=emitter,
        config=AgentiqueConfig(
            enable_background_tasks=False,
            prefetch_cards=False,
        ),
    )

    assert server is not None


def test_create_server_registers_fastmcp_middleware():
    """create_server should automatically register AgentiqueMiddleware."""
    from agentique.server import create_server
    from agentique.bridge.fastmcp_middleware import AgentiqueMiddleware

    agent = AgentInfo(name="test", base_url="http://localhost:9999")
    adapter = MockAdapter(
        agents={"test": agent},
        responses={"test": "hello"},
    )

    server = create_server(
        agents=[agent],
        adapter=adapter,
        config=AgentiqueConfig(
            enable_background_tasks=False,
            prefetch_cards=False,
        ),
    )

    # Check that middleware was added
    assert any(
        isinstance(mw, AgentiqueMiddleware)
        for mw in server.middleware
    )


# ---- ToolResult structured content tests ----


def test_tool_result_import():
    """ToolResult should be importable from fastmcp."""
    from fastmcp.tools.tool import ToolResult

    result = ToolResult(
        content="test content",
        structured_content={"key": "value"},
    )
    assert result.structured_content == {"key": "value"}


def test_tool_result_with_json_content():
    """ToolResult should handle JSON string content alongside structured data."""
    from fastmcp.tools.tool import ToolResult

    data = {"agents": [{"name": "a1"}, {"name": "a2"}]}
    result = ToolResult(
        content=json.dumps(data, indent=2),
        structured_content=data,
    )

    assert result.structured_content["agents"][0]["name"] == "a1"


# ---- TaskConfig tests ----


def test_task_config_import():
    """TaskConfig should be importable from fastmcp."""
    try:
        from fastmcp.server.tasks.config import TaskConfig

        config = TaskConfig(mode="optional")
        assert config.mode == "optional"
        assert config.supports_tasks() is True

        config2 = TaskConfig(mode="forbidden")
        assert config2.supports_tasks() is False
    except ImportError:
        pytest.skip("TaskConfig not available in this FastMCP version")


def test_task_config_from_bool():
    """TaskConfig.from_bool should convert booleans."""
    try:
        from fastmcp.server.tasks.config import TaskConfig

        config_true = TaskConfig.from_bool(True)
        assert config_true.supports_tasks() is True

        config_false = TaskConfig.from_bool(False)
        assert config_false.supports_tasks() is False
    except ImportError:
        pytest.skip("TaskConfig not available")
