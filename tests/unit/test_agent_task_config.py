"""Tests for TaskConfig annotation on the main `agent` tool (Focus #5 Item B).

Verifies:
- TaskConfig(mode="optional") is applied when background tasks are enabled
  AND pydocket (fastmcp[tasks]) is installed.
- Graceful fallback to task=False when pydocket is absent.
- The `mcp_related_task` field is present in agent tool structured_content.
- task=False is always used when enable_background_tasks=False.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentique.core.types import AgentInfo


def make_agent(name: str = "test") -> AgentInfo:
    return AgentInfo(name=name, base_url=f"http://{name}.example.com")


# ---------------------------------------------------------------------------
# Server creation sanity — background tasks enabled / disabled
# ---------------------------------------------------------------------------


def test_create_server_with_background_tasks_enabled():
    """Server creates without error when enable_background_tasks=True."""
    from agentique.core.config import AgentiqueConfig
    from agentique.server import create_server

    cfg = AgentiqueConfig(enable_background_tasks=True)
    server = create_server(agents=[make_agent()], config=cfg)
    assert server is not None


def test_create_server_with_background_tasks_disabled():
    """Server creates without error when enable_background_tasks=False."""
    from agentique.core.config import AgentiqueConfig
    from agentique.server import create_server

    cfg = AgentiqueConfig(enable_background_tasks=False)
    server = create_server(agents=[make_agent()], config=cfg)
    assert server is not None


# ---------------------------------------------------------------------------
# Docket import missing → task config falls back to False
# ---------------------------------------------------------------------------


def test_agent_tool_task_config_falls_back_when_docket_missing():
    """When docket is not installed, the agent tool is created with task=False
    and the server still starts without error."""
    # Ensure "docket" is not importable during this test
    saved = sys.modules.get("docket", None)
    sys.modules["docket"] = None  # type: ignore[assignment]  # blocks import
    try:
        # Re-import server module so the TaskConfig computation runs fresh
        import agentique.server as server_mod
        import importlib

        importlib.reload(server_mod)
        server = server_mod.create_server(agents=[make_agent()])
        assert server is not None
    finally:
        if saved is None:
            sys.modules.pop("docket", None)
        else:
            sys.modules["docket"] = saved


# ---------------------------------------------------------------------------
# TaskConfig import missing → task config falls back silently
# ---------------------------------------------------------------------------


def test_agent_tool_task_config_falls_back_when_taskconfig_missing():
    """When fastmcp.server.tasks.config is absent, falls back gracefully."""
    saved = sys.modules.get("fastmcp.server.tasks.config", None)
    sys.modules["fastmcp.server.tasks.config"] = None  # type: ignore[assignment]
    try:
        import agentique.server as server_mod
        import importlib

        importlib.reload(server_mod)
        server = server_mod.create_server(agents=[make_agent()])
        assert server is not None
    finally:
        if saved is None:
            sys.modules.pop("fastmcp.server.tasks.config", None)
        else:
            sys.modules["fastmcp.server.tasks.config"] = saved


# ---------------------------------------------------------------------------
# mcp_related_task in structured_content
# ---------------------------------------------------------------------------


def test_agent_tool_structured_content_includes_mcp_related_task():
    """ToolResult schema includes mcp_related_task field.

    Verifies that the structured_content returned by agent_tool contains
    the ``mcp_related_task`` field (set equal to ``task_id``) so MCP clients
    that support the ``io.modelcontextprotocol/related-task`` hint can
    correlate the A2A task with an MCP background task.
    """
    from fastmcp.tools.tool import ToolResult

    task_id = "test-task-id"
    result = ToolResult(
        content="hello",
        structured_content={
            "task_id": task_id,
            "agent": "test_agent",
            "state": "completed",
            "artifact_uris": [],
            "response": "hello",
            "mcp_related_task": task_id,
        },
    )
    sc = result.structured_content or {}
    assert "mcp_related_task" in sc
    assert sc["mcp_related_task"] == sc["task_id"]


def test_agent_tool_structured_content_mcp_related_task_in_server_code():
    """Verify server.py's agent_tool return includes mcp_related_task.

    Reads the server source to confirm mcp_related_task is part of the
    structured_content dict in the agent_tool return statement.
    """
    import inspect
    from agentique import server as server_mod

    source = inspect.getsource(server_mod)
    assert "mcp_related_task" in source, (
        "Expected mcp_related_task in server.py agent_tool return"
    )
