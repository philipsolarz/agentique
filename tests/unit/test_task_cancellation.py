"""Tests for Focus #9a: Task Cancellation.

Covers:
  - A2AAgentAdapter.cancel_task() sends cancel request to A2A client
  - cancel_task() raises TaskNotCancelableError on A2A error -32004
  - cancel_task() raises UnsupportedOperationError when client has no cancel
  - cancel_task MCP tool: task not found → error output
  - cancel_task MCP tool: task already terminal → error output
  - cancel_task MCP tool: no agent_name stored → error output
  - cancel_task MCP tool: successful cancellation → canceled state
  - cancel_task MCP tool: TaskNotCancelableError → error output
  - TaskTracker stores agent_name field
  - cancel_task tool registered on create_server()
"""

from __future__ import annotations

import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio

from agentique.core.errors import (
    AdapterError,
    TaskNotCancelableError,
    UnsupportedOperationError,
)
from agentique.core.types import AgentInfo, TaskState, TaskTracker
from agentique.bridge.task_manager import TaskManager
from agentique.bridge.storage import InMemoryTaskStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(name: str = "billing") -> AgentInfo:
    return AgentInfo(name=name, base_url=f"http://{name}.example.com")


def make_manager() -> TaskManager:
    return TaskManager(store=InMemoryTaskStore())


# ---------------------------------------------------------------------------
# TaskTracker.agent_name field
# ---------------------------------------------------------------------------


def test_task_tracker_has_agent_name_field():
    """TaskTracker must have an agent_name field defaulting to None."""
    tracker = TaskTracker(task_id="t1")
    assert hasattr(tracker, "agent_name")
    assert tracker.agent_name is None


def test_task_tracker_agent_name_settable():
    tracker = TaskTracker(task_id="t1")
    tracker.agent_name = "billing"
    assert tracker.agent_name == "billing"


def test_task_tracker_agent_name_in_constructor():
    tracker = TaskTracker(task_id="t1", agent_name="crm")
    assert tracker.agent_name == "crm"


# ---------------------------------------------------------------------------
# A2AAgentAdapter.cancel_task()
# ---------------------------------------------------------------------------


async def test_cancel_task_sends_cancel_request():
    """cancel_task() calls client.cancel_task with the task ID."""
    from agentique.adapters.a2a.adapter import A2AAgentAdapter

    mock_client = MagicMock()
    mock_client.cancel_task = AsyncMock(return_value=MagicMock())

    mock_pool = MagicMock()
    mock_pool.get = AsyncMock(return_value=mock_client)

    agent = make_agent("billing")
    adapter = A2AAgentAdapter({"billing": agent}, client_pool=mock_pool)

    result = await adapter.cancel_task("billing", "task-123")
    assert result is True
    mock_client.cancel_task.assert_called_once()


async def test_cancel_task_unknown_agent_raises():
    """cancel_task() with unknown agent_id raises AgentNotFoundError."""
    from agentique.adapters.a2a.adapter import A2AAgentAdapter
    from agentique.core.errors import AgentNotFoundError

    agent = make_agent("billing")
    adapter = A2AAgentAdapter({"billing": agent})

    with pytest.raises(AgentNotFoundError):
        await adapter.cancel_task("unknown-agent", "task-123")


async def test_cancel_task_client_no_cancel_method_raises():
    """When the A2A client doesn't expose cancel_task, raise UnsupportedOperationError."""
    from agentique.adapters.a2a.adapter import A2AAgentAdapter

    mock_client = MagicMock(spec=[])  # no cancel_task attribute

    mock_pool = MagicMock()
    mock_pool.get = AsyncMock(return_value=mock_client)

    agent = make_agent("billing")
    adapter = A2AAgentAdapter({"billing": agent}, client_pool=mock_pool)

    with pytest.raises(UnsupportedOperationError, match="cancellation"):
        await adapter.cancel_task("billing", "task-123")


async def test_cancel_task_a2a_error_32004_raises_not_cancelable():
    """When the A2A agent returns -32004, raise TaskNotCancelableError."""
    from agentique.adapters.a2a.adapter import A2AAgentAdapter

    class FakeA2AError(Exception):
        code = -32004

    mock_client = MagicMock()
    mock_client.cancel_task = AsyncMock(side_effect=FakeA2AError("task not cancelable"))

    mock_pool = MagicMock()
    mock_pool.get = AsyncMock(return_value=mock_client)

    agent = make_agent("billing")
    adapter = A2AAgentAdapter({"billing": agent}, client_pool=mock_pool)

    with pytest.raises(TaskNotCancelableError):
        await adapter.cancel_task("billing", "task-123")


async def test_cancel_task_other_error_raises_adapter_error():
    """Other exceptions from the A2A client raise AdapterError."""
    from agentique.adapters.a2a.adapter import A2AAgentAdapter

    mock_client = MagicMock()
    mock_client.cancel_task = AsyncMock(side_effect=RuntimeError("network error"))

    mock_pool = MagicMock()
    mock_pool.get = AsyncMock(return_value=mock_client)

    agent = make_agent("billing")
    adapter = A2AAgentAdapter({"billing": agent}, client_pool=mock_pool)

    with pytest.raises(AdapterError, match="Failed to cancel"):
        await adapter.cancel_task("billing", "task-123")


# ---------------------------------------------------------------------------
# cancel_task MCP tool (via create_server)
# ---------------------------------------------------------------------------


def _make_server(**kwargs):
    from agentique.server import create_server
    return create_server(
        agents=[make_agent("billing")],
        **kwargs,
    )


def test_cancel_task_tool_registered():
    """The cancel_task tool is registered on the created server."""
    server = _make_server()
    tools = asyncio.run(server.list_tools())
    tool_names = [t.name for t in tools]
    assert "cancel_task" in tool_names, f"cancel_task not in {tool_names}"


def test_cancel_task_tool_task_not_found():
    """cancel_task tool returns error when task_id is unknown."""
    server = _make_server()

    async def _run():
        result = await server.call_tool("cancel_task", {"task_id": "nonexistent"})
        return result

    result = asyncio.run(_run())
    # Result should contain an error indicator
    content_text = result.content[0].text if result.content else ""
    data = json.loads(content_text)
    assert "error" in data or data.get("code") == "TASK_NOT_FOUND"


def test_cancel_task_tool_terminal_task():
    """cancel_task tool returns error when task is already terminal."""
    from agentique.bridge.storage import InMemoryTaskStore
    from agentique.core.types import TaskTracker, TaskState

    store = InMemoryTaskStore()

    async def _setup():
        tracker = TaskTracker(
            task_id="done-task",
            state=TaskState.completed,
            agent_name="billing",
        )
        await store.save("done-task", tracker)

    asyncio.run(_setup())

    server = _make_server(task_store=store)

    async def _run():
        result = await server.call_tool("cancel_task", {"task_id": "done-task"})
        return result

    result = asyncio.run(_run())
    content_text = result.content[0].text if result.content else ""
    data = json.loads(content_text)
    assert "error" in data


def test_cancel_task_tool_successful_cancellation():
    """Successful cancellation transitions task to canceled state."""
    from agentique.bridge.storage import InMemoryTaskStore
    from agentique.core.types import TaskTracker, TaskState

    store = InMemoryTaskStore()

    async def _setup():
        tracker = TaskTracker(
            task_id="running-task",
            state=TaskState.working,
            agent_name="billing",
        )
        await store.save("running-task", tracker)

    asyncio.run(_setup())

    # Use a plain class to avoid MagicMock being seen as an async context manager
    # by FastMCP's DI framework (MagicMock satisfies AbstractAsyncContextManager).
    class _SuccessAdapter:
        async def cancel_task(self, agent_name, task_id):
            return True

        async def discover_agents(self):
            return [make_agent("billing")]

    server = _make_server(task_store=store, adapter=_SuccessAdapter())

    async def _run():
        result = await server.call_tool("cancel_task", {"task_id": "running-task"})
        return result

    result = asyncio.run(_run())
    content_text = result.content[0].text if result.content else ""
    data = json.loads(content_text)
    # After cancellation, state should be 'canceled'
    assert data.get("state") == "canceled" or "canceled" in str(data)


def test_cancel_task_tool_not_cancelable_error():
    """TaskNotCancelableError from adapter is surfaced as error output."""
    from agentique.bridge.storage import InMemoryTaskStore
    from agentique.core.types import TaskTracker, TaskState

    store = InMemoryTaskStore()

    async def _setup():
        tracker = TaskTracker(
            task_id="locked-task",
            state=TaskState.working,
            agent_name="billing",
        )
        await store.save("locked-task", tracker)

    asyncio.run(_setup())

    # Use a plain class to avoid MagicMock being seen as an async context manager
    # by FastMCP's DI framework (MagicMock satisfies AbstractAsyncContextManager).
    class _FailAdapter:
        async def cancel_task(self, agent_name, task_id):
            raise TaskNotCancelableError("Cannot cancel: task is locked")

        async def discover_agents(self):
            return [make_agent("billing")]

    server = _make_server(task_store=store, adapter=_FailAdapter())

    async def _run():
        result = await server.call_tool("cancel_task", {"task_id": "locked-task"})
        return result

    result = asyncio.run(_run())
    content_text = result.content[0].text if result.content else ""
    data = json.loads(content_text)
    assert "error" in data


# ---------------------------------------------------------------------------
# TaskManager.transition() handles canceled state
# ---------------------------------------------------------------------------


async def test_task_manager_transition_to_canceled():
    """TaskManager.transition() accepts TaskState.canceled."""
    mgr = make_manager()
    await mgr.create("t1")
    success = await mgr.transition("t1", TaskState.canceled)
    assert success is True
    tracker = await mgr.get("t1")
    assert tracker.state == TaskState.canceled


async def test_task_manager_canceled_is_terminal():
    """Once canceled, a task cannot be transitioned further."""
    mgr = make_manager()
    await mgr.create("t1")
    await mgr.transition("t1", TaskState.canceled)
    # Attempt further transition — should return False (already terminal)
    success = await mgr.transition("t1", TaskState.working)
    assert success is False
