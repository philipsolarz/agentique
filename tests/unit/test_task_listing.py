"""Tests for TaskManager.list_tasks() and the a2a://tasks resource.

Covers:
  - list_tasks() delegates to the store's list_ids()
  - list_tasks() returns correct IDs after tasks are created
  - a2a://tasks resource is registered on create_server()
  - a2a://tasks returns valid TaskListOutput JSON
  - edge cases: empty store, multiple tasks, context_id propagation
"""

from __future__ import annotations

import json

import pytest

from agentique.bridge.storage import InMemoryTaskStore
from agentique.bridge.task_manager import TaskManager
from agentique.bridge.output_models import TaskListOutput
from agentique.core.types import AgentInfo

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager() -> TaskManager:
    return TaskManager(store=InMemoryTaskStore())


# ---------------------------------------------------------------------------
# TaskManager.list_tasks()
# ---------------------------------------------------------------------------


async def test_list_tasks_empty_store():
    mgr = make_manager()
    ids = await mgr.list_tasks()
    assert ids == []


async def test_list_tasks_after_create():
    mgr = make_manager()
    await mgr.create("task-1")
    ids = await mgr.list_tasks()
    assert "task-1" in ids


async def test_list_tasks_multiple_tasks():
    mgr = make_manager()
    for i in range(5):
        await mgr.create(f"task-{i}")
    ids = await mgr.list_tasks()
    assert len(ids) == 5
    for i in range(5):
        assert f"task-{i}" in ids


async def test_list_tasks_returns_list_of_strings():
    mgr = make_manager()
    await mgr.create("t1")
    ids = await mgr.list_tasks()
    assert all(isinstance(i, str) for i in ids)


async def test_task_manager_list_tasks_delegates_to_store():
    """list_tasks() result must match store.list_ids() directly."""
    store = InMemoryTaskStore()
    mgr = TaskManager(store=store)
    await mgr.create("abc")
    await mgr.create("def")
    store_ids = await store.list_ids()
    mgr_ids = await mgr.list_tasks()
    assert set(mgr_ids) == set(store_ids)


async def test_list_tasks_with_context_id():
    mgr = make_manager()
    tracker = await mgr.create("t1", context_id="ctx-1")
    assert tracker.context_id == "ctx-1"
    ids = await mgr.list_tasks()
    assert "t1" in ids


# ---------------------------------------------------------------------------
# a2a://tasks resource registration
# ---------------------------------------------------------------------------


def _make_server(**kwargs):
    from agentique.server import create_server
    return create_server(
        agents=[AgentInfo(name="test", base_url="http://test.example.com")],
        **kwargs,
    )


def _read_resource_sync(mcp, uri: str) -> str:
    """Read a resource synchronously via asyncio.run()."""
    import asyncio

    async def _read():
        result = await mcp.read_resource(uri)
        return result.contents[0].content

    return asyncio.run(_read())


# ---------------------------------------------------------------------------
# a2a://tasks resource registration
# ---------------------------------------------------------------------------


def test_tasks_resource_uri_registered():
    import asyncio

    mcp = _make_server()

    async def _check():
        resources = await mcp.list_resources()
        return [str(r.uri) for r in resources]

    uris = asyncio.run(_check())
    assert any("tasks" in u for u in uris), f"a2a://tasks not found in {uris}"


def test_tasks_resource_is_static_not_template():
    """a2a://tasks must resolve and return a response (not a template error)."""
    content = _read_resource_sync(_make_server(), "a2a://tasks")
    parsed = json.loads(content)
    # Static resources return a dict; templates would 404
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# a2a://tasks resource content
# ---------------------------------------------------------------------------


def test_tasks_resource_returns_valid_json():
    """Reading a2a://tasks should return parseable JSON."""
    content = _read_resource_sync(_make_server(), "a2a://tasks")
    parsed = json.loads(content)
    assert "tasks" in parsed
    assert "count" in parsed


def test_tasks_resource_empty_when_no_tasks():
    content = _read_resource_sync(_make_server(), "a2a://tasks")
    parsed = json.loads(content)
    assert parsed["count"] == 0
    assert parsed["tasks"] == []


def test_tasks_resource_count_field_is_integer():
    content = _read_resource_sync(_make_server(), "a2a://tasks")
    parsed = json.loads(content)
    assert isinstance(parsed["count"], int)


def test_tasks_resource_conforms_to_task_list_output_schema():
    content = _read_resource_sync(_make_server(), "a2a://tasks")
    parsed = json.loads(content)
    # Validate by instantiating the model — pydantic will raise on bad data
    output = TaskListOutput(**parsed)
    assert output.count == len(output.tasks)


def test_tasks_resource_reflects_custom_task_store():
    """When a pre-populated task store is provided, a2a://tasks reflects it."""
    import asyncio
    from agentique.bridge.storage import InMemoryTaskStore
    from agentique.core.types import TaskTracker, TaskState

    store = InMemoryTaskStore()

    async def _setup():
        tracker = TaskTracker(task_id="pre-existing", state=TaskState.completed)
        await store.save("pre-existing", tracker)

    asyncio.run(_setup())

    content = _read_resource_sync(_make_server(task_store=store), "a2a://tasks")
    parsed = json.loads(content)
    assert parsed["count"] == 1
    assert parsed["tasks"][0]["task_id"] == "pre-existing"
    assert parsed["tasks"][0]["state"] == "completed"


def test_tasks_resource_task_summary_has_required_fields():
    import asyncio
    from agentique.bridge.storage import InMemoryTaskStore
    from agentique.core.types import TaskTracker, TaskState

    store = InMemoryTaskStore()

    async def _setup():
        tracker = TaskTracker(task_id="t1", context_id="ctx-42", state=TaskState.working)
        await store.save("t1", tracker)

    asyncio.run(_setup())

    content = _read_resource_sync(_make_server(task_store=store), "a2a://tasks")
    parsed = json.loads(content)
    task = parsed["tasks"][0]
    assert task["task_id"] == "t1"
    assert task["state"] == "working"
    assert "event_count" in task
    assert "artifact_count" in task
