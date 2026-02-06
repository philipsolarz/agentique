"""Tests for the pluggable storage backend abstraction."""

from __future__ import annotations

import pytest

from agentique.bridge.storage import InMemoryTaskStore, TaskStore
from agentique.bridge.task_manager import TaskManager
from agentique.core.errors import TaskNotFoundError
from agentique.core.types import TaskState, TaskTracker


# ---- InMemoryTaskStore tests ----


@pytest.mark.asyncio
async def test_inmemory_store_save_and_load():
    store = InMemoryTaskStore()
    tracker = TaskTracker(task_id="t1", context_id="ctx1")

    await store.save("t1", tracker)
    loaded = await store.load("t1")

    assert loaded is tracker
    assert loaded.task_id == "t1"
    assert loaded.context_id == "ctx1"


@pytest.mark.asyncio
async def test_inmemory_store_load_nonexistent():
    store = InMemoryTaskStore()
    result = await store.load("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_inmemory_store_delete():
    store = InMemoryTaskStore()
    tracker = TaskTracker(task_id="t1")
    await store.save("t1", tracker)

    await store.delete("t1")
    assert await store.load("t1") is None


@pytest.mark.asyncio
async def test_inmemory_store_delete_nonexistent():
    store = InMemoryTaskStore()
    await store.delete("nonexistent")  # Should not raise


@pytest.mark.asyncio
async def test_inmemory_store_list_ids():
    store = InMemoryTaskStore()
    await store.save("t1", TaskTracker(task_id="t1"))
    await store.save("t2", TaskTracker(task_id="t2"))

    ids = await store.list_ids()
    assert sorted(ids) == ["t1", "t2"]


# ---- TaskStore protocol compliance ----


def test_inmemory_store_satisfies_protocol():
    """InMemoryTaskStore should satisfy the TaskStore protocol."""
    assert isinstance(InMemoryTaskStore(), TaskStore)


# ---- TaskManager with custom store ----


@pytest.mark.asyncio
async def test_task_manager_uses_custom_store():
    """TaskManager should use the injected store."""
    store = InMemoryTaskStore()
    manager = TaskManager(store=store)

    tracker = await manager.create("t1", "ctx1")
    assert tracker.task_id == "t1"

    # Verify it's in the store
    loaded = await store.load("t1")
    assert loaded is tracker


@pytest.mark.asyncio
async def test_task_manager_get_from_store():
    store = InMemoryTaskStore()
    manager = TaskManager(store=store)

    await manager.create("t1")
    tracker = await manager.get("t1")
    assert tracker.task_id == "t1"


@pytest.mark.asyncio
async def test_task_manager_get_or_none_from_store():
    store = InMemoryTaskStore()
    manager = TaskManager(store=store)

    result = await manager.get_or_none("nonexistent")
    assert result is None

    await manager.create("t1")
    result = await manager.get_or_none("t1")
    assert result is not None


@pytest.mark.asyncio
async def test_task_manager_not_found():
    store = InMemoryTaskStore()
    manager = TaskManager(store=store)

    with pytest.raises(TaskNotFoundError):
        await manager.get("nonexistent")


@pytest.mark.asyncio
async def test_task_manager_default_store():
    """TaskManager without explicit store should use InMemoryTaskStore."""
    manager = TaskManager()
    tracker = await manager.create("t1")
    assert tracker.task_id == "t1"

    loaded = await manager.get("t1")
    assert loaded is tracker
