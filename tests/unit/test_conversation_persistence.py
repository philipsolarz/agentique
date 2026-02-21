"""Tests for Focus #9c: Conversation History Persistence.

Covers:
  - TaskStore protocol includes save_conversation / load_conversation
  - InMemoryTaskStore implements conversation persistence
  - TaskManager.aget_conversation_history() reads from store
  - TaskManager.aappend_conversation() writes to store
  - Conversation history survives TaskManager re-creation with same store
  - Redis/DynamoDB stores implement conversation methods
  - max_turns slicing in aget_conversation_history
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.anyio

from agentique.bridge.storage import InMemoryTaskStore, TaskStore
from agentique.bridge.task_manager import TaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager(store=None) -> TaskManager:
    return TaskManager(store=store or InMemoryTaskStore())


# ---------------------------------------------------------------------------
# TaskStore protocol includes conversation methods
# ---------------------------------------------------------------------------


def test_task_store_protocol_has_save_conversation():
    """TaskStore protocol must declare save_conversation."""
    import inspect
    members = dict(inspect.getmembers(TaskStore))
    assert "save_conversation" in members, (
        "TaskStore protocol must include save_conversation"
    )


def test_task_store_protocol_has_load_conversation():
    """TaskStore protocol must declare load_conversation."""
    import inspect
    members = dict(inspect.getmembers(TaskStore))
    assert "load_conversation" in members, (
        "TaskStore protocol must include load_conversation"
    )


def test_in_memory_store_implements_task_store_protocol():
    """InMemoryTaskStore must satisfy the TaskStore protocol."""
    store = InMemoryTaskStore()
    assert isinstance(store, TaskStore)


# ---------------------------------------------------------------------------
# InMemoryTaskStore conversation storage
# ---------------------------------------------------------------------------


async def test_in_memory_save_and_load_conversation():
    """save_conversation followed by load_conversation returns the same turns."""
    store = InMemoryTaskStore()
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "agent", "content": "Hi there!"},
    ]
    await store.save_conversation("ctx-1", turns)
    loaded = await store.load_conversation("ctx-1")
    assert loaded == turns


async def test_in_memory_load_absent_conversation_returns_empty():
    store = InMemoryTaskStore()
    result = await store.load_conversation("nonexistent-ctx")
    assert result == []


async def test_in_memory_save_conversation_overwrites():
    """Saving conversation again replaces the previous history."""
    store = InMemoryTaskStore()
    await store.save_conversation("ctx-1", [{"role": "user", "content": "First"}])
    await store.save_conversation("ctx-1", [{"role": "user", "content": "Second"}])
    loaded = await store.load_conversation("ctx-1")
    assert len(loaded) == 1
    assert loaded[0]["content"] == "Second"


async def test_in_memory_conversations_are_isolated_by_context():
    """Conversations for different context IDs don't interfere."""
    store = InMemoryTaskStore()
    await store.save_conversation("ctx-a", [{"role": "user", "content": "A"}])
    await store.save_conversation("ctx-b", [{"role": "user", "content": "B"}])
    a = await store.load_conversation("ctx-a")
    b = await store.load_conversation("ctx-b")
    assert a[0]["content"] == "A"
    assert b[0]["content"] == "B"


# ---------------------------------------------------------------------------
# TaskManager conversation delegation
# ---------------------------------------------------------------------------


async def test_aappend_conversation_persists_to_store():
    """aappend_conversation writes turns to the backing store."""
    store = InMemoryTaskStore()
    mgr = make_manager(store)
    await mgr.aappend_conversation("ctx-1", "What is 2+2?", "4")
    loaded = await store.load_conversation("ctx-1")
    assert len(loaded) == 2
    assert loaded[0] == {"role": "user", "content": "What is 2+2?"}
    assert loaded[1] == {"role": "agent", "content": "4"}


async def test_aget_conversation_history_reads_from_store():
    """aget_conversation_history returns what's in the store."""
    store = InMemoryTaskStore()
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "agent", "content": "Hi"},
    ]
    await store.save_conversation("ctx-1", turns)
    mgr = make_manager(store)
    result = await mgr.aget_conversation_history("ctx-1")
    assert result == turns


async def test_aget_conversation_history_empty_for_new_context():
    mgr = make_manager()
    result = await mgr.aget_conversation_history("new-ctx")
    assert result == []


async def test_aget_conversation_history_respects_max_turns():
    """aget_conversation_history slices to max_turns pairs."""
    store = InMemoryTaskStore()
    # Add 10 turns (5 pairs)
    turns = []
    for i in range(5):
        turns.append({"role": "user", "content": f"msg {i}"})
        turns.append({"role": "agent", "content": f"reply {i}"})
    await store.save_conversation("ctx-1", turns)
    mgr = make_manager(store)

    result = await mgr.aget_conversation_history("ctx-1", max_turns=4)
    assert len(result) == 4
    # Should be the last 4 turns
    assert result == turns[-4:]


async def test_aappend_conversation_truncates_to_max_turns():
    """aappend_conversation applies max_turns cap."""
    store = InMemoryTaskStore()
    # Fill up to 20 turns (default max)
    for i in range(10):
        await InMemoryTaskStore.save_conversation(
            store,
            "ctx-1",
            [{"role": "user", "content": f"u{i}"}, {"role": "agent", "content": f"a{i}"}] * 1,
        )
    mgr = make_manager(store)
    # Add many turns and check max
    for i in range(12):
        await mgr.aappend_conversation("ctx-1", f"u{i}", f"a{i}", max_turns=6)
    history = await mgr.aget_conversation_history("ctx-1", max_turns=100)
    assert len(history) <= 6


# ---------------------------------------------------------------------------
# Conversation survives TaskManager re-creation with same store
# ---------------------------------------------------------------------------


async def test_conversation_persists_across_manager_instances():
    """When the same store is reused, conversation history is preserved."""
    store = InMemoryTaskStore()

    # First manager appends a conversation
    mgr1 = make_manager(store)
    await mgr1.aappend_conversation("ctx-1", "Hello", "Hi there!")

    # Second manager (same store) reads it back
    mgr2 = make_manager(store)
    history = await mgr2.aget_conversation_history("ctx-1")
    assert len(history) == 2
    assert history[0]["content"] == "Hello"
    assert history[1]["content"] == "Hi there!"


# ---------------------------------------------------------------------------
# Redis store conversation methods
# ---------------------------------------------------------------------------


async def test_redis_store_save_and_load_conversation():
    """RedisTaskStore implements save_conversation and load_conversation."""
    import json
    from unittest.mock import AsyncMock, MagicMock

    from agentique.bridge.persistent_stores import RedisTaskStore

    mock_redis = MagicMock()
    mock_redis.setex = AsyncMock()
    mock_redis.get = AsyncMock(
        return_value=json.dumps([{"role": "user", "content": "hi"}])
    )

    store = RedisTaskStore(client=mock_redis)
    turns = [{"role": "user", "content": "hi"}]
    await store.save_conversation("ctx-1", turns)
    mock_redis.setex.assert_called_once()

    loaded = await store.load_conversation("ctx-1")
    assert loaded == turns


async def test_redis_store_load_absent_conversation_returns_empty():
    from unittest.mock import AsyncMock, MagicMock
    from agentique.bridge.persistent_stores import RedisTaskStore

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)

    store = RedisTaskStore(client=mock_redis)
    result = await store.load_conversation("nonexistent")
    assert result == []


# ---------------------------------------------------------------------------
# DynamoDB store conversation methods
# ---------------------------------------------------------------------------


async def test_dynamodb_store_save_and_load_conversation():
    """DynamoDBTaskStore implements save_conversation and load_conversation."""
    import json
    from unittest.mock import AsyncMock, MagicMock

    from agentique.bridge.persistent_stores import DynamoDBTaskStore

    turns = [{"role": "agent", "content": "Hello DynamoDB!"}]
    mock_client = MagicMock()
    mock_client.put_item = AsyncMock()
    mock_client.get_item = AsyncMock(
        return_value={
            "Item": {
                "task_id": {"S": "conv:ctx-1"},
                "data": {"S": json.dumps(turns)},
            }
        }
    )

    try:
        import aiobotocore  # noqa: F401
    except ImportError:
        pytest.skip("aiobotocore not installed")

    store = DynamoDBTaskStore()
    store._client = mock_client

    await store.save_conversation("ctx-1", turns)
    mock_client.put_item.assert_called_once()

    loaded = await store.load_conversation("ctx-1")
    assert loaded == turns


async def test_dynamodb_store_load_absent_conversation_returns_empty():
    """DynamoDB returns empty list when context has no saved conversation."""
    from unittest.mock import AsyncMock, MagicMock
    from agentique.bridge.persistent_stores import DynamoDBTaskStore

    mock_client = MagicMock()
    mock_client.get_item = AsyncMock(return_value={"Item": None})

    try:
        import aiobotocore  # noqa: F401
    except ImportError:
        pytest.skip("aiobotocore not installed")

    store = DynamoDBTaskStore()
    store._client = mock_client

    result = await store.load_conversation("nonexistent")
    assert result == []
