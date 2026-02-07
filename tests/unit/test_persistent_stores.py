"""Tests for Redis and DynamoDB TaskStore implementations.

Since we can't depend on actual Redis/DynamoDB in tests, we use mock
clients that simulate the async interface. This validates serialization,
deserialization, and store logic.
"""

from __future__ import annotations

import json
import pytest

from agentique.core.types import (
    AgentEvent,
    AgentHierarchy,
    TaskState,
    TaskTracker,
)
from agentique.bridge.persistent_stores import (
    _serialize_tracker,
    _deserialize_tracker,
)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip_basic(self):
        tracker = TaskTracker(
            task_id="t1",
            context_id="ctx-1",
            state=TaskState.working,
            progress=42.0,
            message="half done",
        )
        raw = _serialize_tracker(tracker)
        restored = _deserialize_tracker(raw)

        assert restored.task_id == "t1"
        assert restored.context_id == "ctx-1"
        assert restored.state == TaskState.working
        assert restored.progress == 42.0
        assert restored.message == "half done"

    def test_round_trip_with_events(self):
        event = AgentEvent(
            kind="message",
            text="hello",
            task_id="t1",
            branch="root.sub",
            author="agent-a",
        )
        tracker = TaskTracker(task_id="t1", events=[event])
        raw = _serialize_tracker(tracker)
        restored = _deserialize_tracker(raw)

        assert len(restored.events) == 1
        assert restored.events[0].kind == "message"
        assert restored.events[0].text == "hello"
        assert restored.events[0].branch == "root.sub"

    def test_round_trip_with_hierarchy(self):
        hierarchy = AgentHierarchy(root="main")
        hierarchy.add_agent("sub-1", description="First sub-agent", parent=None)
        hierarchy.add_agent("sub-2", parent="sub-1")

        tracker = TaskTracker(task_id="t1", hierarchy=hierarchy)
        raw = _serialize_tracker(tracker)
        restored = _deserialize_tracker(raw)

        assert restored.hierarchy is not None
        assert restored.hierarchy.root == "main"
        assert "sub-1" in restored.hierarchy.agents
        assert restored.hierarchy.agents["sub-2"].parent == "sub-1"

    def test_unknown_state_fallback(self):
        data = json.dumps({
            "task_id": "t1",
            "state": "some-future-state",
            "events": [],
        })
        restored = _deserialize_tracker(data)
        assert restored.state == TaskState.unknown

    def test_serialize_with_metadata(self):
        tracker = TaskTracker(
            task_id="t1",
            metadata={"key": "value"},
            artifacts=[{"id": "a1", "name": "result.txt"}],
        )
        raw = _serialize_tracker(tracker)
        restored = _deserialize_tracker(raw)
        assert restored.metadata == {"key": "value"}
        assert len(restored.artifacts) == 1


# ---------------------------------------------------------------------------
# Mock Redis client
# ---------------------------------------------------------------------------


class MockRedisClient:
    """In-memory mock of redis.asyncio.Redis for testing."""

    def __init__(self):
        self._data: dict[str, str] = {}
        self._ttl: dict[str, int] = {}

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._data[key] = value
        self._ttl[key] = ttl

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def scan_iter(self, match: str = "*"):
        import fnmatch
        for key in list(self._data.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# RedisTaskStore tests
# ---------------------------------------------------------------------------


class TestRedisTaskStore:
    @pytest.fixture
    def store(self):
        from agentique.bridge.persistent_stores import RedisTaskStore
        client = MockRedisClient()
        return RedisTaskStore(client=client), client

    @pytest.mark.anyio
    async def test_save_and_load(self, store):
        store_inst, _ = store
        tracker = TaskTracker(task_id="t1", state=TaskState.working)
        await store_inst.save("t1", tracker)

        loaded = await store_inst.load("t1")
        assert loaded is not None
        assert loaded.task_id == "t1"
        assert loaded.state == TaskState.working

    @pytest.mark.anyio
    async def test_load_missing(self, store):
        store_inst, _ = store
        assert await store_inst.load("nonexistent") is None

    @pytest.mark.anyio
    async def test_delete(self, store):
        store_inst, _ = store
        tracker = TaskTracker(task_id="t1")
        await store_inst.save("t1", tracker)
        await store_inst.delete("t1")
        assert await store_inst.load("t1") is None

    @pytest.mark.anyio
    async def test_list_ids(self, store):
        store_inst, _ = store
        await store_inst.save("t1", TaskTracker(task_id="t1"))
        await store_inst.save("t2", TaskTracker(task_id="t2"))

        ids = await store_inst.list_ids()
        assert sorted(ids) == ["t1", "t2"]

    @pytest.mark.anyio
    async def test_ttl_applied(self, store):
        store_inst, client = store
        tracker = TaskTracker(task_id="t1")
        await store_inst.save("t1", tracker)
        # Default TTL is 86400
        assert client._ttl.get("agentique:task:t1") == 86400

    @pytest.mark.anyio
    async def test_no_ttl(self):
        from agentique.bridge.persistent_stores import RedisTaskStore
        client = MockRedisClient()
        store_inst = RedisTaskStore(client=client, ttl=0)
        tracker = TaskTracker(task_id="t1")
        await store_inst.save("t1", tracker)
        assert "agentique:task:t1" not in client._ttl
        assert await store_inst.load("t1") is not None


# ---------------------------------------------------------------------------
# Mock DynamoDB client
# ---------------------------------------------------------------------------


class MockDynamoDBClient:
    """In-memory mock of aiobotocore DynamoDB client for testing."""

    def __init__(self):
        self._tables: dict[str, list[dict]] = {}

    async def put_item(self, TableName: str, Item: dict) -> None:
        if TableName not in self._tables:
            self._tables[TableName] = []
        # Remove existing item with same key
        task_id = Item.get("task_id", {}).get("S")
        self._tables[TableName] = [
            i for i in self._tables[TableName]
            if i.get("task_id", {}).get("S") != task_id
        ]
        self._tables[TableName].append(Item)

    async def get_item(self, TableName: str, Key: dict) -> dict:
        task_id = Key.get("task_id", {}).get("S")
        for item in self._tables.get(TableName, []):
            if item.get("task_id", {}).get("S") == task_id:
                return {"Item": item}
        return {}

    async def delete_item(self, TableName: str, Key: dict) -> None:
        task_id = Key.get("task_id", {}).get("S")
        if TableName in self._tables:
            self._tables[TableName] = [
                i for i in self._tables[TableName]
                if i.get("task_id", {}).get("S") != task_id
            ]

    def get_paginator(self, method: str):
        return MockPaginator(self)


class MockPaginator:
    def __init__(self, client: MockDynamoDBClient):
        self._client = client

    def paginate(self, TableName: str, **kwargs):
        return MockPageIterator(self._client._tables.get(TableName, []))


class MockPageIterator:
    def __init__(self, items: list[dict]):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not hasattr(self, "_done"):
            self._done = True
            return {"Items": self._items}
        raise StopAsyncIteration


# ---------------------------------------------------------------------------
# DynamoDBTaskStore tests
# ---------------------------------------------------------------------------


class TestDynamoDBTaskStore:
    @pytest.fixture
    def store(self):
        from agentique.bridge.persistent_stores import DynamoDBTaskStore

        store_inst = DynamoDBTaskStore.__new__(DynamoDBTaskStore)
        store_inst._table_name = "test-tasks"
        store_inst._region = "us-east-1"
        store_inst._endpoint_url = None
        store_inst._session = None
        store_inst._client_ctx = None
        store_inst._client = MockDynamoDBClient()
        return store_inst

    @pytest.mark.anyio
    async def test_save_and_load(self, store):
        tracker = TaskTracker(task_id="t1", state=TaskState.completed)
        await store.save("t1", tracker)

        loaded = await store.load("t1")
        assert loaded is not None
        assert loaded.task_id == "t1"
        assert loaded.state == TaskState.completed

    @pytest.mark.anyio
    async def test_load_missing(self, store):
        assert await store.load("nonexistent") is None

    @pytest.mark.anyio
    async def test_delete(self, store):
        tracker = TaskTracker(task_id="t1")
        await store.save("t1", tracker)
        await store.delete("t1")
        assert await store.load("t1") is None

    @pytest.mark.anyio
    async def test_list_ids(self, store):
        await store.save("t1", TaskTracker(task_id="t1"))
        await store.save("t2", TaskTracker(task_id="t2"))

        ids = await store.list_ids()
        assert sorted(ids) == ["t1", "t2"]

    @pytest.mark.anyio
    async def test_overwrite(self, store):
        tracker1 = TaskTracker(task_id="t1", state=TaskState.working)
        await store.save("t1", tracker1)

        tracker2 = TaskTracker(task_id="t1", state=TaskState.completed)
        await store.save("t1", tracker2)

        loaded = await store.load("t1")
        assert loaded.state == TaskState.completed
