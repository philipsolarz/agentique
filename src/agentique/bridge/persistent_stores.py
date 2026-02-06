"""Persistent TaskStore implementations for Redis and DynamoDB.

These stores implement the ``TaskStore`` protocol for production use,
providing durable task persistence across server restarts.

Both implementations gracefully handle missing dependencies — importing
this module always succeeds, but instantiating a store without its
backing library will raise ``ImportError`` at construction time.

Usage::

    # Redis
    from agentique.bridge.persistent_stores import RedisTaskStore
    store = RedisTaskStore(url="redis://localhost:6379/0")

    # DynamoDB
    from agentique.bridge.persistent_stores import DynamoDBTaskStore
    store = DynamoDBTaskStore(table_name="agentique-tasks", region="us-east-1")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from agentique.core.types import TaskState, TaskTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_tracker(tracker: TaskTracker) -> str:
    """Serialize a TaskTracker to JSON string."""
    data = tracker.to_dict()
    # Include events as serializable dicts
    data["events"] = [e.to_dict() for e in tracker.events]
    data["artifacts"] = tracker.artifacts
    return json.dumps(data)


def _deserialize_tracker(raw: str) -> TaskTracker:
    """Deserialize a TaskTracker from JSON string."""
    from agentique.core.types import AgentEvent, AgentHierarchy, SubAgentInfo

    data = json.loads(raw)
    events = []
    for evt_data in data.get("events", []):
        events.append(AgentEvent(
            kind=evt_data.get("kind", "message"),
            text=evt_data.get("text"),
            task_id=evt_data.get("task_id"),
            context_id=evt_data.get("context_id"),
            progress=evt_data.get("progress"),
            artifact_id=evt_data.get("artifact_id"),
            artifact_name=evt_data.get("artifact_name"),
            branch=evt_data.get("branch"),
            author=evt_data.get("author"),
            state=evt_data.get("state"),
            is_final=evt_data.get("is_final", False),
            requires_confirmation=evt_data.get("requires_confirmation", False),
            tool_call=evt_data.get("tool_call"),
            event_metadata=evt_data.get("metadata"),
        ))

    hierarchy = None
    hierarchy_data = data.get("hierarchy")
    if hierarchy_data and isinstance(hierarchy_data, dict):
        hierarchy = AgentHierarchy(root=hierarchy_data.get("root", ""))
        for name, info in hierarchy_data.get("agents", {}).items():
            hierarchy.agents[name] = SubAgentInfo(
                name=info.get("name", name),
                description=info.get("description"),
                skills=info.get("skills", []),
                parent=info.get("parent"),
                depth=info.get("depth", 0),
            )

    try:
        state = TaskState(data.get("state", "submitted"))
    except ValueError:
        state = TaskState.unknown

    return TaskTracker(
        task_id=data["task_id"],
        context_id=data.get("context_id"),
        state=state,
        events=events,
        artifacts=data.get("artifacts", []),
        progress=data.get("progress", 0.0),
        message=data.get("message"),
        metadata=data.get("metadata", {}),
        hierarchy=hierarchy,
    )


# ---------------------------------------------------------------------------
# Redis TaskStore
# ---------------------------------------------------------------------------


class RedisTaskStore:
    """TaskStore backed by Redis.

    Uses ``redis.asyncio`` for async operations. Tasks are stored as
    JSON strings with configurable key prefix and TTL.

    Args:
        url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        key_prefix: Prefix for all Redis keys. Defaults to ``agentique:task:``.
        ttl: Time-to-live in seconds for task entries. Defaults to 86400 (1 day).
            Set to 0 for no expiration.
        client: Pre-configured ``redis.asyncio.Redis`` instance.
            If provided, *url* is ignored.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "agentique:task:",
        ttl: int = 86400,
        client: Any | None = None,
    ) -> None:
        self._key_prefix = key_prefix
        self._ttl = ttl

        if client is not None:
            self._client = client
        else:
            try:
                import redis.asyncio as aioredis
            except ImportError:
                raise ImportError(
                    "redis.asyncio is required for RedisTaskStore. "
                    "Install with: pip install redis"
                )
            self._client = aioredis.from_url(url, decode_responses=True)

    def _key(self, task_id: str) -> str:
        return f"{self._key_prefix}{task_id}"

    async def save(self, task_id: str, tracker: TaskTracker) -> None:
        """Persist a task tracker to Redis."""
        data = _serialize_tracker(tracker)
        if self._ttl > 0:
            await self._client.setex(self._key(task_id), self._ttl, data)
        else:
            await self._client.set(self._key(task_id), data)

    async def load(self, task_id: str) -> TaskTracker | None:
        """Load a task tracker from Redis."""
        raw = await self._client.get(self._key(task_id))
        if raw is None:
            return None
        try:
            return _deserialize_tracker(raw)
        except Exception:
            logger.warning(
                "Failed to deserialize task %s from Redis", task_id,
                exc_info=True,
            )
            return None

    async def delete(self, task_id: str) -> None:
        """Remove a task tracker from Redis."""
        await self._client.delete(self._key(task_id))

    async def list_ids(self) -> list[str]:
        """Return all task IDs stored in Redis."""
        pattern = f"{self._key_prefix}*"
        keys = []
        async for key in self._client.scan_iter(match=pattern):
            task_id = key
            if isinstance(task_id, bytes):
                task_id = task_id.decode()
            task_id = task_id.removeprefix(self._key_prefix)
            keys.append(task_id)
        return keys

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._client.aclose()


# ---------------------------------------------------------------------------
# DynamoDB TaskStore
# ---------------------------------------------------------------------------


class DynamoDBTaskStore:
    """TaskStore backed by Amazon DynamoDB.

    Uses ``aiobotocore`` for async DynamoDB operations. Requires a
    pre-created table with ``task_id`` as the partition key (string).

    Args:
        table_name: DynamoDB table name.
        region: AWS region. Defaults to ``us-east-1``.
        endpoint_url: Optional custom endpoint (for local development).
        session: Pre-configured ``aiobotocore.AioSession``.
            If not provided, a default session is created.
    """

    def __init__(
        self,
        table_name: str = "agentique-tasks",
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        session: Any | None = None,
    ) -> None:
        self._table_name = table_name
        self._region = region
        self._endpoint_url = endpoint_url

        try:
            import aiobotocore.session
        except ImportError:
            raise ImportError(
                "aiobotocore is required for DynamoDBTaskStore. "
                "Install with: pip install aiobotocore"
            )

        self._session = session or aiobotocore.session.AioSession()
        self._client_ctx: Any | None = None
        self._client: Any | None = None

    async def _get_client(self) -> Any:
        """Get or create the DynamoDB client."""
        if self._client is None:
            kwargs: dict[str, Any] = {
                "service_name": "dynamodb",
                "region_name": self._region,
            }
            if self._endpoint_url:
                kwargs["endpoint_url"] = self._endpoint_url
            self._client_ctx = self._session.create_client(**kwargs)
            self._client = await self._client_ctx.__aenter__()
        return self._client

    async def save(self, task_id: str, tracker: TaskTracker) -> None:
        """Persist a task tracker to DynamoDB."""
        client = await self._get_client()
        data = _serialize_tracker(tracker)
        await client.put_item(
            TableName=self._table_name,
            Item={
                "task_id": {"S": task_id},
                "data": {"S": data},
                "state": {"S": tracker.state.value},
            },
        )

    async def load(self, task_id: str) -> TaskTracker | None:
        """Load a task tracker from DynamoDB."""
        client = await self._get_client()
        response = await client.get_item(
            TableName=self._table_name,
            Key={"task_id": {"S": task_id}},
        )
        item = response.get("Item")
        if not item:
            return None
        raw = item.get("data", {}).get("S")
        if not raw:
            return None
        try:
            return _deserialize_tracker(raw)
        except Exception:
            logger.warning(
                "Failed to deserialize task %s from DynamoDB", task_id,
                exc_info=True,
            )
            return None

    async def delete(self, task_id: str) -> None:
        """Remove a task tracker from DynamoDB."""
        client = await self._get_client()
        await client.delete_item(
            TableName=self._table_name,
            Key={"task_id": {"S": task_id}},
        )

    async def list_ids(self) -> list[str]:
        """Return all task IDs in the DynamoDB table."""
        client = await self._get_client()
        ids: list[str] = []
        paginator = client.get_paginator("scan")
        async for page in paginator.paginate(
            TableName=self._table_name,
            ProjectionExpression="task_id",
        ):
            for item in page.get("Items", []):
                task_id = item.get("task_id", {}).get("S")
                if task_id:
                    ids.append(task_id)
        return ids

    async def close(self) -> None:
        """Close the DynamoDB client."""
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(None, None, None)
            self._client = None
            self._client_ctx = None
