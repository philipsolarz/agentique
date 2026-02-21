"""Task lifecycle management.

Provides task tracking with pluggable storage backends.
Defaults to in-memory storage; production deployments can use Redis,
DynamoDB, or other backends via the ``TaskStore`` protocol.

Artifact capture:
    When an agent emits an artifact event, ``capture_artifact()`` stores
    the content keyed by ``(task_id, artifact_id)``. The artifact can then
    be served as an MCP Resource via the ``a2a://{task_id}/artifacts/{id}``
    URI scheme registered in ``server.py``.

Artifact TTL:
    When ``artifact_ttl`` is set (seconds), ``evict_artifacts()`` removes
    artifact entries older than the given age. The cleanup lifespan in
    ``lifespans.py`` can call this periodically.

Conversation persistence:
    Conversation history is persisted through the ``TaskStore`` backend
    (via ``save_conversation`` / ``load_conversation``), so that gateway
    restarts do not lose context when a persistent store is used.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agentique.core.errors import TaskNotFoundError
from agentique.core.types import AgentEvent, AgentHierarchy, TaskState, TaskTracker
from .storage import InMemoryTaskStore, TaskStore


class TaskManager:
    """Manages task lifecycle and state transitions.

    Args:
        store: Pluggable storage backend. Defaults to ``InMemoryTaskStore``.
        artifact_ttl: Optional time-to-live for artifacts in seconds.
            When set, ``evict_artifacts()`` removes entries older than this.
            ``None`` (default) keeps artifacts indefinitely.
    """

    def __init__(
        self,
        store: TaskStore | None = None,
        *,
        artifact_ttl: float | None = None,
    ) -> None:
        self._store = store or InMemoryTaskStore()
        self._lock = asyncio.Lock()
        self._artifact_ttl = artifact_ttl
        # task_id → artifact_id → artifact payload (includes created_at timestamp)
        self._artifacts: dict[str, dict[str, dict[str, Any]]] = {}

    async def create(
        self,
        task_id: str,
        context_id: str | None = None,
    ) -> TaskTracker:
        tracker = TaskTracker(task_id=task_id, context_id=context_id)
        await self._store.save(task_id, tracker)
        return tracker

    async def get(self, task_id: str) -> TaskTracker:
        tracker = await self._store.load(task_id)
        if tracker is None:
            raise TaskNotFoundError(f"Task {task_id} not found")
        return tracker

    async def get_or_none(self, task_id: str) -> TaskTracker | None:
        return await self._store.load(task_id)

    async def add_event(self, task_id: str, event: AgentEvent) -> None:
        tracker = await self.get(task_id)
        tracker.add_event(event)

    async def transition(
        self, task_id: str, state: TaskState, message: str | None = None,
    ) -> bool:
        tracker = await self.get(task_id)
        return tracker.transition(state, message)

    async def set_hierarchy(self, task_id: str, hierarchy: AgentHierarchy) -> None:
        tracker = await self.get(task_id)
        tracker.hierarchy = hierarchy

    async def list_tasks(self) -> list[str]:
        """Return all task IDs currently in the store."""
        return await self._store.list_ids()

    async def list_tasks_by_state(self, state: TaskState) -> list[str]:
        """Return all task IDs whose current state matches *state*.

        Args:
            state: The ``TaskState`` to filter by.

        Returns:
            List of task IDs in the given state.
        """
        ids = await self._store.list_ids()
        result: list[str] = []
        for tid in ids:
            tracker = await self._store.load(tid)
            if tracker is not None and tracker.state == state:
                result.append(tid)
        return result

    # ---- Artifact capture & retrieval ----

    def capture_artifact(
        self,
        task_id: str,
        artifact_id: str,
        content: str,
        *,
        name: str | None = None,
        mime_type: str = "text/plain",
    ) -> str:
        """Store an artifact emitted by an agent.

        Artifacts are keyed by ``(task_id, artifact_id)`` and can be
        retrieved via ``get_artifact()`` or listed via ``list_artifacts()``.
        The MCP Resource URI for each artifact is::

            a2a://{task_id}/artifacts/{artifact_id}

        Args:
            task_id: The task that produced the artifact.
            artifact_id: Unique identifier for the artifact within the task.
            content: The artifact's text content.
            name: Human-readable artifact name (defaults to *artifact_id*).
            mime_type: MIME type of the content.

        Returns:
            The MCP Resource URI for the stored artifact.
        """
        if task_id not in self._artifacts:
            self._artifacts[task_id] = {}
        self._artifacts[task_id][artifact_id] = {
            "content": content,
            "name": name or artifact_id,
            "mime_type": mime_type,
            "task_id": task_id,
            "artifact_id": artifact_id,
            "created_at": time.monotonic(),
        }
        return f"a2a://{task_id}/artifacts/{artifact_id}"

    def get_artifact(
        self,
        task_id: str,
        artifact_id: str,
    ) -> str | None:
        """Retrieve the content of a stored artifact.

        Args:
            task_id: The task that produced the artifact.
            artifact_id: The artifact identifier.

        Returns:
            The artifact's text content, or ``None`` if not found.
        """
        task_artifacts = self._artifacts.get(task_id)
        if task_artifacts is None:
            return None
        entry = task_artifacts.get(artifact_id)
        return entry["content"] if entry else None

    def get_artifact_metadata(
        self,
        task_id: str,
        artifact_id: str,
    ) -> dict[str, Any] | None:
        """Return full metadata for a stored artifact.

        Returns:
            Dict with ``content``, ``name``, ``mime_type``, ``task_id``,
            ``artifact_id``, and ``created_at`` keys, or ``None`` if not found.
        """
        task_artifacts = self._artifacts.get(task_id)
        if task_artifacts is None:
            return None
        return task_artifacts.get(artifact_id)

    def list_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        """List all artifacts captured for *task_id*.

        Returns:
            List of artifact metadata dicts (without ``content`` and
            ``created_at`` for compactness). Use ``get_artifact()`` to
            retrieve content.
        """
        task_artifacts = self._artifacts.get(task_id, {})
        return [
            {k: v for k, v in entry.items() if k not in ("content", "created_at")}
            for entry in task_artifacts.values()
        ]

    def evict_artifacts(self, task_id: str, *, before: float) -> int:
        """Remove artifacts created before *before* (monotonic timestamp).

        Args:
            task_id: The task to evict artifacts from.
            before: Monotonic timestamp threshold (``time.monotonic()`` value).
                Artifacts with ``created_at < before`` are removed.

        Returns:
            Number of artifacts evicted.
        """
        task_artifacts = self._artifacts.get(task_id)
        if not task_artifacts:
            return 0
        to_remove = [
            aid for aid, entry in task_artifacts.items()
            if entry.get("created_at", 0.0) < before
        ]
        for aid in to_remove:
            del task_artifacts[aid]
        if not task_artifacts:
            del self._artifacts[task_id]
        return len(to_remove)

    def evict_all_expired_artifacts(self) -> int:
        """Remove expired artifacts from all tasks (uses configured TTL).

        Only has effect when ``artifact_ttl`` was set at construction.

        Returns:
            Total number of artifacts evicted across all tasks.
        """
        if self._artifact_ttl is None:
            return 0
        cutoff = time.monotonic() - self._artifact_ttl
        total = 0
        for task_id in list(self._artifacts):
            total += self.evict_artifacts(task_id, before=cutoff)
        return total

    # ---- Conversation continuity (delegated to store) ----

    def get_conversation_history(
        self, context_id: str, max_turns: int = 10,
    ) -> list[dict[str, str]]:
        """Retrieve conversation history for *context_id* (sync, from store cache).

        For async-safe access use ``aget_conversation_history()``.
        Returns the last *max_turns* turns.
        """
        # Sync wrapper using asyncio.run when called from sync context.
        # In an async context, prefer aget_conversation_history().
        import asyncio as _asyncio
        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're in an async context — can't use asyncio.run().
            # Fall back to whatever the store cached in memory.
            # This is a best-effort synchronous path; callers in async
            # contexts should use aget_conversation_history() directly.
            store = self._store
            if hasattr(store, "_conversations"):
                history = store._conversations.get(context_id, [])
                return history[-max_turns:]
            return []
        else:
            history = _asyncio.run(self._store.load_conversation(context_id))
            return history[-max_turns:]

    async def aget_conversation_history(
        self, context_id: str, max_turns: int = 10,
    ) -> list[dict[str, str]]:
        """Async-safe retrieval of conversation history for *context_id*."""
        history = await self._store.load_conversation(context_id)
        return history[-max_turns:]

    def append_conversation(
        self,
        context_id: str,
        user_message: str,
        agent_response: str,
        max_turns: int = 20,
    ) -> None:
        """Append a conversation turn (sync wrapper — schedules async save).

        In async contexts, prefer ``aappend_conversation()``.
        """
        import asyncio as _asyncio
        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Schedule the coroutine as a task without blocking
            loop.create_task(
                self.aappend_conversation(
                    context_id, user_message, agent_response, max_turns
                )
            )
        else:
            _asyncio.run(
                self.aappend_conversation(
                    context_id, user_message, agent_response, max_turns
                )
            )

    async def aappend_conversation(
        self,
        context_id: str,
        user_message: str,
        agent_response: str,
        max_turns: int = 20,
    ) -> None:
        """Async-safe: append a conversation turn and persist to store."""
        history = await self._store.load_conversation(context_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "agent", "content": agent_response})
        history = history[-max_turns:]
        await self._store.save_conversation(context_id, history)
