"""Task lifecycle management.

Provides task tracking with pluggable storage backends.
Defaults to in-memory storage; production deployments can use Redis,
DynamoDB, or other backends via the ``TaskStore`` protocol.

Artifact capture:
    When an agent emits an artifact event, ``capture_artifact()`` stores
    the content keyed by ``(task_id, artifact_id)``. The artifact can then
    be served as an MCP Resource via the ``a2a://{task_id}/artifacts/{id}``
    URI scheme registered in ``server.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentique.core.errors import TaskNotFoundError
from agentique.core.types import AgentEvent, AgentHierarchy, TaskState, TaskTracker
from .storage import InMemoryTaskStore, TaskStore


class TaskManager:
    """Manages task lifecycle and state transitions.

    Args:
        store: Pluggable storage backend. Defaults to ``InMemoryTaskStore``.
    """

    def __init__(self, store: TaskStore | None = None) -> None:
        self._store = store or InMemoryTaskStore()
        self._lock = asyncio.Lock()
        self._conversations: dict[str, list[dict[str, str]]] = {}
        # task_id → artifact_id → artifact payload
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
            and ``artifact_id`` keys, or ``None`` if not found.
        """
        task_artifacts = self._artifacts.get(task_id)
        if task_artifacts is None:
            return None
        return task_artifacts.get(artifact_id)

    def list_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        """List all artifacts captured for *task_id*.

        Returns:
            List of artifact metadata dicts (without ``content`` for
            compactness). Use ``get_artifact()`` to retrieve content.
        """
        task_artifacts = self._artifacts.get(task_id, {})
        return [
            {k: v for k, v in entry.items() if k != "content"}
            for entry in task_artifacts.values()
        ]

    # ---- Conversation continuity ----

    def get_conversation_history(
        self, context_id: str, max_turns: int = 10,
    ) -> list[dict[str, str]]:
        history = self._conversations.get(context_id, [])
        return history[-max_turns:]

    def append_conversation(
        self,
        context_id: str,
        user_message: str,
        agent_response: str,
        max_turns: int = 20,
    ) -> None:
        if context_id not in self._conversations:
            self._conversations[context_id] = []
        self._conversations[context_id].append(
            {"role": "user", "content": user_message}
        )
        self._conversations[context_id].append(
            {"role": "agent", "content": agent_response}
        )
        self._conversations[context_id] = self._conversations[context_id][-max_turns:]
