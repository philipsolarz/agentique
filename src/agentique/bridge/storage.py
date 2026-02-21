"""Pluggable storage backends for task and conversation persistence.

Provides a ``TaskStore`` protocol and an in-memory default implementation.
Production deployments can swap in Redis, DynamoDB, or filesystem backends
by implementing the protocol.

Usage::

    from agentique.bridge.storage import InMemoryTaskStore, TaskStore

    store = InMemoryTaskStore()
    manager = TaskManager(store=store)
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from agentique.core.types import TaskTracker


@runtime_checkable
class TaskStore(Protocol):
    """Protocol for task persistence backends.

    Any class implementing these async methods is a valid store.

    Task CRUD methods (required):
        save, load, delete, list_ids

    Conversation persistence methods (required):
        save_conversation, load_conversation
    """

    async def save(self, task_id: str, tracker: TaskTracker) -> None:
        """Persist a task tracker."""
        ...

    async def load(self, task_id: str) -> TaskTracker | None:
        """Load a task tracker by ID, or None if not found."""
        ...

    async def delete(self, task_id: str) -> None:
        """Remove a task tracker."""
        ...

    async def list_ids(self) -> list[str]:
        """Return all stored task IDs."""
        ...

    async def save_conversation(
        self,
        context_id: str,
        turns: list[dict[str, str]],
    ) -> None:
        """Persist the conversation history for a context ID.

        Args:
            context_id: The A2A context ID for this conversation.
            turns: Complete list of conversation turns (replaces any existing).
                Each turn is a dict with ``role`` and ``content`` keys.
        """
        ...

    async def load_conversation(
        self,
        context_id: str,
    ) -> list[dict[str, str]]:
        """Load conversation history for a context ID.

        Args:
            context_id: The A2A context ID to retrieve history for.

        Returns:
            List of conversation turns, or empty list if none stored.
        """
        ...


class InMemoryTaskStore:
    """Default in-memory task store.

    Suitable for development and testing. Not persistent across restarts.
    Implements both task CRUD and conversation history storage.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskTracker] = {}
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._lock = asyncio.Lock()

    async def save(self, task_id: str, tracker: TaskTracker) -> None:
        async with self._lock:
            self._tasks[task_id] = tracker

    async def load(self, task_id: str) -> TaskTracker | None:
        async with self._lock:
            return self._tasks.get(task_id)

    async def delete(self, task_id: str) -> None:
        async with self._lock:
            self._tasks.pop(task_id, None)

    async def list_ids(self) -> list[str]:
        async with self._lock:
            return list(self._tasks.keys())

    async def save_conversation(
        self,
        context_id: str,
        turns: list[dict[str, str]],
    ) -> None:
        async with self._lock:
            self._conversations[context_id] = list(turns)

    async def load_conversation(
        self,
        context_id: str,
    ) -> list[dict[str, str]]:
        async with self._lock:
            return list(self._conversations.get(context_id, []))
