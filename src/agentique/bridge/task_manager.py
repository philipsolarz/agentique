"""Task lifecycle management.

Provides task tracking with pluggable storage backends.
Defaults to in-memory storage; production deployments can use Redis,
DynamoDB, or other backends via the ``TaskStore`` protocol.
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
