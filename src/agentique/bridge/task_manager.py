"""Task lifecycle management.

Provides in-memory task tracking with future extensibility to pluggable
storage backends (Redis, DynamoDB, etc.).
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentique.core.errors import TaskNotFoundError
from agentique.core.types import AgentEvent, AgentHierarchy, TaskState, TaskTracker


class TaskManager:
    """Manages task lifecycle and state transitions."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskTracker] = {}
        self._lock = asyncio.Lock()
        self._conversations: dict[str, list[dict[str, str]]] = {}

    async def create(
        self,
        task_id: str,
        context_id: str | None = None,
    ) -> TaskTracker:
        tracker = TaskTracker(task_id=task_id, context_id=context_id)
        async with self._lock:
            self._tasks[task_id] = tracker
        return tracker

    async def get(self, task_id: str) -> TaskTracker:
        async with self._lock:
            tracker = self._tasks.get(task_id)
        if tracker is None:
            raise TaskNotFoundError(f"Task {task_id} not found")
        return tracker

    async def get_or_none(self, task_id: str) -> TaskTracker | None:
        async with self._lock:
            return self._tasks.get(task_id)

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
