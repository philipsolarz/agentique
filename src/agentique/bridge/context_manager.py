"""Context manager for MCP session ↔ A2A context ID mapping.

Maintains the bidirectional relationship between MCP sessions and
A2A context IDs. Each MCP session can spawn multiple A2A contexts
(one per conversation thread). This module ensures proper context
propagation across protocol boundaries.

A2A rules enforced:
    - Agents MUST infer contextId from task if only taskId is provided
    - Agents MUST reject messages with mismatching contextId and taskId
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agentique.core.types import ContextMapping

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages MCP session ↔ A2A context lifecycle.

    Thread-safe via asyncio locks. Provides context resolution,
    binding, and cleanup operations.
    """

    def __init__(self) -> None:
        self._mapping = ContextMapping()
        self._lock = asyncio.Lock()

    async def resolve_context(
        self,
        *,
        session_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """Resolve or create a context ID.

        If *context_id* is provided, bind it to the session and return it.
        If only *session_id* is given, return the most recent context or
        create a new one derived from *task_id* (or session_id).
        """
        async with self._lock:
            if context_id:
                if session_id:
                    self._mapping.bind(session_id, context_id)
                return context_id

            # Try to find existing context for this session
            if session_id:
                existing = self._mapping.get_contexts(session_id)
                if existing:
                    # Return most recently bound context
                    return max(existing)

            # Create a new context ID
            new_ctx = task_id or session_id or _generate_id()
            if session_id:
                self._mapping.bind(session_id, new_ctx)
            return new_ctx

    async def bind(self, session_id: str, context_id: str) -> None:
        """Explicitly bind a context to a session."""
        async with self._lock:
            self._mapping.bind(session_id, context_id)

    async def track_task(self, context_id: str, task_id: str) -> None:
        """Associate a task ID with its context."""
        async with self._lock:
            self._mapping.track_task(context_id, task_id)

    async def get_tasks(self, context_id: str) -> list[str]:
        """Return all task IDs for a given context."""
        async with self._lock:
            return self._mapping.get_tasks(context_id)

    async def get_session(self, context_id: str) -> str | None:
        """Look up the MCP session for a context."""
        async with self._lock:
            return self._mapping.get_session(context_id)

    async def cleanup_session(self, session_id: str) -> None:
        """Remove all mappings for a disconnected MCP session."""
        async with self._lock:
            self._mapping.unbind_session(session_id)
        logger.debug("Cleaned up context mappings for session %s", session_id)

    def validate_context_task(
        self,
        context_id: str | None,
        task_id: str | None,
    ) -> bool:
        """Validate that context and task IDs are consistent.

        Per A2A spec: agents MUST reject messages with mismatching
        contextId and taskId.
        """
        if context_id is None or task_id is None:
            return True
        tasks = self._mapping.get_tasks(context_id)
        if not tasks:
            # New context — no conflict
            return True
        return task_id in tasks


def _generate_id() -> str:
    """Generate a unique context ID."""
    from uuid import uuid4
    return str(uuid4())
