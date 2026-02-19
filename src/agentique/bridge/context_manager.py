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


_STATE_KEY_CONTEXTS = "agentique:contexts"
_STATE_KEY_CTX_TASKS_PREFIX = "agentique:ctx_tasks:"


class ContextManager:
    """Manages MCP session ↔ A2A context lifecycle.

    Thread-safe via asyncio locks. Provides context resolution,
    binding, and cleanup operations.

    When a FastMCP ``ctx`` is available, session↔context and
    context↔task mappings are also persisted via ``ctx.set_state``/
    ``ctx.get_state`` so they survive gateway restarts and work
    across horizontally-scaled replicas.  In-memory ``ContextMapping``
    is kept as a fast cache and fallback for code paths without ``ctx``.
    """

    def __init__(self) -> None:
        self._mapping = ContextMapping()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers for session-state persistence
    # ------------------------------------------------------------------

    async def _read_persisted_contexts(self, ctx: Any) -> list[str]:
        """Read the persisted context list from session state."""
        get_state = getattr(ctx, "get_state", None)
        if callable(get_state):
            try:
                result = await get_state(_STATE_KEY_CONTEXTS)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        return []

    async def _write_persisted_contexts(
        self, ctx: Any, contexts: list[str]
    ) -> None:
        """Write the context list back to session state."""
        set_state = getattr(ctx, "set_state", None)
        if callable(set_state):
            try:
                await set_state(_STATE_KEY_CONTEXTS, contexts)
            except Exception:
                pass

    async def _append_persisted_context(
        self, ctx: Any, context_id: str, existing: list[str]
    ) -> None:
        """Append *context_id* to the persisted list if not already there."""
        if context_id not in existing:
            await self._write_persisted_contexts(ctx, [*existing, context_id])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve_context(
        self,
        *,
        session_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        ctx: Any = None,
    ) -> str:
        """Resolve or create a context ID.

        If *context_id* is provided, bind it to the session and return it.
        If only *session_id* is given, return the most recent context or
        create a new one derived from *task_id* (or session_id).

        When *ctx* (a FastMCP ``Context``) is supplied, the resolved
        context ID is persisted in session state so it survives restarts.
        """
        # Read persisted contexts outside the lock to minimise lock hold time
        persisted: list[str] = []
        if ctx is not None:
            persisted = await self._read_persisted_contexts(ctx)

        async with self._lock:
            if context_id:
                if session_id:
                    self._mapping.bind(session_id, context_id)
                if ctx is not None:
                    # Persist outside lock body but we already hold it — safe
                    # because _append_persisted_context is idempotent and its
                    # I/O doesn't depend on internal state.
                    pass  # handled after the lock block
            else:
                # Try persisted contexts first (cross-restart continuity)
                if persisted:
                    return max(persisted)

                # Try in-memory mapping
                if session_id:
                    existing = self._mapping.get_contexts(session_id)
                    if existing:
                        return max(existing)

                # Create a new context ID
                context_id = task_id or session_id or _generate_id()
                if session_id:
                    self._mapping.bind(session_id, context_id)

        # Persist outside lock to avoid holding it across async I/O
        if ctx is not None and context_id:
            await self._append_persisted_context(ctx, context_id, persisted)

        return context_id  # type: ignore[return-value]

    async def bind(self, session_id: str, context_id: str) -> None:
        """Explicitly bind a context to a session."""
        async with self._lock:
            self._mapping.bind(session_id, context_id)

    async def track_task(
        self,
        context_id: str,
        task_id: str,
        ctx: Any = None,
    ) -> None:
        """Associate a task ID with its context.

        When *ctx* is provided, also persists the task mapping in session
        state under ``agentique:ctx_tasks:{context_id}``.
        """
        async with self._lock:
            self._mapping.track_task(context_id, task_id)

        # Persist task list outside the lock
        if ctx is not None:
            state_key = f"{_STATE_KEY_CTX_TASKS_PREFIX}{context_id}"
            get_state = getattr(ctx, "get_state", None)
            set_state = getattr(ctx, "set_state", None)
            if callable(get_state) and callable(set_state):
                try:
                    existing_tasks: list[str] = (await get_state(state_key)) or []
                    if task_id not in existing_tasks:
                        await set_state(state_key, [*existing_tasks, task_id])
                except Exception:
                    pass

    async def get_tasks(self, context_id: str) -> list[str]:
        """Return all task IDs for a given context."""
        async with self._lock:
            return self._mapping.get_tasks(context_id)

    async def get_session(self, context_id: str) -> str | None:
        """Look up the MCP session for a context."""
        async with self._lock:
            return self._mapping.get_session(context_id)

    async def cleanup_session(self, session_id: str, ctx: Any = None) -> None:
        """Remove all mappings for a disconnected MCP session.

        When *ctx* is provided, also removes the persisted context list
        from session state.
        """
        async with self._lock:
            self._mapping.unbind_session(session_id)

        if ctx is not None:
            delete_state = getattr(ctx, "delete_state", None)
            if callable(delete_state):
                try:
                    await delete_state(_STATE_KEY_CONTEXTS)
                except Exception:
                    pass

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
