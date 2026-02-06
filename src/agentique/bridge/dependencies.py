"""Dependency injection factories for agentique tools.

Provides ``Depends()``-compatible factory functions that can be used
as defaults in FastMCP tool parameters.  Each factory returns a
component from the server's dependency registry, enabling clean
separation of concerns and easier testing.

Supports two scopes:

- **Server scope** (default): Components registered via ``configure()``
  are shared across all sessions.
- **Session scope**: Per-session overrides stored via
  ``set_session_override()`` using FastMCP's ``ctx.set_state()``
  mechanism.  This enables multi-tenant configurations where
  different sessions get different adapters or routing strategies.

Usage in tools::

    from fastmcp.dependencies import Depends
    from agentique.bridge.dependencies import get_router, get_adapter

    @mcp.tool()
    async def my_tool(
        message: str,
        router: AgentRouter = Depends(get_router),
        adapter: Any = Depends(get_adapter),
    ):
        resolved = router.resolve(message=message)
        ...

Session-scoped overrides::

    # Inside a tool with access to ctx:
    from agentique.bridge.dependencies import set_session_override

    await set_session_override(ctx, "router", my_custom_router)
    # Subsequent tool calls in this session will use my_custom_router
"""

from __future__ import annotations

import logging
from typing import Any

from agentique.core.config import AgentiqueConfig
from agentique.core.events import AsyncEventEmitter
from agentique.bridge.router import AgentRouter
from agentique.bridge.task_manager import TaskManager
from agentique.bridge.context_manager import ContextManager

logger = logging.getLogger(__name__)

_SESSION_KEY_PREFIX = "_agentique_dep_"


# ---------------------------------------------------------------------------
# Module-level registry (server scope)
# ---------------------------------------------------------------------------

_registry: dict[str, Any] = {}


def configure(
    *,
    router: AgentRouter | None = None,
    adapter: Any | None = None,
    task_manager: TaskManager | None = None,
    config: AgentiqueConfig | None = None,
    emitter: AsyncEventEmitter | None = None,
    context_manager: ContextManager | None = None,
) -> None:
    """Populate the dependency registry.

    Called by ``create_server()`` after all components are built.
    """
    if router is not None:
        _registry["router"] = router
    if adapter is not None:
        _registry["adapter"] = adapter
    if task_manager is not None:
        _registry["task_manager"] = task_manager
    if config is not None:
        _registry["config"] = config
    if emitter is not None:
        _registry["emitter"] = emitter
    if context_manager is not None:
        _registry["context_manager"] = context_manager


def clear() -> None:
    """Reset the registry (useful for testing)."""
    _registry.clear()


# ---------------------------------------------------------------------------
# Session-scoped overrides via FastMCP Context state
# ---------------------------------------------------------------------------


async def set_session_override(ctx: Any, key: str, value: Any) -> None:
    """Store a per-session dependency override.

    The override will be used by subsequent ``get_*`` calls within
    the same MCP session, taking precedence over the server-scope
    registry.

    Args:
        ctx: The FastMCP ``Context`` instance.
        key: Registry key (e.g. ``"router"``, ``"adapter"``).
        value: The dependency to use for this session.
    """
    state_key = f"{_SESSION_KEY_PREFIX}{key}"
    set_state = getattr(ctx, "set_state", None)
    if callable(set_state):
        await set_state(state_key, value)
    else:
        logger.debug("Context does not support set_state; session override ignored")


async def get_session_override(ctx: Any, key: str) -> Any | None:
    """Retrieve a per-session dependency override, or None."""
    state_key = f"{_SESSION_KEY_PREFIX}{key}"
    get_state = getattr(ctx, "get_state", None)
    if callable(get_state):
        try:
            return await get_state(state_key)
        except Exception:
            return None
    return None


async def clear_session_overrides(ctx: Any) -> None:
    """Remove all session overrides for the current session.

    Args:
        ctx: The FastMCP ``Context`` instance.
    """
    for key in ("router", "adapter", "task_manager", "config", "emitter", "context_manager"):
        state_key = f"{_SESSION_KEY_PREFIX}{key}"
        delete_state = getattr(ctx, "delete_state", None)
        if callable(delete_state):
            try:
                await delete_state(state_key)
            except Exception:
                pass


def _resolve(key: str, label: str) -> Any:
    """Resolve from server-scope registry (sync path)."""
    value = _registry.get(key)
    if value is None:
        raise RuntimeError(
            f"{label} not configured. "
            "Ensure create_server() has been called."
        )
    return value


# ---------------------------------------------------------------------------
# Dependency factories — each is a ``Depends()``-compatible callable
# ---------------------------------------------------------------------------


def get_router() -> AgentRouter:
    """Return the ``AgentRouter`` registered during server construction."""
    return _resolve("router", "AgentRouter")


def get_adapter() -> Any:
    """Return the agent adapter registered during server construction."""
    return _resolve("adapter", "Adapter")


def get_task_manager() -> TaskManager:
    """Return the ``TaskManager`` registered during server construction."""
    return _resolve("task_manager", "TaskManager")


def get_config() -> AgentiqueConfig:
    """Return the ``AgentiqueConfig`` registered during server construction."""
    return _resolve("config", "AgentiqueConfig")


def get_emitter() -> AsyncEventEmitter:
    """Return the ``AsyncEventEmitter`` registered during server construction."""
    return _resolve("emitter", "AsyncEventEmitter")


def get_context_manager() -> ContextManager:
    """Return the ``ContextManager`` registered during server construction."""
    return _resolve("context_manager", "ContextManager")


# ---------------------------------------------------------------------------
# Session-aware factories (async — use inside tools with ctx access)
# ---------------------------------------------------------------------------


async def get_session_router(ctx: Any) -> AgentRouter:
    """Return session-scoped router override, or server-scope default."""
    override = await get_session_override(ctx, "router")
    if override is not None:
        return override
    return get_router()


async def get_session_adapter(ctx: Any) -> Any:
    """Return session-scoped adapter override, or server-scope default."""
    override = await get_session_override(ctx, "adapter")
    if override is not None:
        return override
    return get_adapter()


async def get_session_config(ctx: Any) -> AgentiqueConfig:
    """Return session-scoped config override, or server-scope default."""
    override = await get_session_override(ctx, "config")
    if override is not None:
        return override
    return get_config()
