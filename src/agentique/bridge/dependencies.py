"""Dependency injection factories for agentique tools.

Provides ``Depends()``-compatible factory functions that can be used
as defaults in FastMCP tool parameters.  Each factory returns a
component from the server's dependency registry, enabling clean
separation of concerns and easier testing.

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

The registry is populated by ``create_server()`` during server
construction and reset via ``clear()`` for testing.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from agentique.core.config import AgentiqueConfig
from agentique.core.events import AsyncEventEmitter
from agentique.bridge.router import AgentRouter
from agentique.bridge.task_manager import TaskManager
from agentique.bridge.context_manager import ContextManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level registry
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
# Dependency factories — each is a ``Depends()``-compatible callable
# ---------------------------------------------------------------------------


def get_router() -> AgentRouter:
    """Return the ``AgentRouter`` registered during server construction."""
    router = _registry.get("router")
    if router is None:
        raise RuntimeError(
            "AgentRouter not configured. "
            "Ensure create_server() has been called."
        )
    return router


def get_adapter() -> Any:
    """Return the agent adapter registered during server construction."""
    adapter = _registry.get("adapter")
    if adapter is None:
        raise RuntimeError(
            "Adapter not configured. "
            "Ensure create_server() has been called."
        )
    return adapter


def get_task_manager() -> TaskManager:
    """Return the ``TaskManager`` registered during server construction."""
    tasks = _registry.get("task_manager")
    if tasks is None:
        raise RuntimeError(
            "TaskManager not configured. "
            "Ensure create_server() has been called."
        )
    return tasks


def get_config() -> AgentiqueConfig:
    """Return the ``AgentiqueConfig`` registered during server construction."""
    config = _registry.get("config")
    if config is None:
        raise RuntimeError(
            "AgentiqueConfig not configured. "
            "Ensure create_server() has been called."
        )
    return config


def get_emitter() -> AsyncEventEmitter:
    """Return the ``AsyncEventEmitter`` registered during server construction."""
    emitter = _registry.get("emitter")
    if emitter is None:
        raise RuntimeError(
            "AsyncEventEmitter not configured. "
            "Ensure create_server() has been called."
        )
    return emitter


def get_context_manager() -> ContextManager:
    """Return the ``ContextManager`` registered during server construction."""
    ctx_mgr = _registry.get("context_manager")
    if ctx_mgr is None:
        raise RuntimeError(
            "ContextManager not configured. "
            "Ensure create_server() has been called."
        )
    return ctx_mgr
