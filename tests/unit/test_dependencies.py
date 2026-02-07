"""Tests for the dependency injection module."""

from __future__ import annotations

import pytest

from agentique.bridge.dependencies import (
    clear,
    configure,
    get_adapter,
    get_config,
    get_context_manager,
    get_emitter,
    get_router,
    get_task_manager,
)
from agentique.bridge.context_manager import ContextManager
from agentique.bridge.router import AgentRouter
from agentique.bridge.storage import InMemoryTaskStore
from agentique.bridge.task_manager import TaskManager
from agentique.core.config import AgentiqueConfig
from agentique.core.events import AsyncEventEmitter
from agentique.core.types import AgentInfo


# ---- Setup / teardown ----


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure a clean registry for each test."""
    clear()
    yield
    clear()


# ---- Tests ----


def test_configure_and_get_router():
    """configure() should register a router that get_router() returns."""
    router = AgentRouter([
        AgentInfo(name="a1", base_url="http://localhost:9001"),
    ])
    configure(router=router)
    assert get_router() is router


def test_configure_and_get_adapter():
    """configure() should register an adapter that get_adapter() returns."""
    adapter = object()
    configure(adapter=adapter)
    assert get_adapter() is adapter


def test_configure_and_get_task_manager():
    """configure() should register a task manager."""
    tasks = TaskManager(store=InMemoryTaskStore())
    configure(task_manager=tasks)
    assert get_task_manager() is tasks


def test_configure_and_get_config():
    """configure() should register config."""
    config = AgentiqueConfig(name="test")
    configure(config=config)
    result = get_config()
    assert result.name == "test"


def test_configure_and_get_emitter():
    """configure() should register an event emitter."""
    emitter = AsyncEventEmitter()
    configure(emitter=emitter)
    assert get_emitter() is emitter


def test_configure_and_get_context_manager():
    """configure() should register a context manager."""
    ctx_mgr = ContextManager()
    configure(context_manager=ctx_mgr)
    assert get_context_manager() is ctx_mgr


def test_get_router_raises_when_not_configured():
    """get_router() should raise RuntimeError when not configured."""
    with pytest.raises(RuntimeError, match="AgentRouter not configured"):
        get_router()


def test_get_adapter_raises_when_not_configured():
    """get_adapter() should raise RuntimeError when not configured."""
    with pytest.raises(RuntimeError, match="Adapter not configured"):
        get_adapter()


def test_get_task_manager_raises_when_not_configured():
    """get_task_manager() should raise RuntimeError when not configured."""
    with pytest.raises(RuntimeError, match="TaskManager not configured"):
        get_task_manager()


def test_get_config_raises_when_not_configured():
    """get_config() should raise RuntimeError when not configured."""
    with pytest.raises(RuntimeError, match="AgentiqueConfig not configured"):
        get_config()


def test_get_emitter_raises_when_not_configured():
    """get_emitter() should raise RuntimeError when not configured."""
    with pytest.raises(RuntimeError, match="AsyncEventEmitter not configured"):
        get_emitter()


def test_get_context_manager_raises_when_not_configured():
    """get_context_manager() should raise RuntimeError when not configured."""
    with pytest.raises(RuntimeError, match="ContextManager not configured"):
        get_context_manager()


def test_clear_resets_registry():
    """clear() should remove all registered dependencies."""
    configure(
        router=AgentRouter([]),
        adapter=object(),
        config=AgentiqueConfig(),
        emitter=AsyncEventEmitter(),
    )
    # Verify they work
    get_router()
    get_adapter()
    get_config()
    get_emitter()

    clear()

    with pytest.raises(RuntimeError):
        get_router()
    with pytest.raises(RuntimeError):
        get_adapter()


def test_configure_partial():
    """configure() should accept partial registration."""
    configure(router=AgentRouter([]))
    get_router()  # works

    with pytest.raises(RuntimeError):
        get_adapter()  # not registered


def test_configure_overwrites():
    """configure() should overwrite previously registered dependencies."""
    router1 = AgentRouter([])
    router2 = AgentRouter([
        AgentInfo(name="x", base_url="http://localhost:9999"),
    ])
    configure(router=router1)
    assert get_router() is router1

    configure(router=router2)
    assert get_router() is router2
