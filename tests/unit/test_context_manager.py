"""Tests for ContextManager and ContextMapping."""

from __future__ import annotations

import pytest

from agentique.bridge.context_manager import ContextManager
from agentique.core.types import ContextMapping


# ---- ContextMapping ----


def test_mapping_bind_and_lookup():
    m = ContextMapping()
    m.bind("session-1", "ctx-a")
    assert m.get_session("ctx-a") == "session-1"
    assert "ctx-a" in m.get_contexts("session-1")


def test_mapping_multiple_contexts():
    m = ContextMapping()
    m.bind("s1", "c1")
    m.bind("s1", "c2")
    contexts = m.get_contexts("s1")
    assert contexts == {"c1", "c2"}


def test_mapping_track_tasks():
    m = ContextMapping()
    m.track_task("ctx-1", "task-a")
    m.track_task("ctx-1", "task-b")
    assert m.get_tasks("ctx-1") == ["task-a", "task-b"]


def test_mapping_unbind_session():
    m = ContextMapping()
    m.bind("s1", "c1")
    m.bind("s1", "c2")
    m.track_task("c1", "t1")
    m.unbind_session("s1")
    assert m.get_session("c1") is None
    assert m.get_contexts("s1") == set()
    assert m.get_tasks("c1") == []


def test_mapping_empty_lookups():
    m = ContextMapping()
    assert m.get_session("nonexistent") is None
    assert m.get_contexts("nonexistent") == set()
    assert m.get_tasks("nonexistent") == []


# ---- ContextManager (async) ----


@pytest.mark.asyncio
async def test_ctx_mgr_resolve_with_explicit_context():
    mgr = ContextManager()
    ctx_id = await mgr.resolve_context(session_id="s1", context_id="c1")
    assert ctx_id == "c1"

    # Session should now be bound
    session = await mgr.get_session("c1")
    assert session == "s1"


@pytest.mark.asyncio
async def test_ctx_mgr_resolve_reuses_existing():
    mgr = ContextManager()
    # First call creates binding
    await mgr.resolve_context(session_id="s1", context_id="c1")
    # Second call without context_id should return existing
    ctx_id = await mgr.resolve_context(session_id="s1")
    assert ctx_id == "c1"


@pytest.mark.asyncio
async def test_ctx_mgr_resolve_creates_new():
    mgr = ContextManager()
    ctx_id = await mgr.resolve_context(session_id="s1", task_id="t1")
    assert ctx_id == "t1"


@pytest.mark.asyncio
async def test_ctx_mgr_track_tasks():
    mgr = ContextManager()
    await mgr.track_task("c1", "t1")
    await mgr.track_task("c1", "t2")
    tasks = await mgr.get_tasks("c1")
    assert tasks == ["t1", "t2"]


@pytest.mark.asyncio
async def test_ctx_mgr_cleanup():
    mgr = ContextManager()
    await mgr.bind("s1", "c1")
    await mgr.cleanup_session("s1")
    session = await mgr.get_session("c1")
    assert session is None


@pytest.mark.asyncio
async def test_ctx_mgr_validate_consistent():
    mgr = ContextManager()
    await mgr.track_task("c1", "t1")
    assert mgr.validate_context_task("c1", "t1")
    assert not mgr.validate_context_task("c1", "t999")


@pytest.mark.asyncio
async def test_ctx_mgr_validate_new_context():
    mgr = ContextManager()
    # No tasks tracked — should allow any task
    assert mgr.validate_context_task("new-ctx", "any-task")
    assert mgr.validate_context_task(None, "any-task")
