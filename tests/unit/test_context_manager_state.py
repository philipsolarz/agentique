"""Tests for ContextManager session-state persistence (Focus #5 Item A).

Verifies that resolve_context(), track_task(), and cleanup_session()
correctly integrate with FastMCP ctx.get_state / ctx.set_state /
ctx.delete_state, with graceful degradation when ctx is absent or broken.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentique.bridge.context_manager import (
    ContextManager,
    _STATE_KEY_CONTEXTS,
    _STATE_KEY_CTX_TASKS_PREFIX,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ctx(
    *,
    get_state_return: object = None,
    get_state_side_effect: object = None,
    set_state_side_effect: object = None,
    delete_state_side_effect: object = None,
) -> MagicMock:
    """Return a mock FastMCP Context with async state methods."""
    ctx = MagicMock()
    if get_state_side_effect is not None:
        ctx.get_state = AsyncMock(side_effect=get_state_side_effect)
    else:
        ctx.get_state = AsyncMock(return_value=get_state_return)
    if set_state_side_effect is not None:
        ctx.set_state = AsyncMock(side_effect=set_state_side_effect)
    else:
        ctx.set_state = AsyncMock(return_value=None)
    if delete_state_side_effect is not None:
        ctx.delete_state = AsyncMock(side_effect=delete_state_side_effect)
    else:
        ctx.delete_state = AsyncMock(return_value=None)
    return ctx


# ---------------------------------------------------------------------------
# resolve_context — no ctx (in-memory fallback)
# ---------------------------------------------------------------------------


async def test_resolve_context_without_ctx_uses_memory():
    mgr = ContextManager()
    ctx_id = await mgr.resolve_context(session_id="sess-1")
    assert ctx_id == "sess-1"
    # Second call returns the same context
    ctx_id2 = await mgr.resolve_context(session_id="sess-1")
    assert ctx_id2 == ctx_id


async def test_resolve_context_without_ctx_explicit_context_id():
    mgr = ContextManager()
    ctx_id = await mgr.resolve_context(
        session_id="sess-1", context_id="explicit-ctx"
    )
    assert ctx_id == "explicit-ctx"


# ---------------------------------------------------------------------------
# resolve_context — with ctx, no prior state
# ---------------------------------------------------------------------------


async def test_resolve_context_with_ctx_stores_in_state():
    """When ctx is present and no prior state, the new ctx_id is persisted."""
    mgr = ContextManager()
    ctx = make_ctx(get_state_return=None)

    ctx_id = await mgr.resolve_context(session_id="sess-1", ctx=ctx)

    # set_state must be called with the canonical key and a non-empty list
    ctx.set_state.assert_awaited_once()
    call_args = ctx.set_state.await_args
    assert call_args.args[0] == _STATE_KEY_CONTEXTS
    persisted_list = call_args.args[1]
    assert isinstance(persisted_list, list)
    assert ctx_id in persisted_list


async def test_resolve_context_with_ctx_explicit_context_id_persists():
    """Explicit context_id is also persisted in session state."""
    mgr = ContextManager()
    ctx = make_ctx(get_state_return=None)

    ctx_id = await mgr.resolve_context(
        session_id="sess-2", context_id="explicit-ctx", ctx=ctx
    )
    assert ctx_id == "explicit-ctx"

    ctx.set_state.assert_awaited_once()
    _, kwargs = ctx.set_state.call_args
    # call was positional
    call_args = ctx.set_state.await_args
    assert "explicit-ctx" in call_args.args[1]


# ---------------------------------------------------------------------------
# resolve_context — cross-restart continuity via persisted state
# ---------------------------------------------------------------------------


async def test_resolve_context_cross_restart_continuity():
    """Memory is empty but session state has a context — return it."""
    mgr = ContextManager()
    # Simulate a prior run that persisted "old-ctx"
    ctx = make_ctx(get_state_return=["old-ctx"])

    ctx_id = await mgr.resolve_context(session_id="sess-new", ctx=ctx)
    assert ctx_id == "old-ctx"


async def test_resolve_context_persisted_list_not_overwritten_if_already_present():
    """If the resolved ctx_id is already in the persisted list, set_state
    should NOT be called again (idempotent)."""
    mgr = ContextManager()
    ctx = make_ctx(get_state_return=["already-ctx"])

    ctx_id = await mgr.resolve_context(session_id="sess-x", ctx=ctx)
    assert ctx_id == "already-ctx"
    # No write needed — already in list
    ctx.set_state.assert_not_awaited()


# ---------------------------------------------------------------------------
# resolve_context — graceful degradation on errors
# ---------------------------------------------------------------------------


async def test_resolve_context_ctx_get_state_error_falls_back_gracefully():
    """Exception in ctx.get_state must not raise; fall back to in-memory."""
    mgr = ContextManager()
    ctx = make_ctx(get_state_side_effect=RuntimeError("redis down"))
    # set_state also raises — we must still not crash
    ctx.set_state = AsyncMock(side_effect=RuntimeError("redis down"))

    ctx_id = await mgr.resolve_context(session_id="sess-err", ctx=ctx)
    # Falls back to session_id as the context
    assert ctx_id == "sess-err"


async def test_resolve_context_ctx_missing_state_methods():
    """An object without get_state / set_state is treated the same as no ctx."""
    mgr = ContextManager()
    ctx = MagicMock(spec=[])  # no methods at all

    ctx_id = await mgr.resolve_context(session_id="sess-bare", ctx=ctx)
    assert ctx_id == "sess-bare"


# ---------------------------------------------------------------------------
# track_task — with and without ctx
# ---------------------------------------------------------------------------


async def test_track_task_without_ctx_falls_back_to_memory():
    mgr = ContextManager()
    await mgr.resolve_context(session_id="sess-1", context_id="ctx-1")
    await mgr.track_task("ctx-1", "task-1")
    tasks = await mgr.get_tasks("ctx-1")
    assert "task-1" in tasks


async def test_track_task_with_ctx_persists_task_id():
    """track_task writes the task ID to session state under the ctx key."""
    mgr = ContextManager()
    ctx = make_ctx(get_state_return=None)

    await mgr.track_task("ctx-1", "task-42", ctx=ctx)

    expected_key = f"{_STATE_KEY_CTX_TASKS_PREFIX}ctx-1"
    ctx.set_state.assert_awaited_once()
    call_args = ctx.set_state.await_args
    assert call_args.args[0] == expected_key
    assert "task-42" in call_args.args[1]


async def test_track_task_with_ctx_appends_to_existing_tasks():
    """Existing task list in state is extended, not replaced."""
    mgr = ContextManager()
    ctx = make_ctx(get_state_return=["task-old"])

    await mgr.track_task("ctx-2", "task-new", ctx=ctx)

    call_args = ctx.set_state.await_args
    task_list = call_args.args[1]
    assert "task-old" in task_list
    assert "task-new" in task_list


async def test_track_task_with_ctx_idempotent():
    """If the task_id is already in state, set_state is not called."""
    mgr = ContextManager()
    ctx = make_ctx(get_state_return=["task-42"])

    await mgr.track_task("ctx-3", "task-42", ctx=ctx)

    ctx.set_state.assert_not_awaited()


async def test_track_task_ctx_error_does_not_raise():
    """Errors in ctx.set_state are silenced gracefully."""
    mgr = ContextManager()
    ctx = make_ctx(get_state_side_effect=RuntimeError("boom"))

    # Should not raise
    await mgr.track_task("ctx-x", "task-x", ctx=ctx)


# ---------------------------------------------------------------------------
# cleanup_session — with and without ctx
# ---------------------------------------------------------------------------


async def test_cleanup_session_without_ctx_removes_memory_binding():
    mgr = ContextManager()
    await mgr.resolve_context(session_id="sess-1", context_id="ctx-1")
    await mgr.cleanup_session("sess-1")
    # After cleanup, a new resolve_context creates a fresh context
    ctx_id = await mgr.resolve_context(session_id="sess-1")
    assert ctx_id == "sess-1"  # defaults to session_id when no memory


async def test_cleanup_session_with_ctx_deletes_state():
    """cleanup_session calls ctx.delete_state for the contexts key."""
    mgr = ContextManager()
    ctx = make_ctx()

    await mgr.cleanup_session("sess-1", ctx=ctx)

    ctx.delete_state.assert_awaited_once_with(_STATE_KEY_CONTEXTS)


async def test_cleanup_session_ctx_delete_error_does_not_raise():
    """Exception in ctx.delete_state is suppressed silently."""
    mgr = ContextManager()
    ctx = make_ctx(delete_state_side_effect=RuntimeError("redis down"))

    # Must not raise
    await mgr.cleanup_session("sess-1", ctx=ctx)


# ---------------------------------------------------------------------------
# State key name
# ---------------------------------------------------------------------------


def test_state_key_constants():
    """Verify the well-known state key names have not changed."""
    assert _STATE_KEY_CONTEXTS == "agentique:contexts"
    assert _STATE_KEY_CTX_TASKS_PREFIX == "agentique:ctx_tasks:"
