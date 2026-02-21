"""Tests for Focus #9b: Artifact TTL/Eviction.

Covers:
  - capture_artifact() records created_at timestamp
  - evict_artifacts() removes entries older than threshold
  - evict_artifacts() keeps entries newer than threshold
  - evict_all_expired_artifacts() works across all tasks
  - evict_all_expired_artifacts() no-ops when artifact_ttl is None
  - TaskManager accepts artifact_ttl param
  - make_artifact_cleanup_lifespan() creates a valid lifespan
  - create_server() accepts artifact_ttl param
"""

from __future__ import annotations

import time
import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio

from agentique.bridge.task_manager import TaskManager
from agentique.bridge.storage import InMemoryTaskStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager(artifact_ttl: float | None = None) -> TaskManager:
    return TaskManager(store=InMemoryTaskStore(), artifact_ttl=artifact_ttl)


# ---------------------------------------------------------------------------
# capture_artifact() — timestamp recording
# ---------------------------------------------------------------------------


def test_capture_artifact_records_created_at():
    """Captured artifacts must have a created_at timestamp."""
    mgr = make_manager()
    before = time.monotonic()
    mgr.capture_artifact("t1", "a1", "content")
    after = time.monotonic()
    meta = mgr.get_artifact_metadata("t1", "a1")
    assert meta is not None
    assert "created_at" in meta
    assert before <= meta["created_at"] <= after


def test_capture_artifact_created_at_is_float():
    mgr = make_manager()
    mgr.capture_artifact("t1", "a1", "content")
    meta = mgr.get_artifact_metadata("t1", "a1")
    assert isinstance(meta["created_at"], float)


def test_list_artifacts_omits_created_at():
    """list_artifacts() should NOT expose created_at (implementation detail)."""
    mgr = make_manager()
    mgr.capture_artifact("t1", "a1", "content")
    artifacts = mgr.list_artifacts("t1")
    for a in artifacts:
        assert "created_at" not in a


# ---------------------------------------------------------------------------
# evict_artifacts()
# ---------------------------------------------------------------------------


def test_evict_artifacts_removes_old_entries():
    """evict_artifacts() removes entries with created_at < before."""
    mgr = make_manager()
    mgr.capture_artifact("t1", "old", "old content")

    # Force created_at to be in the past
    old_ts = time.monotonic() - 1000
    mgr._artifacts["t1"]["old"]["created_at"] = old_ts

    before = time.monotonic()
    evicted = mgr.evict_artifacts("t1", before=before)
    assert evicted == 1
    assert mgr.get_artifact("t1", "old") is None


def test_evict_artifacts_keeps_new_entries():
    """evict_artifacts() does not remove entries newer than the threshold."""
    mgr = make_manager()
    mgr.capture_artifact("t1", "new", "new content")

    # Threshold is in the past (all artifacts are newer)
    before = time.monotonic() - 1000
    evicted = mgr.evict_artifacts("t1", before=before)
    assert evicted == 0
    assert mgr.get_artifact("t1", "new") is not None


def test_evict_artifacts_mixed_keeps_only_new():
    """evict_artifacts() removes only old artifacts, keeps new ones."""
    mgr = make_manager()
    mgr.capture_artifact("t1", "old", "old content")
    mgr.capture_artifact("t1", "new", "new content")

    # Force the "old" artifact to be 1000 seconds in the past
    old_ts = time.monotonic() - 1000
    mgr._artifacts["t1"]["old"]["created_at"] = old_ts

    # Set the threshold 100 seconds in the past: between old (-1000s) and new (~0s)
    before = time.monotonic() - 100
    evicted = mgr.evict_artifacts("t1", before=before)
    assert evicted == 1
    assert mgr.get_artifact("t1", "old") is None
    assert mgr.get_artifact("t1", "new") is not None


def test_evict_artifacts_empty_task_returns_zero():
    mgr = make_manager()
    evicted = mgr.evict_artifacts("nonexistent", before=time.monotonic())
    assert evicted == 0


def test_evict_artifacts_removes_task_dict_when_empty():
    """When all artifacts for a task are evicted, the task entry is removed."""
    mgr = make_manager()
    mgr.capture_artifact("t1", "a1", "content")
    mgr._artifacts["t1"]["a1"]["created_at"] = 0.0  # epoch — always old
    mgr.evict_artifacts("t1", before=time.monotonic())
    assert "t1" not in mgr._artifacts


# ---------------------------------------------------------------------------
# evict_all_expired_artifacts()
# ---------------------------------------------------------------------------


def test_evict_all_no_ttl_returns_zero():
    """Without artifact_ttl, evict_all returns 0 and does nothing."""
    mgr = make_manager(artifact_ttl=None)
    mgr.capture_artifact("t1", "a1", "content")
    assert mgr.evict_all_expired_artifacts() == 0
    # Artifact should still be there
    assert mgr.get_artifact("t1", "a1") is not None


def test_evict_all_with_ttl_removes_old():
    """With artifact_ttl set, evict_all removes expired artifacts."""
    mgr = make_manager(artifact_ttl=60.0)  # 60 second TTL
    mgr.capture_artifact("t1", "old", "old content")
    mgr._artifacts["t1"]["old"]["created_at"] = 0.0  # epoch — way expired
    evicted = mgr.evict_all_expired_artifacts()
    assert evicted == 1
    assert mgr.get_artifact("t1", "old") is None


def test_evict_all_with_ttl_keeps_recent():
    """With artifact_ttl set, evict_all keeps recent artifacts."""
    mgr = make_manager(artifact_ttl=3600.0)  # 1 hour TTL
    mgr.capture_artifact("t1", "recent", "recent content")
    evicted = mgr.evict_all_expired_artifacts()
    assert evicted == 0
    assert mgr.get_artifact("t1", "recent") is not None


def test_evict_all_across_multiple_tasks():
    """evict_all evicts from all tasks, not just one."""
    mgr = make_manager(artifact_ttl=60.0)
    for i in range(3):
        mgr.capture_artifact(f"t{i}", "old", "content")
        mgr._artifacts[f"t{i}"]["old"]["created_at"] = 0.0

    evicted = mgr.evict_all_expired_artifacts()
    assert evicted == 3


# ---------------------------------------------------------------------------
# TaskManager(artifact_ttl=...) constructor
# ---------------------------------------------------------------------------


def test_task_manager_accepts_artifact_ttl():
    mgr = TaskManager(artifact_ttl=300.0)
    assert mgr._artifact_ttl == 300.0


def test_task_manager_default_artifact_ttl_none():
    mgr = TaskManager()
    assert mgr._artifact_ttl is None


# ---------------------------------------------------------------------------
# make_artifact_cleanup_lifespan()
# ---------------------------------------------------------------------------


def test_make_artifact_cleanup_lifespan_returns_lifespan():
    """make_artifact_cleanup_lifespan() returns a non-None lifespan object."""
    from agentique.bridge.lifespans import make_artifact_cleanup_lifespan

    mgr = make_manager(artifact_ttl=60.0)
    ls = make_artifact_cleanup_lifespan(mgr, ttl=60.0)
    assert ls is not None


def test_make_artifact_cleanup_lifespan_default_interval():
    """Default interval is max(ttl / 2, 60)."""
    from agentique.bridge.lifespans import make_artifact_cleanup_lifespan

    mgr = make_manager(artifact_ttl=200.0)
    # Just verify it creates without error
    ls = make_artifact_cleanup_lifespan(mgr, ttl=200.0)
    assert ls is not None


def test_make_artifact_cleanup_lifespan_custom_interval():
    """Custom interval is accepted."""
    from agentique.bridge.lifespans import make_artifact_cleanup_lifespan

    mgr = make_manager()
    ls = make_artifact_cleanup_lifespan(mgr, ttl=60.0, interval=30.0)
    assert ls is not None


# ---------------------------------------------------------------------------
# create_server() accepts artifact_ttl
# ---------------------------------------------------------------------------


def test_create_server_accepts_artifact_ttl_none():
    from agentique.core.types import AgentInfo
    from agentique.server import create_server

    server = create_server(
        agents=[AgentInfo(name="test", base_url="http://test.example.com")],
        artifact_ttl=None,
    )
    assert server is not None


def test_create_server_accepts_artifact_ttl_value():
    from agentique.core.types import AgentInfo
    from agentique.server import create_server

    server = create_server(
        agents=[AgentInfo(name="test", base_url="http://test.example.com")],
        artifact_ttl=3600.0,
    )
    assert server is not None


def test_make_artifact_cleanup_lifespan_importable_from_agentique():
    from agentique import make_artifact_cleanup_lifespan
    assert callable(make_artifact_cleanup_lifespan)
