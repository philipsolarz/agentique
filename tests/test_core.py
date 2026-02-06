"""Unit tests for core types, router, events, and config."""

from __future__ import annotations

import asyncio

import pytest

from agentique.core.types import (
    AgentEvent,
    AgentHierarchy,
    AgentInfo,
    BridgeContext,
    ContextMapping,
    StreamChunk,
    TaskState,
    TaskTracker,
)
from agentique.core.events import AsyncEventEmitter
from agentique.core.errors import AgentNotFoundError
from agentique.bridge.router import AgentRouter, KeywordRouter, DirectRouter


# ---- TaskState ----

def test_task_state_terminal():
    assert TaskState.completed.is_terminal
    assert TaskState.failed.is_terminal
    assert not TaskState.working.is_terminal
    assert not TaskState.submitted.is_terminal


def test_task_state_interruptible():
    assert TaskState.input_required.is_interruptible
    assert TaskState.auth_required.is_interruptible
    assert not TaskState.working.is_interruptible


# ---- TaskTracker ----

def test_tracker_transition():
    t = TaskTracker(task_id="t1")
    assert t.state == TaskState.submitted
    assert t.transition(TaskState.working)
    assert t.state == TaskState.working
    assert t.transition(TaskState.completed)
    assert t.state == TaskState.completed
    # Cannot transition from terminal
    assert not t.transition(TaskState.working)


def test_tracker_add_event():
    t = TaskTracker(task_id="t1")
    event = AgentEvent(kind="message", text="hello", progress=50.0)
    t.add_event(event)
    assert len(t.events) == 1
    assert t.progress == 50.0


# ---- AgentInfo ----

def test_agent_info_to_dict():
    info = AgentInfo(name="test", base_url="http://x", skills=("a", "b"))
    d = info.to_dict()
    assert d["name"] == "test"
    assert d["skills"] == ["a", "b"]


# ---- BridgeContext ----

def test_bridge_context_metadata():
    ctx = BridgeContext(session_id="s1", request_id="r1")
    meta = ctx.to_metadata()
    assert meta["session_id"] == "s1"
    assert meta["request_id"] == "r1"


def test_bridge_context_replace():
    ctx = BridgeContext(session_id="s1", request_id="r1")
    new_ctx = ctx.replace(session_id="s2")
    assert new_ctx.session_id == "s2"
    assert new_ctx.request_id == "r1"  # unchanged


def test_bridge_context_replace_conversation_history():
    ctx = BridgeContext(session_id="s1")
    history = [{"role": "user", "content": "hello"}]
    new_ctx = ctx.replace(conversation_history=history)
    assert new_ctx.conversation_history == history
    assert new_ctx.session_id == "s1"


# ---- AgentEvent ----

def test_event_properties():
    e = AgentEvent(kind="message", text="hi", branch="root.calc")
    assert e.is_content
    assert not e.is_status_update
    assert e.agent_path == ["root", "calc"]


def test_event_to_dict_minimal():
    e = AgentEvent(kind="status", text="working")
    d = e.to_dict()
    assert d["kind"] == "status"
    assert "task_id" not in d  # None fields omitted


# ---- StreamChunk ----

def test_chunk_from_event():
    e = AgentEvent(kind="message", text="hi", task_id="t1")
    c = StreamChunk.from_event("agent1", 0, e)
    assert c.agent == "agent1"
    assert c.task_id == "t1"


# ---- AgentHierarchy ----

def test_hierarchy_path():
    h = AgentHierarchy(root="root")
    h.add_agent("root")  # explicitly add root
    h.add_agent("calc", parent="root")
    h.add_agent("add", parent="calc")
    assert h.get_path("add") == ["root", "calc", "add"]
    assert h.agents["add"].depth == 2


# ---- ContextMapping ----

def test_context_mapping_basic():
    m = ContextMapping()
    m.bind("s1", "c1")
    m.bind("s1", "c2")
    assert m.get_session("c1") == "s1"
    assert m.get_contexts("s1") == {"c1", "c2"}


def test_context_mapping_tasks():
    m = ContextMapping()
    m.track_task("c1", "t1")
    m.track_task("c1", "t2")
    assert m.get_tasks("c1") == ["t1", "t2"]
    assert m.get_tasks("c_unknown") == []


# ---- Router ----

def test_router_register_and_resolve():
    r = AgentRouter()
    r.register(AgentInfo(name="a", base_url="http://a", skills=("math",)))
    r.register(AgentInfo(name="b", base_url="http://b", skills=("text",)))
    assert r.resolve(name="a").name == "a"
    assert r.resolve(skill="text").name == "b"
    # Default is first registered
    assert r.resolve().name == "a"


def test_router_unknown_agent():
    r = AgentRouter([AgentInfo(name="a", base_url="http://a")])
    with pytest.raises(AgentNotFoundError):
        r.describe("nonexistent")


def test_keyword_router():
    agents = [
        AgentInfo(name="calc", base_url="http://a", skills=("math", "calculator")),
        AgentInfo(name="text", base_url="http://b", skills=("nlp", "text")),
    ]
    kr = KeywordRouter()
    result = kr.select("do some math for me", agents)
    assert result.name == "calc"


def test_direct_router():
    agents = [
        AgentInfo(name="a", base_url="http://a"),
        AgentInfo(name="b", base_url="http://b"),
    ]
    dr = DirectRouter("b")
    assert dr.select("anything", agents).name == "b"


# ---- Events ----

@pytest.mark.asyncio
async def test_event_emitter_sync_handler():
    emitter = AsyncEventEmitter()
    results = []
    emitter.on("test", lambda x: results.append(x))
    await emitter.emit("test", 42)
    assert results == [42]


@pytest.mark.asyncio
async def test_event_emitter_async_handler():
    emitter = AsyncEventEmitter()
    results = []

    async def handler(x: int) -> None:
        results.append(x)

    emitter.on("test", handler)
    await emitter.emit("test", 99)
    assert results == [99]


@pytest.mark.asyncio
async def test_event_emitter_off():
    emitter = AsyncEventEmitter()
    results = []
    handler = lambda: results.append(1)
    emitter.on("e", handler)
    emitter.off("e", handler)
    await emitter.emit("e")
    assert results == []
