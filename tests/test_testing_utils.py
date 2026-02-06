"""Tests for the agentique.testing utilities."""

from __future__ import annotations

import pytest

from agentique.core.types import AgentInfo, BridgeContext
from agentique.testing import (
    MockAdapter,
    MockStreamingAdapter,
    RecordingMiddleware,
    mock_agent_card,
    mock_agent_info,
)


# ---- MockAdapter ----


@pytest.mark.asyncio
async def test_mock_adapter_send():
    adapter = MockAdapter(responses={"calc": "42"})
    ctx = BridgeContext(session_id="s1")
    resp = await adapter.send_message("calc", "what is 6*7?", ctx)
    assert resp.text == "42"
    assert resp.agent == "calc"
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_mock_adapter_default_response():
    adapter = MockAdapter(default_response="default")
    ctx = BridgeContext()
    resp = await adapter.send_message("any", "hi", ctx)
    assert resp.text == "default"


@pytest.mark.asyncio
async def test_mock_adapter_stream():
    adapter = MockAdapter(responses={"a": "streamed"})
    ctx = BridgeContext()
    events = []
    async for event in adapter.stream_message("a", "msg", ctx):
        events.append(event)
    assert len(events) == 1
    assert events[0].text == "streamed"


@pytest.mark.asyncio
async def test_mock_adapter_close():
    adapter = MockAdapter()
    assert not adapter.closed
    await adapter.close()
    assert adapter.closed


@pytest.mark.asyncio
async def test_mock_adapter_get_card():
    info = mock_agent_info("calc", skills=("math",))
    adapter = MockAdapter(agents={"calc": info})
    card = await adapter.get_agent_card("calc")
    assert card is not None
    assert card["name"] == "calc"


@pytest.mark.asyncio
async def test_mock_adapter_get_card_unknown():
    adapter = MockAdapter()
    card = await adapter.get_agent_card("unknown")
    assert card is None


@pytest.mark.asyncio
async def test_mock_adapter_discover():
    info = mock_agent_info("test")
    adapter = MockAdapter(agents={"test": info})
    agents = await adapter.discover_agents()
    assert len(agents) == 1
    assert agents[0].name == "test"


# ---- MockStreamingAdapter ----


@pytest.mark.asyncio
async def test_streaming_adapter_chunks():
    adapter = MockStreamingAdapter(
        responses={"a": "hello world from agent"},
        chunk_delay=0,
    )
    ctx = BridgeContext()
    events = []
    async for event in adapter.stream_message("a", "hi", ctx):
        events.append(event)
    assert len(events) == 4  # four words
    assert events[-1].is_final
    full = "".join(e.text for e in events)
    assert full == "hello world from agent"


# ---- RecordingMiddleware ----


@pytest.mark.asyncio
async def test_recording_middleware():
    from agentique.bridge.middleware import MiddlewareChain

    mw = RecordingMiddleware()
    chain = MiddlewareChain()
    chain.add(mw)

    async def handler(req):
        return {"result": "ok"}

    await chain.execute({"agent": "test"}, handler)

    assert len(mw.requests) == 1
    assert mw.requests[0] == {"agent": "test"}
    assert len(mw.responses) == 1
    assert mw.responses[0] == {"result": "ok"}


# ---- Factory helpers ----


def test_mock_agent_info():
    info = mock_agent_info("calc", skills=("math", "stats"))
    assert info.name == "calc"
    assert info.skills == ("math", "stats")
    assert info.base_url == "http://localhost:9999"


def test_mock_agent_card_basic():
    card = mock_agent_card("test")
    assert card["name"] == "test"
    assert card["capabilities"]["streaming"] is True


def test_mock_agent_card_with_tools():
    tools = [{"name": "add", "description": "Add numbers"}]
    card = mock_agent_card("calc", tools=tools)
    exts = card["capabilities"]["extensions"]
    assert len(exts) == 1
    assert exts[0]["params"]["mcpTools"] == tools


def test_mock_agent_card_with_sub_agents():
    subs = [{"name": "Calculator", "description": "Math"}]
    card = mock_agent_card("root", sub_agents=subs)
    assert card["metadata"]["sub_agents"] == subs
