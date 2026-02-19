"""Tests for agentique.testing — MockAdapter, assert_adapter_protocol, InMemoryBridge."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.anyio

from agentique.core.types import AgentEvent, AgentInfo, BridgeContext
from agentique.testing import (
    InMemoryBridge,
    MockAdapter,
    assert_adapter_protocol,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_agent(name: str = "bot", url: str = "http://bot") -> AgentInfo:
    return AgentInfo(name=name, base_url=url)


CTX = BridgeContext()


# ---------------------------------------------------------------------------
# MockAdapter — protocol compliance
# ---------------------------------------------------------------------------


def test_mock_adapter_satisfies_protocol():
    adapter = MockAdapter()
    assert_adapter_protocol(adapter)


async def test_discover_agents_returns_configured_list():
    agents = [make_agent("a"), make_agent("b")]
    adapter = MockAdapter(agents)
    result = await adapter.discover_agents()
    assert [ag.name for ag in result] == ["a", "b"]


async def test_discover_agents_empty():
    adapter = MockAdapter()
    assert await adapter.discover_agents() == []


# ---------------------------------------------------------------------------
# MockAdapter — send_message
# ---------------------------------------------------------------------------


async def test_send_message_returns_default_response():
    adapter = MockAdapter([make_agent()], default_response="hello")
    resp = await adapter.send_message("bot", "ping", CTX)
    assert resp.text == "hello"
    assert resp.agent == "bot"


async def test_send_message_returns_per_agent_response():
    adapter = MockAdapter(
        [make_agent("a"), make_agent("b")],
        responses={"a": "response-a", "b": "response-b"},
    )
    resp_a = await adapter.send_message("a", "msg", CTX)
    resp_b = await adapter.send_message("b", "msg", CTX)
    assert resp_a.text == "response-a"
    assert resp_b.text == "response-b"


async def test_send_message_logs_call():
    adapter = MockAdapter([make_agent()])
    await adapter.send_message("bot", "hello", CTX)
    assert adapter.calls == [{"agent_id": "bot", "message": "hello", "kind": "send"}]


async def test_close_is_noop():
    adapter = MockAdapter()
    await adapter.close()  # should not raise


# ---------------------------------------------------------------------------
# MockAdapter — stream_message
# ---------------------------------------------------------------------------


async def test_stream_message_yields_default_event():
    adapter = MockAdapter([make_agent()], default_response="streamed")
    events = []
    async for event in adapter.stream_message("bot", "ping", CTX):
        events.append(event)
    assert len(events) == 1
    assert events[0].text == "streamed"
    assert events[0].kind == "message"


async def test_stream_message_yields_configured_events():
    custom_events = [
        AgentEvent(kind="status", text="working"),
        AgentEvent(kind="message", text="done"),
    ]
    adapter = MockAdapter([make_agent()], events={"bot": custom_events})
    collected = []
    async for event in adapter.stream_message("bot", "go", CTX):
        collected.append(event)
    assert collected == custom_events


async def test_stream_message_logs_call():
    adapter = MockAdapter([make_agent()])
    async for _ in adapter.stream_message("bot", "msg", CTX):
        pass
    assert adapter.calls[0]["kind"] == "stream"


# ---------------------------------------------------------------------------
# MockAdapter — call log helpers
# ---------------------------------------------------------------------------


async def test_call_count():
    adapter = MockAdapter([make_agent("x"), make_agent("y")])
    await adapter.send_message("x", "a", CTX)
    await adapter.send_message("y", "b", CTX)
    await adapter.send_message("x", "c", CTX)
    assert adapter.call_count() == 3
    assert adapter.call_count("x") == 2
    assert adapter.call_count("y") == 1


async def test_last_message():
    adapter = MockAdapter([make_agent()])
    await adapter.send_message("bot", "first", CTX)
    await adapter.send_message("bot", "second", CTX)
    assert adapter.last_message("bot") == "second"
    assert adapter.last_message() == "second"


async def test_reset_calls():
    adapter = MockAdapter([make_agent()])
    await adapter.send_message("bot", "msg", CTX)
    assert adapter.call_count() == 1
    adapter.reset_calls()
    assert adapter.call_count() == 0


async def test_set_response_at_runtime():
    adapter = MockAdapter([make_agent()], default_response="old")
    resp1 = await adapter.send_message("bot", "m", CTX)
    assert resp1.text == "old"
    adapter.set_response("bot", "new")
    resp2 = await adapter.send_message("bot", "m", CTX)
    assert resp2.text == "new"


async def test_set_events_at_runtime():
    adapter = MockAdapter([make_agent()])
    adapter.set_events("bot", [AgentEvent(kind="message", text="override")])
    events = []
    async for e in adapter.stream_message("bot", "go", CTX):
        events.append(e)
    assert events[0].text == "override"


# ---------------------------------------------------------------------------
# assert_adapter_protocol
# ---------------------------------------------------------------------------


def test_assert_adapter_protocol_passes_for_mock():
    assert_adapter_protocol(MockAdapter())


def test_assert_adapter_protocol_fails_for_plain_object():
    with pytest.raises(AssertionError, match="does not satisfy"):
        assert_adapter_protocol(object())


class _PartialAdapter:
    """Missing stream_message and close — should fail the protocol check."""
    async def discover_agents(self): ...
    async def send_message(self, agent_id, message, context): ...


def test_assert_adapter_protocol_fails_for_partial():
    with pytest.raises(AssertionError):
        assert_adapter_protocol(_PartialAdapter())


# ---------------------------------------------------------------------------
# InMemoryBridge
# ---------------------------------------------------------------------------


def test_in_memory_bridge_creates_server():
    bridge = InMemoryBridge(agents=[make_agent()])
    assert bridge.server is not None


def test_in_memory_bridge_exposes_mock_adapter():
    bridge = InMemoryBridge(agents=[make_agent()])
    assert isinstance(bridge.adapter, MockAdapter)


def test_in_memory_bridge_reset_calls_delegates():
    bridge = InMemoryBridge(agents=[make_agent()])
    bridge.adapter.calls.append({"dummy": True})
    bridge.reset_calls()
    assert bridge.adapter.calls == []


def test_in_memory_bridge_empty_agents():
    bridge = InMemoryBridge()
    assert bridge.server is not None
    assert bridge.adapter.call_count() == 0
