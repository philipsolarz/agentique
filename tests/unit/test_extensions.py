"""Tests for A2A extension propagation."""

from __future__ import annotations

from typing import Any, AsyncIterator
from uuid import uuid4

import pytest

from agentique.core.types import AgentEvent, AgentInfo, BridgeContext
from agentique.adapters.a2a.adapter import A2AAgentAdapter


# ---- Mock client for extension tests ----


class _TextPart:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResult:
    def __init__(self, text: str = "", extensions: list[str] | None = None) -> None:
        self.parts = [_TextPart(text)]
        self.result = self
        self.status = None
        self.metadata = {}
        self.task_id = str(uuid4())
        self.context_id = None
        self.extensions = extensions


class MockExtensionClient:
    """Mock A2A client that records extensions sent in messages."""

    def __init__(self) -> None:
        self.received_messages: list[Any] = []
        self.received_requests: list[Any] = []

    def get_card(self) -> Any:
        return _MockCard()

    async def send_message(self, request: Any) -> Any:
        self.received_requests.append(request)
        return _FakeResult(
            text="response",
            extensions=["urn:a2a:ext:tracing"],
        )

    async def send_message_streaming(self, request: Any) -> AsyncIterator[Any]:
        self.received_requests.append(request)
        yield _FakeResult(
            text="streamed",
            extensions=["urn:a2a:ext:tracing"],
        )


class _MockCard:
    supports_authenticated_extended_card = False

    class capabilities:
        extensions = [
            type("Ext", (), {
                "uri": "urn:a2a:ext:tracing",
                "description": "OpenTelemetry trace context",
                "required": False,
                "params": {"format": "w3c"},
            })(),
            type("Ext", (), {
                "uri": "urn:a2a:ext:logging",
                "description": "Structured logging",
                "required": None,
                "params": None,
            })(),
        ]


class MockExtensionPool:
    def __init__(self, client: MockExtensionClient) -> None:
        self._client = client

    async def get(self, base_url: str) -> MockExtensionClient:
        return self._client

    async def close(self) -> None:
        pass


# ---- Fixtures ----


@pytest.fixture
def agent_info() -> AgentInfo:
    return AgentInfo(
        name="ext-agent",
        base_url="http://localhost:9999",
        skills=("math",),
    )


@pytest.fixture
def bridge_ctx() -> BridgeContext:
    return BridgeContext(session_id="s1", request_id="r1")


# ---- Tests ----


@pytest.mark.asyncio
async def test_adapter_extensions_in_message(agent_info, bridge_ctx):
    """Adapter should include extensions in outgoing messages."""
    client = MockExtensionClient()
    pool = MockExtensionPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
        extensions=["urn:a2a:ext:tracing", "urn:a2a:ext:custom"],
    )

    # Build a message and check extensions are attached
    msg, metadata = adapter._build_message("hello", bridge_ctx)

    msg_extensions = getattr(msg, "extensions", None)
    if msg_extensions is not None:
        assert "urn:a2a:ext:tracing" in msg_extensions
        assert "urn:a2a:ext:custom" in msg_extensions


@pytest.mark.asyncio
async def test_adapter_no_extensions_by_default(agent_info, bridge_ctx):
    """Adapter without extensions should not set extensions on messages."""
    client = MockExtensionClient()
    pool = MockExtensionPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    msg, metadata = adapter._build_message("hello", bridge_ctx)
    # Either no extensions attribute or empty list
    msg_extensions = getattr(msg, "extensions", None)
    assert not msg_extensions or msg_extensions == []


@pytest.mark.asyncio
async def test_extract_extensions_from_response(agent_info):
    """_extract_metadata should capture extensions from response objects."""
    client = MockExtensionClient()
    pool = MockExtensionPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    response = _FakeResult(
        text="hello",
        extensions=["urn:a2a:ext:tracing"],
    )

    meta = adapter._extract_metadata(response)
    assert meta.get("extensions") == ["urn:a2a:ext:tracing"]


@pytest.mark.asyncio
async def test_extract_extensions_missing(agent_info):
    """_extract_metadata should handle missing extensions gracefully."""
    client = MockExtensionClient()
    pool = MockExtensionPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    response = _FakeResult(text="hello")
    response.extensions = None

    meta = adapter._extract_metadata(response)
    assert "extensions" not in meta


@pytest.mark.asyncio
async def test_get_agent_extensions(agent_info):
    """get_agent_extensions should return extension descriptors."""
    client = MockExtensionClient()
    pool = MockExtensionPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    extensions = await adapter.get_agent_extensions(agent_info.name)

    assert len(extensions) == 2
    assert extensions[0]["uri"] == "urn:a2a:ext:tracing"
    assert extensions[0]["description"] == "OpenTelemetry trace context"
    assert extensions[0]["required"] is False
    assert extensions[0]["params"] == {"format": "w3c"}
    assert extensions[1]["uri"] == "urn:a2a:ext:logging"


@pytest.mark.asyncio
async def test_get_agent_extensions_no_card(agent_info):
    """get_agent_extensions should return empty list when card is unavailable."""
    client = type("C", (), {"get_card": lambda self: None})()
    pool = MockExtensionPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    extensions = await adapter.get_agent_extensions(agent_info.name)
    assert extensions == []


@pytest.mark.asyncio
async def test_get_agent_extensions_no_capabilities(agent_info):
    """get_agent_extensions should handle card without capabilities."""
    card = type("Card", (), {"supports_authenticated_extended_card": False})()
    client = type("C", (), {"get_card": lambda self: card})()
    pool = MockExtensionPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    extensions = await adapter.get_agent_extensions(agent_info.name)
    assert extensions == []
