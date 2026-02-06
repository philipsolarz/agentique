"""Tests for A2A adapter Phase 2/3 features.

Tests push notification config, task resubscription, extended agent cards,
and resilient streaming.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from uuid import uuid4

import pytest

from agentique.core.types import AgentEvent, AgentInfo, AgentResponse, BridgeContext
from agentique.adapters.a2a.adapter import A2AAgentAdapter


# ---- Mock A2A client for testing ----


class MockA2AClient:
    """Mock A2A SDK client for testing adapter features."""

    def __init__(
        self,
        *,
        card: dict[str, Any] | None = None,
        extended_card: dict[str, Any] | None = None,
        push_notification_support: bool = True,
        resubscribe_support: bool = True,
    ) -> None:
        self._card = card
        self._extended_card = extended_card
        self._push_support = push_notification_support
        self._resub_support = resubscribe_support
        self.push_configs: list[tuple[str, Any]] = []
        self.resubscribe_calls: list[str] = []

    def get_card(self) -> Any:
        return self._card

    async def get_authenticated_extended_card(self) -> Any:
        return self._extended_card

    async def send_message(self, request: Any) -> Any:
        return _FakeResult(text="hello")

    async def send_message_streaming(self, request: Any) -> AsyncIterator[Any]:
        yield _FakeResult(text="hello ")
        yield _FakeResult(text="world", is_final=True)

    async def set_push_notification_config(self, task_id: str, config: Any) -> None:
        self.push_configs.append((task_id, config))

    async def resubscribe(self, params: Any) -> AsyncIterator[Any]:
        task_id = getattr(params, "id", str(params))
        self.resubscribe_calls.append(task_id)
        yield _FakeResult(text="resumed")


class _FakeResult:
    def __init__(self, text: str = "", is_final: bool = False) -> None:
        self.parts = [_TextPart(text)]
        self.result = self
        self.status = None
        self.metadata = {}
        self.task_id = str(uuid4())
        self.context_id = None
        self.is_final = is_final

    class result:
        pass


class _TextPart:
    def __init__(self, text: str) -> None:
        self.text = text


class MockClientPool:
    """Mock client pool that returns our mock client."""

    def __init__(self, client: MockA2AClient) -> None:
        self._client = client

    async def get(self, base_url: str) -> MockA2AClient:
        return self._client

    async def close(self) -> None:
        pass


# ---- Fixtures ----


@pytest.fixture
def agent_info() -> AgentInfo:
    return AgentInfo(
        name="test-agent",
        base_url="http://localhost:9999",
        skills=("math", "text"),
    )


@pytest.fixture
def bridge_ctx() -> BridgeContext:
    return BridgeContext(session_id="s1", request_id="r1")


# ---- Push notification tests ----


@pytest.mark.asyncio
async def test_configure_push_notifications(agent_info, bridge_ctx):
    """Should configure push notifications on the A2A client."""
    client = MockA2AClient()
    pool = MockClientPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
        push_notification_url="https://example.com/webhook",
    )

    result = await adapter.configure_push_notifications(
        agent_info.name, "task-1",
    )

    assert result is True
    assert len(client.push_configs) == 1
    task_id, config = client.push_configs[0]
    assert task_id == "task-1"


@pytest.mark.asyncio
async def test_configure_push_notifications_custom_url(agent_info):
    """Should use custom callback URL when provided."""
    client = MockA2AClient()
    pool = MockClientPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    result = await adapter.configure_push_notifications(
        agent_info.name, "task-1",
        callback_url="https://custom.example.com/webhook",
    )

    assert result is True
    assert len(client.push_configs) == 1


@pytest.mark.asyncio
async def test_configure_push_notifications_no_url(agent_info):
    """Should return False when no push URL is configured."""
    client = MockA2AClient()
    pool = MockClientPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    result = await adapter.configure_push_notifications(
        agent_info.name, "task-1",
    )

    assert result is False


# ---- Task resubscription tests ----


@pytest.mark.asyncio
async def test_resubscribe(agent_info):
    """Should resubscribe to a task's event stream."""
    client = MockA2AClient()
    pool = MockClientPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    events: list[AgentEvent] = []
    async for event in adapter.resubscribe(agent_info.name, "task-123"):
        events.append(event)

    assert len(events) == 1
    assert events[0].text == "resumed"
    assert "task-123" in client.resubscribe_calls


# ---- Extended agent card tests ----


@pytest.mark.asyncio
async def test_get_agent_card_basic(agent_info):
    """Should fetch basic agent card."""
    card = {"name": "test-agent", "description": "A test agent"}
    client = MockA2AClient(card=card)
    pool = MockClientPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    result = await adapter.get_agent_card(agent_info.name)
    assert result == card


@pytest.mark.asyncio
async def test_get_agent_card_extended(agent_info):
    """Should fetch extended card when authenticated=True and supported."""
    basic_card = type("Card", (), {
        "name": "test-agent",
        "supports_authenticated_extended_card": True,
    })()
    extended_card = {"name": "test-agent", "skills": ["secret_skill"]}

    client = MockA2AClient(card=basic_card, extended_card=extended_card)
    pool = MockClientPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    result = await adapter.get_agent_card(agent_info.name, authenticated=True)
    assert result == extended_card


@pytest.mark.asyncio
async def test_get_agent_card_no_extended_support(agent_info):
    """Should return basic card when extended is not supported."""
    basic_card = type("Card", (), {
        "name": "test-agent",
        "supports_authenticated_extended_card": False,
    })()

    client = MockA2AClient(card=basic_card)
    pool = MockClientPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
    )

    result = await adapter.get_agent_card(agent_info.name, authenticated=True)
    assert result is basic_card  # Falls back to basic


# ---- Retry on disconnect tests ----


@pytest.mark.asyncio
async def test_adapter_retry_disabled(agent_info, bridge_ctx):
    """Adapter with retry disabled should not attempt resubscription."""
    client = MockA2AClient()
    pool = MockClientPool(client)

    adapter = A2AAgentAdapter(
        {agent_info.name: agent_info},
        client_pool=pool,
        retry_on_disconnect=False,
    )

    # Normal streaming should still work
    events: list[AgentEvent] = []
    async for event in adapter.stream_message(
        agent_info.name, "hello", bridge_ctx,
    ):
        events.append(event)

    assert len(events) == 2
