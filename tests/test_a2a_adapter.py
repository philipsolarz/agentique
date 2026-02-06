"""Tests for enhanced A2A adapter behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("a2a")

from agentique.adapters.a2a.adapter import A2AAgentAdapter
from agentique.core.types import AgentInfo, BridgeContext


@dataclass
class _RawMessage:
    text: str
    task_id: str | None = None
    context_id: str | None = None

    @property
    def parts(self) -> list[dict[str, str]]:
        return [{"text": self.text}]


class _FakePool:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def get(self, base_url: str) -> Any:
        return self._client

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_send_message_forwards_extensions_and_configuration():
    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, dict[str, Any]]] = []

        async def send_message(self, request: Any, **kwargs: Any):
            self.calls.append((request, kwargs))
            yield _RawMessage("ok", task_id="task-1", context_id="ctx-1")

    client = Client()
    adapter = A2AAgentAdapter(
        {"echo": AgentInfo(name="echo", base_url="http://example")},
        client_pool=_FakePool(client),
        default_extensions=("server_default",),
    )

    ctx = BridgeContext(
        session_id="s1",
        meta={
            "a2a_extensions": ["trace"],
            "a2a": {
                "history_length": 7,
                "push_notification": {
                    "url": "https://callback.example/a2a",
                    "token": "secret",
                },
            },
        },
    )

    response = await adapter.send_message("echo", "hello", ctx)

    assert response.text == "ok"
    assert client.calls
    _, kwargs = client.calls[0]
    assert set(kwargs["extensions"]) == {"server_default", "trace"}
    configuration = kwargs["configuration"]
    assert configuration is not None
    assert configuration.history_length == 7
    assert configuration.push_notification_config.url == "https://callback.example/a2a"


@pytest.mark.asyncio
async def test_stream_message_resubscribes_when_stream_drops():
    class Client:
        def __init__(self) -> None:
            self.resubscribe_calls: list[Any] = []

        async def send_message(self, request: Any, **kwargs: Any):
            yield _RawMessage("first", task_id="task-stream-1", context_id="ctx")
            raise ConnectionResetError("stream dropped")

        async def resubscribe(self, request: Any, **kwargs: Any):
            self.resubscribe_calls.append((request, kwargs))
            yield _RawMessage("second", task_id=request.id, context_id="ctx")

    client = Client()
    adapter = A2AAgentAdapter(
        {"echo": AgentInfo(name="echo", base_url="http://example")},
        client_pool=_FakePool(client),
        max_resubscribe_attempts=1,
    )

    ctx = BridgeContext(session_id="s1")
    events = [
        event
        async for event in adapter.stream_message("echo", "hello", ctx)
    ]

    assert [e.text for e in events] == ["first", "second"]
    assert len(client.resubscribe_calls) == 1


@pytest.mark.asyncio
async def test_get_agent_card_passes_default_extensions():
    class Client:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] | None = None

        async def get_card(self, **kwargs: Any):
            self.kwargs = kwargs
            return {"name": "echo", "url": "http://example"}

    client = Client()
    adapter = A2AAgentAdapter(
        {"echo": AgentInfo(name="echo", base_url="http://example")},
        client_pool=_FakePool(client),
        default_extensions=("trace",),
    )

    card = await adapter.get_agent_card("echo")

    assert card is not None
    assert card["name"] == "echo"
    assert client.kwargs == {"extensions": ["trace"]}
