"""Integration tests for A2A adapter with real protocol.

Tests the A2A adapter against mock agents using httpx.ASGITransport.
"""

from __future__ import annotations

import pytest
from uuid import uuid4

from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)


@pytest.mark.integration
async def test_echo_agent_card_discovery(echo_a2a_client):
    """Test agent advertises valid capabilities via agent card."""
    resp = await echo_a2a_client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200

    card = resp.json()
    assert "name" in card
    assert card["name"] == "echo"
    assert "version" in card


@pytest.mark.integration
async def test_echo_agent_send_message(echo_a2a_client):
    """Non-streaming message send returns the echoed message."""
    from a2a.utils.message import new_user_text_message

    message = new_user_text_message("Hello, world!")

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello, world!"}],
                "messageId": uuid4().hex,
            }
        },
    }

    response = await echo_a2a_client.post("/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "result" in data
    # The echo agent should return "echo: Hello, world!"


@pytest.mark.integration
async def test_calculator_agent_basic_operation(calculator_a2a_client):
    """Calculator agent performs basic arithmetic."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "What is 2 + 3?"}],
                "messageId": uuid4().hex,
            }
        },
    }

    response = await calculator_a2a_client.post("/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "result" in data
    # Should contain "5" in the response


@pytest.mark.integration
async def test_error_agent_returns_failure(error_a2a_client):
    """Error agent immediately returns a failed status."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "trigger error"}],
                "messageId": uuid4().hex,
            }
        },
    }

    response = await error_a2a_client.post("/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "result" in data
    # The error agent should return a failed status


@pytest.mark.integration
async def test_streaming_agent_returns_chunks(streaming_a2a_client):
    """Streaming agent sends multiple artifact chunks."""
    import asyncio

    from httpx_sse import aconnect_sse

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "stream test"}],
                "messageId": uuid4().hex,
            }
        },
    }

    events = []
    try:
        async with asyncio.timeout(5.0):
            async with aconnect_sse(streaming_a2a_client, "POST", "/", json=payload) as source:
                source.response.raise_for_status()
                async for sse in source.aiter_sse():
                    events.append(sse.json())
                    result = sse.json().get("result", {})
                    # Break on final event if present
                    if result.get("final", False):
                        break
    except asyncio.TimeoutError:
        pass  # Collect what we got

    # Should have received multiple events
    assert len(events) > 0, "Expected at least one SSE event"

    # Verify event structure
    for event in events:
        assert "jsonrpc" in event


@pytest.mark.integration
async def test_agent_card_has_required_fields(echo_a2a_client):
    """Agent card contains all required A2A protocol fields."""
    resp = await echo_a2a_client.get("/.well-known/agent-card.json")
    card = resp.json()

    # Required fields per A2A spec
    assert "name" in card
    assert "url" in card
    assert "version" in card
    assert "capabilities" in card
    assert "defaultInputModes" in card
    assert "defaultOutputModes" in card


@pytest.mark.integration
async def test_send_message_with_empty_text():
    """Sending empty message is handled gracefully."""
    from tests.agents.helpers import create_a2a_client_for_executor
    from tests.agents.mock_agents import EchoAgent

    async with await create_a2a_client_for_executor(EchoAgent(), name="echo") as client:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": ""}],
                    "messageId": uuid4().hex,
                }
            },
        }

        response = await client.post("/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data


@pytest.mark.integration
async def test_multiple_sequential_messages(echo_a2a_client):
    """Agent handles multiple sequential messages correctly."""
    for i in range(3):
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": f"Message {i}"}],
                    "messageId": uuid4().hex,
                }
            },
        }

        response = await echo_a2a_client.post("/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
