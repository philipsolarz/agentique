"""Integration tests for the agentique bridge.

These tests verify end-to-end MCP-to-A2A communication using
in-process ASGI transports (no real network).
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from agentique import AgentInfo, create_server
from agentique.bridge.router import AgentRouter

pytest.importorskip("a2a")
pytest.importorskip("google.adk")
pytest.importorskip("httpx")

from a2a.client import ClientConfig, ClientFactory  # noqa: E402

from tests.support.adk_agent import AdkEchoAgent  # noqa: E402
from tests.support.a2a_server import create_a2a_app  # noqa: E402


class AsyncStubClientPool:
    """Wraps a pre-built A2A client for test injection."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def get(self, base_url: str) -> Any:
        return self._client

    async def close(self) -> None:
        close_fn = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result


@pytest.mark.asyncio
async def test_end_to_end_routing_and_context():
    """Send a message through MCP → A2A and verify context propagation."""
    agent = AdkEchoAgent()
    base_url = "http://testserver"

    import httpx

    app, handler = create_a2a_app(agent, base_url)
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=base_url,
    )
    client_config = ClientConfig(httpx_client=http_client, streaming=False)
    a2a_client = await ClientFactory.connect(base_url, client_config=client_config)

    pool = AsyncStubClientPool(a2a_client)
    router = AgentRouter([AgentInfo(name="echo", base_url=base_url)])

    from agentique.adapters.a2a import A2AAgentAdapter
    adapter = A2AAgentAdapter(
        {"echo": AgentInfo(name="echo", base_url=base_url)},
        client_pool=pool,
    )
    server = create_server(router=router, adapter=adapter)

    from fastmcp import Client

    try:
        async with Client(server) as client:
            result = await client.call_tool("agent", {"message": "hello", "target": "echo"})
    finally:
        await http_client.aclose()
        await pool.close()

    payload = _unwrap_call_tool_data(result)
    assert payload is not None
    assert payload["agent"] == "echo"
    assert handler.last_metadata is not None
    assert "mcp" in handler.last_metadata


@pytest.mark.asyncio
async def test_agents_tool_lists_registered():
    """The ``agents`` tool should return all registered agents."""
    agent = AdkEchoAgent()
    base_url = "http://testserver"

    import httpx

    app, _ = create_a2a_app(agent, base_url)
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=base_url,
    )
    client_config = ClientConfig(httpx_client=http_client, streaming=False)
    a2a_client = await ClientFactory.connect(base_url, client_config=client_config)

    pool = AsyncStubClientPool(a2a_client)
    router = AgentRouter([
        AgentInfo(name="echo", base_url=base_url, skills=("text",)),
    ])

    from agentique.adapters.a2a import A2AAgentAdapter
    adapter = A2AAgentAdapter(
        {"echo": AgentInfo(name="echo", base_url=base_url, skills=("text",))},
        client_pool=pool,
    )
    server = create_server(router=router, adapter=adapter)

    from fastmcp import Client

    try:
        async with Client(server) as client:
            result = await client.call_tool("agents", {})
            data = _unwrap_call_tool_data(result)
            # Should be a list with one agent
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "echo"
    finally:
        await http_client.aclose()
        await pool.close()


@pytest.mark.asyncio
async def test_resources_available():
    """Provider should expose agent catalog as a resource."""
    agent = AdkEchoAgent()
    base_url = "http://testserver"

    import httpx

    app, _ = create_a2a_app(agent, base_url)
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=base_url,
    )
    client_config = ClientConfig(httpx_client=http_client, streaming=False)
    a2a_client = await ClientFactory.connect(base_url, client_config=client_config)

    pool = AsyncStubClientPool(a2a_client)
    router = AgentRouter([AgentInfo(name="echo", base_url=base_url)])

    from agentique.adapters.a2a import A2AAgentAdapter
    adapter = A2AAgentAdapter(
        {"echo": AgentInfo(name="echo", base_url=base_url)},
        client_pool=pool,
    )
    server = create_server(router=router, adapter=adapter)

    from fastmcp import Client

    try:
        async with Client(server) as client:
            if hasattr(client, "read_resource"):
                resource = await client.read_resource("a2a://agents")
                data = _unwrap_resource_data(resource)
                assert data
    finally:
        await http_client.aclose()
        await pool.close()


@pytest.mark.asyncio
async def test_error_handling_invalid_agent():
    """Calling agent tool with an unreachable URL should raise."""
    router = AgentRouter([
        AgentInfo(name="invalid", base_url="http://nonexistent:9999"),
    ])
    server = create_server(router=router)

    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("agent", {"message": "test", "target": "invalid"})


def _unwrap_call_tool_data(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        unwrapped = _unwrap_root(structured)
        if isinstance(unwrapped, dict) and "result" in unwrapped:
            return _unwrap_root(unwrapped["result"])
        return unwrapped

    if getattr(result, "data", None) is not None:
        return _unwrap_root(result.data)

    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        text = getattr(first, "text", None)
        if isinstance(text, str):
            stripped = text.strip()
            if stripped and stripped[0] in "{[":
                try:
                    return json.loads(stripped)
                except Exception:
                    return stripped
            return stripped
    return None


def _unwrap_resource_data(resource: Any) -> Any:
    if isinstance(resource, list):
        if not resource:
            return resource
        first = resource[0]
        text = getattr(first, "text", None)
        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return text
        return resource

    if getattr(resource, "data", None) is not None:
        return resource.data

    if getattr(resource, "content", None):
        content = resource.content[0]
        text = getattr(content, "text", None)
        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return text
    return resource


def _unwrap_root(value: Any) -> Any:
    if isinstance(value, list):
        return [_unwrap_root(v) for v in value]
    root = getattr(value, "root", None)
    if root is not None:
        return _unwrap_root(root)
    return value
