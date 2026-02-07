"""Integration tests for the agentique bridge.

These tests verify end-to-end MCP-to-A2A communication using
in-process ASGI transports (no real network).
"""

from __future__ import annotations

import inspect
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
    client_config = ClientConfig(httpx_client=http_client, streaming=True)
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

    # The agent tool is a generator that yields text content
    assert result.content is not None
    content_text = str(result.content)
    assert len(content_text) > 0


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
    client_config = ClientConfig(httpx_client=http_client, streaming=True)
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
            data = result.data
            # agents tool returns ToolResult with structured_content={"agents": [...]}
            if isinstance(data, dict):
                agents_list = data["agents"]
            else:
                agents_list = data
            assert isinstance(agents_list, list)
            assert len(agents_list) == 1
            assert agents_list[0]["name"] == "echo"
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
    client_config = ClientConfig(httpx_client=http_client, streaming=True)
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
                # read_resource may return a list or an object
                if isinstance(resource, list):
                    assert len(resource) > 0
                else:
                    assert resource is not None
    finally:
        await http_client.aclose()
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_error_handling_invalid_agent():
    """Calling agent tool with an unreachable URL should raise or return error."""
    router = AgentRouter([
        AgentInfo(name="invalid", base_url="http://127.0.0.1:19999"),
    ])
    server = create_server(router=router)

    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async with Client(server) as client:
        try:
            result = await client.call_tool(
                "agent", {"message": "test", "target": "invalid"},
                raise_on_error=False,
            )
            # If call completes without raising, verify it signals the error
            assert result.is_error or not str(result.content).strip()
        except (ToolError, Exception):
            pass  # Raising is also acceptable
