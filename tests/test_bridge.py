from __future__ import annotations

import inspect
from typing import Any

import pytest

from agentique import AgentDescriptor, create_server
from agentique.router import AgentRouter

pytest.importorskip("a2a")
pytest.importorskip("google.adk")
pytest.importorskip("httpx")

from a2a.client import ClientConfig, ClientFactory  # noqa: E402

from tests.support.adk_agent import AdkEchoAgent  # noqa: E402
from tests.support.a2a_server import create_a2a_app  # noqa: E402


class AsyncStubClientFactory:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def get(self, base_url: str) -> Any:
        return self._client

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


def _stream_tool_method(client: Any) -> Any | None:
    for name in ("call_tool_stream", "stream_tool"):
        method = getattr(client, name, None)
        if callable(method):
            return method
    return None


@pytest.mark.asyncio
async def test_end_to_end_routing_and_context():
    agent = AdkEchoAgent()
    base_url = "http://testserver"

    import httpx

    app, handler = create_a2a_app(agent, base_url)
    http_client = httpx.AsyncClient(app=app, base_url=base_url)
    client_config = ClientConfig(httpx_client=http_client, streaming=True)
    a2a_client = await ClientFactory.connect(base_url, client_config=client_config)

    factory = AsyncStubClientFactory(a2a_client)
    router = AgentRouter([AgentDescriptor(name="echo", base_url=base_url)])
    server = create_server(router=router, client_factory=factory)

    from fastmcp import Client

    try:
        async with Client(server) as client:
            result = await client.call_tool("a2a_send", {"message": "hello", "agent": "echo"})
    finally:
        await http_client.aclose()
        await factory.aclose()

    assert result.data["agent"] == "echo"
    assert result.data["text"]
    assert handler.last_metadata is not None
    assert "mcp" in handler.last_metadata


@pytest.mark.asyncio
async def test_resources_and_prompts_available():
    agent = AdkEchoAgent()
    base_url = "http://testserver"

    import httpx

    app, _ = create_a2a_app(agent, base_url)
    http_client = httpx.AsyncClient(app=app, base_url=base_url)
    client_config = ClientConfig(httpx_client=http_client, streaming=True)
    a2a_client = await ClientFactory.connect(base_url, client_config=client_config)

    factory = AsyncStubClientFactory(a2a_client)
    router = AgentRouter([AgentDescriptor(name="echo", base_url=base_url)])
    server = create_server(router=router, client_factory=factory)

    from fastmcp import Client

    try:
        async with Client(server) as client:
            if hasattr(client, "read_resource"):
                resource = await client.read_resource("a2a://agents")
                assert resource.data["agents"]
            if hasattr(client, "get_prompt"):
                prompt = await client.get_prompt("a2a_routing_prompt", {"goal": "say hi"})
                assert "say hi" in prompt.prompt
    finally:
        await http_client.aclose()
        await factory.aclose()


@pytest.mark.asyncio
async def test_streaming_tool_yields_chunks():
    agent = AdkEchoAgent()
    base_url = "http://testserver"

    import httpx

    app, _ = create_a2a_app(agent, base_url)
    http_client = httpx.AsyncClient(app=app, base_url=base_url)
    client_config = ClientConfig(httpx_client=http_client, streaming=True)
    a2a_client = await ClientFactory.connect(base_url, client_config=client_config)

    factory = AsyncStubClientFactory(a2a_client)
    router = AgentRouter([AgentDescriptor(name="echo", base_url=base_url)])
    server = create_server(router=router, client_factory=factory)

    from fastmcp import Client

    try:
        async with Client(server) as client:
            stream_method = _stream_tool_method(client)
            if stream_method is None:
                pytest.skip("FastMCP client streaming helper not available")

            chunks = []
            stream = stream_method("a2a_stream", {"message": "hello", "agent": "echo"})
            if inspect.isawaitable(stream):
                stream = await stream
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks
    finally:
        await http_client.aclose()
        await factory.aclose()
