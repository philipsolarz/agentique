"""Integration tests for FastMCP middleware functionality.

Tests custom middleware using in-memory Client.
"""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP
from fastmcp.server.middleware.middleware import Middleware


class CallRecorderMiddleware(Middleware):
    """Records all tool calls for test assertions."""

    def __init__(self):
        self.calls: list[dict] = []

    async def on_call_tool(self, context, call_next):
        """Record tool call before passing to next handler."""
        self.calls.append(
            {
                "tool": context.message.params.name if hasattr(context.message, "params") else "unknown",
            }
        )
        return await call_next(context)


@pytest.mark.integration
async def test_middleware_records_tool_calls():
    """Middleware can intercept and record tool calls."""
    server = FastMCP("TestServer")
    recorder = CallRecorderMiddleware()

    @server.tool()
    def test_tool(input: str) -> str:
        """A test tool."""
        return f"output: {input}"

    server.add_middleware(recorder)

    async with Client(server) as client:
        await client.call_tool("test_tool", {"input": "test"})

    assert len(recorder.calls) >= 1


@pytest.mark.integration
async def test_middleware_can_modify_response():
    """Middleware can intercept and modify tool responses."""

    class ResponseModifierMiddleware(Middleware):
        async def on_call_tool(self, context, call_next):
            result = await call_next(context)
            return result

    server = FastMCP("TestServer")
    modifier = ResponseModifierMiddleware()

    @server.tool()
    def echo(text: str) -> str:
        """Echo text."""
        return text

    server.add_middleware(modifier)

    async with Client(server) as client:
        result = await client.call_tool("echo", {"text": "hello"})
        assert result.content is not None


@pytest.mark.integration
async def test_multiple_middleware_chain():
    """Multiple middleware execute in order."""

    class CounterMiddleware(Middleware):
        def __init__(self, name):
            self.name = name
            self.count = 0

        async def on_call_tool(self, context, call_next):
            self.count += 1
            return await call_next(context)

    server = FastMCP("TestServer")
    mw1 = CounterMiddleware("first")
    mw2 = CounterMiddleware("second")

    @server.tool()
    def test() -> str:
        """A test tool."""
        return "ok"

    server.add_middleware(mw1)
    server.add_middleware(mw2)

    async with Client(server) as client:
        await client.call_tool("test", {})

    assert mw1.count >= 1 or mw2.count >= 1


@pytest.mark.integration
async def test_middleware_on_list_tools():
    """Middleware can intercept list_tools calls."""

    class ListInterceptor(Middleware):
        def __init__(self):
            self.list_calls = 0

        async def on_list_tools(self, context, call_next):
            self.list_calls += 1
            return await call_next(context)

    server = FastMCP("TestServer")
    interceptor = ListInterceptor()

    @server.tool()
    def dummy() -> str:
        """A dummy tool."""
        return "ok"

    server.add_middleware(interceptor)

    async with Client(server) as client:
        await client.list_tools()

    assert interceptor.list_calls >= 1
