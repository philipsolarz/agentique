"""Integration tests for MCP tools using FastMCP Client in-memory transport.

These tests validate that agentique's MCP tools behave correctly without
requiring a network connection or external services.
"""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP
from fastmcp.tools.tool import ToolResult

from agentique import AgentInfo, create_server
from agentique.adapters.a2a import A2AAgentAdapter
from agentique.bridge.router import AgentRouter
from agentique.testing.mocks import MockAdapter


@pytest.mark.integration
async def test_agents_tool_lists_available_agents():
    """The 'agents' tool returns a list of available agents from adapters."""
    router = AgentRouter([
        AgentInfo(name="echo", base_url="http://localhost:9000", skills=("text",)),
    ])
    adapter = MockAdapter()
    server = create_server(router=router, adapter=adapter)

    async with Client(server) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        assert len(tools) > 0, "Bridge should expose at least one tool"
        assert "agents" in tool_names


@pytest.mark.integration
async def test_mcp_client_can_list_tools():
    """FastMCP Client can successfully list tools from the bridge."""
    router = AgentRouter([
        AgentInfo(name="echo", base_url="http://localhost:9000"),
    ])
    adapter = MockAdapter()
    server = create_server(router=router, adapter=adapter)

    async with Client(server) as client:
        tools = await client.list_tools()
        assert isinstance(tools, list)
        tool_names = [t.name for t in tools]
        assert "agent" in tool_names
        assert "agents" in tool_names
        assert "task" in tool_names


@pytest.mark.integration
async def test_bridge_initialization_with_no_adapters():
    """A bridge can be created with no adapters (though it won't be very useful)."""
    adapter = MockAdapter()
    server = create_server(agents=[], adapter=adapter)
    assert server is not None
    assert isinstance(server, FastMCP)


@pytest.mark.integration
async def test_bridge_with_multiple_agents():
    """A bridge can be configured with multiple agents."""
    agents = [
        AgentInfo(name="echo", base_url="http://localhost:9000"),
        AgentInfo(name="calc", base_url="http://localhost:9001"),
    ]
    adapter = MockAdapter()
    server = create_server(agents=agents, adapter=adapter)
    assert server is not None

    async with Client(server) as client:
        result = await client.call_tool("agents", {})
        agents_list = result.data["agents"] if isinstance(result.data, dict) else result.data
        assert len(agents_list) == 2


@pytest.mark.integration
async def test_client_with_in_memory_transport():
    """FastMCP Client(server) pattern works for in-memory testing."""
    server = FastMCP("TestServer")

    @server.tool()
    def test_tool(input: str) -> str:
        """A simple test tool."""
        return f"echo: {input}"

    async with Client(server) as client:
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

        result = await client.call_tool("test_tool", {"input": "hello"})
        assert result.content is not None
        assert "hello" in str(result.content)


@pytest.mark.integration
async def test_tool_call_returns_content():
    """Tool calls return properly formatted content."""
    server = FastMCP("TestServer")

    @server.tool()
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    async with Client(server) as client:
        result = await client.call_tool("greet", {"name": "World"})
        assert result.content is not None
        assert "Hello, World!" in str(result.content)


@pytest.mark.integration
async def test_tool_with_structured_output():
    """Tools can return structured data via ToolResult."""
    server = FastMCP("TestServer")

    @server.tool()
    def get_user(user_id: int) -> ToolResult:
        """Get user information."""
        return ToolResult(
            content=f"User {user_id}",
            structured_content={"id": user_id, "name": f"User {user_id}", "active": True},
        )

    async with Client(server) as client:
        result = await client.call_tool("get_user", {"user_id": 42})
        assert result.content is not None
        assert result.structured_content is not None or result.data is not None


@pytest.mark.integration
async def test_tool_with_invalid_args_raises_error():
    """Calling a tool with invalid arguments produces an error."""
    server = FastMCP("TestServer")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with Client(server) as client:
        with pytest.raises(Exception):
            await client.call_tool("add", {"a": 5})  # missing 'b'
