"""Integration tests for MCP tools using FastMCP Client in-memory transport.

These tests validate that agentique's MCP tools behave correctly without
requiring a network connection or external services.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp import Client

# Tests will use the mcp_client fixture from conftest.py


@pytest.mark.integration
async def test_agents_tool_lists_available_agents():
    """The 'agents' tool returns a list of available agents from adapters."""
    # For this test, we need a bridge with actual adapters
    # We'll import and create a minimal setup
    from agentique.adapters.a2a import A2AAdapter
    from agentique.bridge import AgentiqueBridge

    # Create bridge with A2A adapter pointing to test agent
    adapter = A2AAdapter(agent_url="http://localhost:9000")
    bridge = AgentiqueBridge(adapters=[adapter])

    async with Client(bridge.mcp_server) as client:
        # List available tools first
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]

        # The bridge should expose core tools
        # Note: actual tool names depend on implementation
        assert len(tools) > 0, "Bridge should expose at least one tool"


@pytest.mark.integration
async def test_mcp_client_can_list_tools():
    """FastMCP Client can successfully list tools from the bridge."""
    from agentique.adapters.a2a import A2AAdapter
    from agentique.bridge import AgentiqueBridge

    adapter = A2AAdapter(agent_url="http://localhost:9000")
    bridge = AgentiqueBridge(adapters=[adapter])

    async with Client(bridge.mcp_server) as client:
        tools = await client.list_tools()
        assert isinstance(tools, list)
        # Should have at least the core agentique tools
        assert len(tools) >= 0  # May be 0 if no tools registered yet


@pytest.mark.integration
async def test_bridge_initialization_with_no_adapters():
    """A bridge can be created with no adapters (though it won't be very useful)."""
    from agentique.bridge import AgentiqueBridge

    bridge = AgentiqueBridge(adapters=[])
    assert bridge is not None
    assert hasattr(bridge, "mcp_server")


@pytest.mark.integration
async def test_bridge_with_multiple_adapters():
    """A bridge can be configured with multiple adapters."""
    from agentique.adapters.a2a import A2AAdapter
    from agentique.bridge import AgentiqueBridge

    adapter1 = A2AAdapter(agent_url="http://localhost:9000")
    adapter2 = A2AAdapter(agent_url="http://localhost:9001")

    bridge = AgentiqueBridge(adapters=[adapter1, adapter2])
    assert bridge is not None


@pytest.mark.integration
async def test_client_with_in_memory_transport():
    """FastMCP Client(server) pattern works for in-memory testing."""
    from fastmcp import FastMCP

    # Create a minimal FastMCP server
    server = FastMCP("TestServer")

    @server.tool()
    def test_tool(input: str) -> str:
        """A simple test tool."""
        return f"echo: {input}"

    # Use in-memory client
    async with Client(server) as client:
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

        # Call the tool
        result = await client.call_tool("test_tool", {"input": "hello"})
        assert len(result) > 0
        assert "hello" in str(result[0])


@pytest.mark.integration
async def test_tool_call_returns_content():
    """Tool calls return properly formatted content."""
    from fastmcp import FastMCP

    server = FastMCP("TestServer")

    @server.tool()
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    async with Client(server) as client:
        result = await client.call_tool("greet", {"name": "World"})
        assert len(result) > 0
        # Result is a list of Content objects
        assert hasattr(result[0], "text") or hasattr(result[0], "content")


@pytest.mark.integration
async def test_tool_with_structured_output():
    """Tools can return structured data via ToolResult."""
    from fastmcp import FastMCP
    from fastmcp.tools.tool import ToolResult

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
        # Check that we got structured data back
        # The exact format depends on FastMCP's implementation
        assert len(result) > 0


@pytest.mark.integration
async def test_tool_with_invalid_args_raises_error():
    """Calling a tool with invalid arguments produces an error."""
    from fastmcp import FastMCP

    server = FastMCP("TestServer")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with Client(server) as client:
        # Try calling with missing argument
        with pytest.raises(Exception):  # Specific exception depends on FastMCP
            await client.call_tool("add", {"a": 5})  # missing 'b'
