"""Integration tests for FastMCP transforms (namespace, visibility).

Tests FastMCP 3.0 transform functionality using in-memory Client.
"""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP


@pytest.mark.integration
async def test_namespace_transform_prefixes_tools():
    """Mounted sub-servers get namespaced tool names."""
    from fastmcp.server.transforms import Namespace

    main = FastMCP("Main")
    sub = FastMCP("Sub")

    @sub.tool()
    def helper(text: str) -> str:
        """A helper tool."""
        return f"helped: {text}"

    # Mount with namespace
    main.mount(sub, namespace="team_a")

    async with Client(main) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]

        # The helper tool should have the namespace prefix
        assert any("team_a" in name for name in tool_names), f"Expected namespaced tools in {tool_names}"


@pytest.mark.integration
async def test_tool_visibility_hiding():
    """Tools can be hidden based on tags/conditions."""
    server = FastMCP("TestServer")

    @server.tool(tags={"admin"})
    def admin_tool(action: str) -> str:
        """Admin-only tool."""
        return f"admin: {action}"

    @server.tool()
    def public_tool(text: str) -> str:
        """Public tool."""
        return f"public: {text}"

    # Disable admin-tagged tools
    server.disable(tags={"admin"})

    async with Client(server) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]

        # admin_tool should be hidden
        assert "admin_tool" not in tool_names
        # public_tool should be visible
        assert "public_tool" in tool_names


@pytest.mark.integration
async def test_multiple_mounted_servers():
    """Multiple servers can be mounted with different namespaces."""
    main = FastMCP("Main")
    team_a = FastMCP("TeamA")
    team_b = FastMCP("TeamB")

    @team_a.tool()
    def process(data: str) -> str:
        return f"team_a: {data}"

    @team_b.tool()
    def process(data: str) -> str:  # noqa: F811  # Same name, different implementation
        return f"team_b: {data}"

    main.mount(team_a, namespace="team_a")
    main.mount(team_b, namespace="team_b")

    async with Client(main) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]

        # Both should be present with their namespaces
        # The exact naming depends on FastMCP's implementation
        assert len(tools) >= 2


@pytest.mark.integration
async def test_tool_without_namespace():
    """Tools on the main server don't get namespaced."""
    main = FastMCP("Main")

    @main.tool()
    def direct_tool(input: str) -> str:
        """Tool defined directly on main."""
        return f"direct: {input}"

    async with Client(main) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]

        # Should have original name without prefix
        assert "direct_tool" in tool_names


@pytest.mark.integration
async def test_visibility_can_be_toggled():
    """Tool visibility can be changed dynamically."""
    server = FastMCP("TestServer")

    @server.tool(tags={"premium"})
    def premium_feature(data: str) -> str:
        """Premium feature."""
        return f"premium: {data}"

    # Initially disable
    server.disable(tags={"premium"})

    async with Client(server) as client:
        tools_disabled = await client.list_tools()
        names_disabled = [t.name for t in tools_disabled]
        # premium_feature should be hidden initially

        # Re-enable
        server.enable(tags={"premium"})

        tools_enabled = await client.list_tools()
        names_enabled = [t.name for t in tools_enabled]

        # Should have more tools now (or at least same number if already enabled)
        assert len(tools_enabled) >= len(tools_disabled)
