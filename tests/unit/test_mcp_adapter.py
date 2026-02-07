"""Tests for the MCP proxy adapter."""

from __future__ import annotations

import pytest

from agentique.core.errors import AgentNotFoundError
from agentique.core.types import AgentInfo, BridgeContext


# ---------------------------------------------------------------------------
# Mock FastMCP proxy
# ---------------------------------------------------------------------------


class MockTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class MockToolResult:
    def __init__(self, content):
        self.content = content


class MockClient:
    """Mock for FastMCP test client."""

    def __init__(self, tools: list[MockTool] | None = None, result_text: str = "ok"):
        self._tools = tools if tools is not None else [MockTool("echo", "echo tool")]
        self._result_text = result_text

    async def list_tools(self) -> list[MockTool]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict | None = None) -> MockToolResult:
        return MockToolResult(content=[type("T", (), {"text": self._result_text})()])


class MockProxy:
    """Mock for FastMCPProxy."""

    def __init__(self, tools: list[MockTool] | None = None, result_text: str = "ok"):
        self._client = MockClient(tools, result_text)

    def test_client(self):
        return _AsyncContextManager(self._client)


class _AsyncContextManager:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMCPProxyAdapter:
    @pytest.fixture
    def agents(self):
        return {
            "remote-1": AgentInfo(
                name="remote-1",
                base_url="http://localhost:9001/mcp",
                description="Remote MCP server",
            ),
        }

    @pytest.fixture
    def adapter(self, agents):
        from agentique.adapters.mcp.adapter import MCPProxyAdapter

        adapter = MCPProxyAdapter(agents)
        # Inject mock proxy
        adapter._proxies["remote-1"] = MockProxy(result_text="hello from remote")
        return adapter

    @pytest.mark.anyio
    async def test_discover_agents(self, adapter):
        agents = await adapter.discover_agents()
        assert len(agents) == 1
        assert agents[0].name == "remote-1"

    @pytest.mark.anyio
    async def test_send_message(self, adapter):
        ctx = BridgeContext()
        response = await adapter.send_message("remote-1", "test", ctx)
        assert response.agent == "remote-1"
        assert "hello from remote" in response.text

    @pytest.mark.anyio
    async def test_send_message_unknown_agent(self, adapter):
        ctx = BridgeContext()
        with pytest.raises(AgentNotFoundError):
            adapter._get_proxy("nonexistent")

    @pytest.mark.anyio
    async def test_stream_message(self, adapter):
        ctx = BridgeContext()
        events = []
        async for event in adapter.stream_message("remote-1", "test", ctx):
            events.append(event)
        assert len(events) == 1
        assert "hello from remote" in events[0].text

    @pytest.mark.anyio
    async def test_get_agent_card(self, adapter):
        card = await adapter.get_agent_card("remote-1")
        assert card is not None
        assert card["name"] == "remote-1"
        assert "skills" in card
        assert card["metadata"]["protocol"] == "mcp"

    @pytest.mark.anyio
    async def test_no_tools_response(self, agents):
        from agentique.adapters.mcp.adapter import MCPProxyAdapter

        adapter = MCPProxyAdapter(agents)
        adapter._proxies["remote-1"] = MockProxy(tools=[])

        ctx = BridgeContext()
        response = await adapter.send_message("remote-1", "test", ctx)
        assert "No tools available" in response.text

    @pytest.mark.anyio
    async def test_close(self, adapter):
        await adapter.close()
        assert len(adapter._proxies) == 0


class TestExtractText:
    def test_string_result(self):
        from agentique.adapters.mcp.adapter import _extract_text
        assert _extract_text("hello") == "hello"

    def test_object_with_text_content(self):
        from agentique.adapters.mcp.adapter import _extract_text

        class Result:
            content = [type("B", (), {"text": "foo"})()]

        assert _extract_text(Result()) == "foo"

    def test_dict_content(self):
        from agentique.adapters.mcp.adapter import _extract_text

        class Result:
            content = [{"text": "bar"}]

        assert _extract_text(Result()) == "bar"

    def test_no_content(self):
        from agentique.adapters.mcp.adapter import _extract_text
        assert _extract_text(42) == "42"
