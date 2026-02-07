"""Tests for the visibility transform integration."""

from __future__ import annotations

import pytest

from agentique.bridge.visibility import AgentVisibility


# ---------------------------------------------------------------------------
# Mock context
# ---------------------------------------------------------------------------


class MockContext:
    """Simulates FastMCP Context with enable/disable visibility methods."""

    def __init__(self):
        self._enabled: set[str] = set()
        self._disabled: set[str] = set()
        self._state: dict[str, object] = {}
        self._reset_called = False

    async def enable_components(self, names: list[str] | None = None, **kwargs):
        if names:
            self._enabled.update(names)
            self._disabled -= set(names)

    async def disable_components(self, names: list[str] | None = None, **kwargs):
        if names:
            self._disabled.update(names)
            self._enabled -= set(names)

    async def reset_visibility(self):
        self._reset_called = True
        self._enabled.clear()
        self._disabled.clear()

    async def set_state(self, key: str, value: object) -> None:
        self._state[key] = value

    async def get_state(self, key: str) -> object | None:
        return self._state.get(key)


class MockMCP:
    """Minimal mock of FastMCP server for testing."""

    def __init__(self):
        self._transforms = []
        self._tools = {}

    def add_transform(self, transform):
        self._transforms.append(transform)

    def tool(self, name: str = "", **kwargs):
        def decorator(fn):
            self._tools[name or fn.__name__] = fn
            return fn
        return decorator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentVisibility:
    def test_init_default_enabled(self):
        vis = AgentVisibility()
        assert vis._default_enabled is True

    def test_init_disabled(self):
        vis = AgentVisibility(default_enabled=False)
        assert vis._default_enabled is False

    def test_has_visibility_support(self):
        vis = AgentVisibility()
        # May or may not have support depending on FastMCP version
        assert isinstance(vis.has_visibility_support, bool)

    @pytest.mark.anyio
    async def test_enable_agent(self):
        vis = AgentVisibility()
        ctx = MockContext()
        result = await vis.enable_agent(ctx, "agent-a")
        assert result is True
        assert "agent-a" in ctx._enabled

    @pytest.mark.anyio
    async def test_disable_agent(self):
        vis = AgentVisibility()
        ctx = MockContext()
        result = await vis.disable_agent(ctx, "agent-b")
        assert result is True
        assert "agent-b" in ctx._disabled

    @pytest.mark.anyio
    async def test_reset_visibility(self):
        vis = AgentVisibility()
        ctx = MockContext()
        await vis.enable_agent(ctx, "agent-a")
        result = await vis.reset_visibility(ctx)
        assert result is True
        assert ctx._reset_called

    @pytest.mark.anyio
    async def test_enable_without_support(self):
        vis = AgentVisibility()

        class NoSupportCtx:
            pass

        result = await vis.enable_agent(NoSupportCtx(), "agent-a")
        assert result is False

    @pytest.mark.anyio
    async def test_disable_without_support(self):
        vis = AgentVisibility()

        class NoSupportCtx:
            pass

        result = await vis.disable_agent(NoSupportCtx(), "agent-a")
        assert result is False

    @pytest.mark.anyio
    async def test_get_visible_agents_default_enabled(self):
        vis = AgentVisibility()
        vis._agent_names = ["agent-a", "agent-b"]
        ctx = MockContext()
        visible = await vis.get_visible_agents(ctx)
        assert visible == ["agent-a", "agent-b"]

    @pytest.mark.anyio
    async def test_get_visible_agents_default_disabled(self):
        vis = AgentVisibility(default_enabled=False)
        ctx = MockContext()
        visible = await vis.get_visible_agents(ctx)
        assert visible == []

    def test_register_tools(self):
        vis = AgentVisibility()
        mcp = MockMCP()
        vis.register_tools(mcp)
        assert "enable_agent" in mcp._tools
        assert "disable_agent" in mcp._tools

    def test_apply_without_visibility(self):
        vis = AgentVisibility()
        mcp = MockMCP()
        # Should not raise even if Visibility transform is unavailable
        vis.apply(mcp, agents=["agent-a"])
