"""Visibility transform integration for per-session agent management.

Wraps FastMCP's ``Visibility`` transform and session-level
``ctx.enable_components()`` / ``ctx.disable_components()`` to provide
dynamic per-session agent enable/disable functionality.

This allows multi-tenant configurations where different MCP sessions
see different subsets of agents.

Usage::

    from agentique.bridge.visibility import AgentVisibility

    vis = AgentVisibility()
    vis.apply(mcp, agents=["agent-a", "agent-b"])

    # Later, inside a tool:
    await vis.disable_agent(ctx, "agent-b")
    await vis.enable_agent(ctx, "agent-a")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import FastMCP Visibility transform
try:
    from fastmcp.server.transforms import Visibility
    _HAS_VISIBILITY = True
except ImportError:
    Visibility = None  # type: ignore[assignment,misc]
    _HAS_VISIBILITY = False


class AgentVisibility:
    """Manages per-session agent visibility using FastMCP transforms.

    Provides a high-level API for enabling/disabling agents at both the
    server level (via ``Visibility`` transform) and session level (via
    ``ctx.enable_components()`` / ``ctx.disable_components()``).

    Args:
        default_enabled: Whether agents are visible by default.
            If False, agents must be explicitly enabled per session.
    """

    def __init__(self, default_enabled: bool = True) -> None:
        self._default_enabled = default_enabled
        self._agent_names: list[str] = []
        self._transform: Any | None = None

    def apply(
        self,
        mcp: Any,
        agents: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Apply visibility transform to the FastMCP server.

        Args:
            mcp: The FastMCP server instance.
            agents: Agent/component names to manage visibility for.
            tags: Optional tags to filter components by.
        """
        if not _HAS_VISIBILITY:
            logger.warning(
                "FastMCP Visibility transform not available; "
                "agent visibility management disabled"
            )
            return

        self._agent_names = agents or []

        self._transform = Visibility(
            enabled=self._default_enabled,
            names=self._agent_names if self._agent_names else None,
            tags=tags,
        )
        mcp.add_transform(self._transform)

    async def enable_agent(self, ctx: Any, agent_name: str) -> bool:
        """Enable an agent for the current session.

        Args:
            ctx: The FastMCP ``Context`` instance.
            agent_name: Name of the agent to enable.

        Returns:
            True if the operation succeeded.
        """
        enable = getattr(ctx, "enable_components", None)
        if callable(enable):
            try:
                await enable(names=[agent_name])
                return True
            except Exception:
                logger.debug(
                    "Failed to enable agent %s", agent_name,
                    exc_info=True,
                )
        return False

    async def disable_agent(self, ctx: Any, agent_name: str) -> bool:
        """Disable an agent for the current session.

        Args:
            ctx: The FastMCP ``Context`` instance.
            agent_name: Name of the agent to disable.

        Returns:
            True if the operation succeeded.
        """
        disable = getattr(ctx, "disable_components", None)
        if callable(disable):
            try:
                await disable(names=[agent_name])
                return True
            except Exception:
                logger.debug(
                    "Failed to disable agent %s", agent_name,
                    exc_info=True,
                )
        return False

    async def reset_visibility(self, ctx: Any) -> bool:
        """Reset visibility to server defaults for the current session.

        Args:
            ctx: The FastMCP ``Context`` instance.

        Returns:
            True if the operation succeeded.
        """
        reset = getattr(ctx, "reset_visibility", None)
        if callable(reset):
            try:
                await reset()
                return True
            except Exception:
                logger.debug("Failed to reset visibility", exc_info=True)
        return False

    async def get_visible_agents(self, ctx: Any) -> list[str] | None:
        """Get the list of currently visible agents for this session.

        Args:
            ctx: The FastMCP ``Context`` instance.

        Returns:
            List of visible agent names, or None if visibility
            tracking is not available.
        """
        get_state = getattr(ctx, "get_state", None)
        if callable(get_state):
            try:
                enabled = await get_state("_visibility_enabled")
                if enabled is not None:
                    return list(enabled)
            except Exception:
                pass

        # Fall back to returning all registered agents
        if self._default_enabled:
            return list(self._agent_names) if self._agent_names else None
        return []

    def register_tools(self, mcp: Any) -> None:
        """Register visibility management tools on the server.

        Adds ``enable_agent`` and ``disable_agent`` tools that let
        MCP clients dynamically manage which agents are visible.

        Args:
            mcp: The FastMCP server instance.
        """
        vis = self

        @mcp.tool(name="enable_agent")
        async def enable_agent_tool(
            agent_name: str,
            ctx: Any = None,
        ) -> str:
            """Enable an agent for this session.

            Args:
                agent_name: Name of the agent to enable
            """
            if ctx is None:
                try:
                    from fastmcp.dependencies import CurrentContext
                    ctx = CurrentContext()
                except ImportError:
                    return '{"error": "Context not available"}'

            success = await vis.enable_agent(ctx, agent_name)
            if success:
                return f'{{"status": "enabled", "agent": "{agent_name}"}}'
            return f'{{"error": "Failed to enable agent", "agent": "{agent_name}"}}'

        @mcp.tool(name="disable_agent")
        async def disable_agent_tool(
            agent_name: str,
            ctx: Any = None,
        ) -> str:
            """Disable an agent for this session.

            Args:
                agent_name: Name of the agent to disable
            """
            if ctx is None:
                try:
                    from fastmcp.dependencies import CurrentContext
                    ctx = CurrentContext()
                except ImportError:
                    return '{"error": "Context not available"}'

            success = await vis.disable_agent(ctx, agent_name)
            if success:
                return f'{{"status": "disabled", "agent": "{agent_name}"}}'
            return f'{{"error": "Failed to disable agent", "agent": "{agent_name}"}}'

    @property
    def has_visibility_support(self) -> bool:
        """Whether FastMCP Visibility transform is available."""
        return _HAS_VISIBILITY
