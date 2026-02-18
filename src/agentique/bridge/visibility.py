"""Visibility transform integration for per-session agent management.

Extends FastMCP's ``Visibility`` transform with **policy-driven
auto-configuration**. Session metadata (HTTP headers, auth tokens) can
automatically determine which agents are visible to each tenant without
any manual tool calls from the client.

Example — static transform::

    vis = AgentVisibility()
    vis.apply(mcp, agents=["agent-a", "agent-b"])

Example — automatic tenant-based policy::

    vis = AgentVisibility()
    vis.configure_policy(
        tenant_header="X-Tenant-ID",
        tenant_agents={
            "acme": ["crm-agent", "billing-agent"],
            "beta": ["analytics-agent"],
        },
    )
    vis.apply(mcp, agents=["crm-agent", "billing-agent", "analytics-agent"])

    # Register the middleware for automatic list_tools filtering
    middleware = vis.build_tenant_middleware(agent_tool_map={"crm-agent": ["agent"]})
    if middleware:
        mcp.add_middleware(middleware)

    # Or apply per-session inside a tool handler:
    await vis.apply_session_policy(ctx, session_metadata={"X-Tenant-ID": "acme"})
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import FastMCP Visibility transform
try:
    from fastmcp.server.transforms import Visibility
    _HAS_VISIBILITY = True
except ImportError:
    Visibility = None  # type: ignore[assignment,misc]
    _HAS_VISIBILITY = False

# Attempt to import FastMCP Middleware base
try:
    from fastmcp.server.middleware.middleware import (
        CallNext,
        Middleware,
        MiddlewareContext,
    )
    from fastmcp.tools.tool import Tool
    _HAS_MIDDLEWARE = True
except ImportError:
    Middleware = object  # type: ignore[assignment,misc]
    _HAS_MIDDLEWARE = False


# ---------------------------------------------------------------------------
# Policy definition
# ---------------------------------------------------------------------------


@dataclass
class VisibilityPolicy:
    """Defines automatic per-session visibility rules based on tenant identity.

    Attributes:
        tenant_header: HTTP header (or session metadata key) that carries
            the tenant identifier. Defaults to ``X-Tenant-ID``.
        tenant_agents: Mapping of tenant ID → list of agent names that
            should be visible to that tenant.
        tag_based: When ``True``, the agent lists in ``tenant_agents`` are
            interpreted as tag values rather than agent names.
        default_visible: When ``True``, agents not matched by any policy
            rule are shown to the client (permissive). When ``False``,
            unmatched tenants see nothing (deny-by-default).
    """

    tenant_header: str = "X-Tenant-ID"
    tenant_agents: dict[str, list[str]] = field(default_factory=dict)
    tag_based: bool = False
    default_visible: bool = True


# ---------------------------------------------------------------------------
# TenantVisibilityMiddleware
# ---------------------------------------------------------------------------


class TenantVisibilityMiddleware(Middleware):  # type: ignore[misc]
    """FastMCP middleware that enforces tenant-scoped agent visibility.

    Intercepts every ``list_tools`` request, reads the tenant identifier
    from the incoming session metadata (HTTP headers or context state), and
    filters the tool list to only include tools that belong to agents
    permitted for that tenant.

    Register via::

        mcp.add_middleware(TenantVisibilityMiddleware(policy, agent_tool_map))

    Args:
        policy: The ``VisibilityPolicy`` that defines tenant → agent rules.
        agent_tool_names: Mapping of agent name → list of MCP tool names
            exposed by that agent (used to filter the tool list).
    """

    def __init__(
        self,
        policy: VisibilityPolicy,
        agent_tool_names: dict[str, list[str]],
    ) -> None:
        self._policy = policy
        self._agent_tool_names = agent_tool_names

    async def on_list_tools(
        self,
        context: MiddlewareContext[Any],  # type: ignore[name-defined]
        call_next: CallNext[Any, Sequence[Tool]],  # type: ignore[name-defined]
    ) -> Sequence[Tool]:  # type: ignore[name-defined]
        """Filter the tool list based on the requesting tenant's policy."""
        tools = await call_next(context)
        tenant_id = self._extract_tenant_id(context)

        if tenant_id is None:
            return tools if self._policy.default_visible else []

        allowed_agents = self._policy.tenant_agents.get(tenant_id)
        if allowed_agents is None:
            # No explicit rule — apply default
            return tools if self._policy.default_visible else []

        allowed_names = self._allowed_tool_names(allowed_agents)
        return [t for t in tools if _tool_name(t) in allowed_names]

    def _extract_tenant_id(self, context: Any) -> str | None:
        """Extract the tenant ID from request metadata or session state."""
        raw_header = self._policy.tenant_header
        lower_header = raw_header.lower()
        snake_header = lower_header.replace("-", "_")

        # 1. Try context.state (FastMCP session/request state dict)
        state = getattr(context, "state", None)
        if isinstance(state, dict):
            for key in (raw_header, lower_header, snake_header):
                val = state.get(key)
                if val:
                    return str(val)
        elif state is not None:
            for key in (raw_header, lower_header, snake_header):
                val = getattr(state, key, None)
                if val:
                    return str(val)

        # 2. Try HTTP request headers
        request = getattr(context, "request", None)
        if request is not None:
            headers = getattr(request, "headers", {})
            for key in (raw_header, lower_header, snake_header):
                val = (
                    headers.get(key)
                    if isinstance(headers, dict)
                    else getattr(headers, key, None)
                )
                if val:
                    return str(val)

        return None

    def _allowed_tool_names(self, allowed_agents: list[str]) -> set[str]:
        """Compute the set of permitted MCP tool names for a tenant."""
        names: set[str] = set()
        for agent_name in allowed_agents:
            names.update(self._agent_tool_names.get(agent_name, []))
        # Core meta-tools are always permitted
        names.update({"agent", "agents", "task", "inspect", "agent_background"})
        return names


def _tool_name(tool: Any) -> str:
    if hasattr(tool, "name"):
        return str(tool.name)
    if isinstance(tool, dict):
        return str(tool.get("name", ""))
    return ""


# ---------------------------------------------------------------------------
# AgentVisibility
# ---------------------------------------------------------------------------


class AgentVisibility:
    """Manages per-session agent visibility using FastMCP transforms.

    Provides three levels of control:

    1. **Server-level static visibility** via the FastMCP ``Visibility``
       transform (``apply()``).
    2. **Session-level dynamic visibility** via ``ctx.enable_components()``
       / ``ctx.disable_components()`` (``enable_agent()`` / ``disable_agent()``).
    3. **Policy-driven automatic visibility** based on session metadata
       (``configure_policy()`` + ``apply_session_policy()``).

    Args:
        default_enabled: Whether agents are visible by default at the server
            level. When ``False``, agents must be explicitly enabled per
            session.
    """

    def __init__(self, default_enabled: bool = True) -> None:
        self._default_enabled = default_enabled
        self._agent_names: list[str] = []
        self._transform: Any | None = None
        self._policy: VisibilityPolicy | None = None

    # -- policy configuration --

    def configure_policy(
        self,
        *,
        tenant_header: str = "X-Tenant-ID",
        tenant_agents: dict[str, list[str]] | None = None,
        tag_based: bool = False,
        default_visible: bool = True,
    ) -> VisibilityPolicy:
        """Configure automatic tenant-based visibility rules.

        Once configured, call ``apply_session_policy()`` inside a tool
        handler to automatically enable/disable agents for the current
        session, or use ``build_tenant_middleware()`` to do it at the
        FastMCP middleware layer (``list_tools`` time).

        Args:
            tenant_header: Header/metadata key carrying the tenant ID.
            tenant_agents: Maps tenant ID → allowed agent names (or tags).
            tag_based: Interpret agent lists as tags rather than names.
            default_visible: Whether unmatched tenants see all agents.

        Returns:
            The configured ``VisibilityPolicy`` (also stored on ``self``).

        Example::

            vis.configure_policy(
                tenant_header="X-Tenant-ID",
                tenant_agents={
                    "acme": ["billing-agent", "crm-agent"],
                    "beta": ["analytics-agent"],
                },
            )
        """
        self._policy = VisibilityPolicy(
            tenant_header=tenant_header,
            tenant_agents=tenant_agents or {},
            tag_based=tag_based,
            default_visible=default_visible,
        )
        return self._policy

    @property
    def policy(self) -> VisibilityPolicy | None:
        """The active ``VisibilityPolicy``, or ``None`` if not configured."""
        return self._policy

    # -- server-level transform --

    def apply(
        self,
        mcp: Any,
        agents: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Apply the FastMCP ``Visibility`` transform to the server.

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

    # -- session-level policy application --

    async def apply_session_policy(
        self,
        ctx: Any,
        *,
        session_metadata: dict[str, Any] | None = None,
    ) -> list[str] | None:
        """Apply the configured policy to the current MCP session.

        Reads the tenant identifier from *session_metadata* (or from
        ``ctx`` state as a fallback) and enables/disables agents according
        to the configured ``VisibilityPolicy``.

        Args:
            ctx: FastMCP ``Context`` for this session.
            session_metadata: Key-value metadata for the session, typically
                HTTP headers forwarded from the MCP transport layer.

        Returns:
            The list of agent names that are now visible for this session,
            or ``None`` if no policy is configured or no tenant ID is found.
        """
        if self._policy is None:
            return None

        metadata = session_metadata or {}
        header = self._policy.tenant_header
        tenant_id: str | None = (
            metadata.get(header)
            or metadata.get(header.lower())
            or metadata.get(header.lower().replace("-", "_"))
        )

        # Fall back to ctx.get_state()
        if tenant_id is None:
            get_state = getattr(ctx, "get_state", None)
            if callable(get_state):
                try:
                    tenant_id = await get_state(header) or await get_state(
                        header.lower()
                    )
                except Exception:
                    pass

        if tenant_id is None:
            logger.debug(
                "No tenant ID found in session metadata (header: %s)", header,
            )
            return None

        tenant_id = str(tenant_id)
        allowed = self._policy.tenant_agents.get(tenant_id)

        if allowed is None:
            logger.info(
                "No visibility policy for tenant '%s'; default=%s",
                tenant_id,
                "allow-all" if self._policy.default_visible else "deny-all",
            )
            if not self._policy.default_visible:
                for name in self._agent_names:
                    await self.disable_agent(ctx, name)
                return []
            return list(self._agent_names) if self._agent_names else None

        logger.info(
            "Applying visibility policy for tenant '%s': agents=%s",
            tenant_id, allowed,
        )
        for name in self._agent_names:
            if name in allowed:
                await self.enable_agent(ctx, name)
            else:
                await self.disable_agent(ctx, name)

        return list(allowed)

    # -- session-level enable / disable --

    async def enable_agent(self, ctx: Any, agent_name: str) -> bool:
        """Enable an agent for the current session.

        Args:
            ctx: The FastMCP ``Context`` instance.
            agent_name: Name of the agent to enable.

        Returns:
            ``True`` if the operation succeeded.
        """
        enable = getattr(ctx, "enable_components", None)
        if callable(enable):
            try:
                await enable(names=[agent_name])
                return True
            except Exception:
                logger.debug("Failed to enable agent %s", agent_name, exc_info=True)
        return False

    async def disable_agent(self, ctx: Any, agent_name: str) -> bool:
        """Disable an agent for the current session.

        Args:
            ctx: The FastMCP ``Context`` instance.
            agent_name: Name of the agent to disable.

        Returns:
            ``True`` if the operation succeeded.
        """
        disable = getattr(ctx, "disable_components", None)
        if callable(disable):
            try:
                await disable(names=[agent_name])
                return True
            except Exception:
                logger.debug("Failed to disable agent %s", agent_name, exc_info=True)
        return False

    async def reset_visibility(self, ctx: Any) -> bool:
        """Reset visibility to server defaults for the current session.

        Args:
            ctx: The FastMCP ``Context`` instance.

        Returns:
            ``True`` if the operation succeeded.
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
            List of visible agent names, or ``None`` if tracking is
            unavailable.
        """
        get_state = getattr(ctx, "get_state", None)
        if callable(get_state):
            try:
                enabled = await get_state("_visibility_enabled")
                if enabled is not None:
                    return list(enabled)
            except Exception:
                pass

        if self._default_enabled:
            return list(self._agent_names) if self._agent_names else None
        return []

    # -- middleware factory --

    def build_tenant_middleware(
        self,
        agent_tool_map: dict[str, list[str]],
    ) -> TenantVisibilityMiddleware | None:
        """Build a ``TenantVisibilityMiddleware`` from the configured policy.

        The middleware filters the ``list_tools`` response on every request,
        so clients only see tools they are permitted to use.

        Args:
            agent_tool_map: Maps agent names → their MCP tool names.

        Returns:
            A configured ``TenantVisibilityMiddleware``, or ``None`` if no
            policy has been set or FastMCP middleware support is absent.
        """
        if self._policy is None or not _HAS_MIDDLEWARE:
            return None
        return TenantVisibilityMiddleware(self._policy, agent_tool_map)

    # -- management tools --

    def register_tools(self, mcp: Any) -> None:
        """Register ``enable_agent`` and ``disable_agent`` tools on *mcp*.

        These tools let MCP clients dynamically manage which agents are
        visible within their own session.

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
