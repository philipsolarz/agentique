"""FastMCP transform factory functions for common gateway patterns.

These helpers create FastMCP-native transforms as an alternative to
``ToolMapper`` for users who prefer the FastMCP transform pipeline.
Transforms are composable, ordered correctly by FastMCP, and integrate
with all other transform types (Namespace, Visibility, ToolTransform,
ResourcesAsTools, PromptsAsTools).

Usage with ``create_server()``::

    from agentique.bridge.tool_transforms import (
        namespace_transform_for_agent,
        visibility_transform,
        resources_as_tools_transform,
    )

    server = create_server(
        agents=agents,
        transforms=[
            namespace_transform_for_agent("billing"),
            visibility_transform(["billing_agent", "crm_agent"]),
            resources_as_tools_transform(),
        ],
    )

How this compares to ToolMapper
---------------------------------

``ToolMapper`` (in ``agentique.core.tool_mapper``) controls which MCP
tools are **created** from each agent's ``AgentInfo`` — it runs inside
the ``AgentProvider`` before tools are registered on the server.

FastMCP transforms control how **already-registered** tools are
presented to clients — they run at list-time, after the provider has
created tools.

The two are complementary:

* Use a ``ToolMapper`` to change the tool-creation strategy (e.g. one
  tool per skill instead of one tool per agent).
* Use FastMCP transforms to rename, namespace, filter, or expose
  resources as tools at the presentation layer.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Namespace transform helper
# ---------------------------------------------------------------------------


def namespace_transform_for_agent(
    agent_name: str,
    *,
    separator: str = "_",
) -> Any:
    """Create a ``Namespace`` transform that prefixes tool names.

    Renames every tool matching *agent_name* so it appears under a
    ``{agent_name}{separator}`` prefix.  Useful when mounting multiple
    bridges under one server and needing to avoid tool-name collisions.

    Args:
        agent_name: Prefix to add to matching tools.
        separator: Separator between prefix and tool name. Defaults to
            ``"_"`` (e.g. ``"billing_create_invoice"``).

    Returns:
        A FastMCP ``Namespace`` transform instance.

    Example::

        # Produces tools: billing_create_invoice, billing_list_invoices
        xf = namespace_transform_for_agent("billing")
        server = create_server(agents=agents, transforms=[xf])
    """
    from fastmcp.server.transforms import Namespace

    return Namespace(prefix=agent_name + separator)


# ---------------------------------------------------------------------------
# Visibility transform helper
# ---------------------------------------------------------------------------


def visibility_transform(
    names: list[str],
    *,
    enabled: bool = True,
    tags: list[str] | None = None,
) -> Any:
    """Create a ``Visibility`` transform to show/hide named components.

    Controls which tools, resources, and prompts are visible to clients
    at the server level.  Use ``AgentVisibility.configure_policy()`` for
    per-session/per-tenant control; use this for static server-level
    visibility.

    Args:
        names: Component names to manage visibility for.
        enabled: When ``True`` (default) the listed names are visible.
            When ``False`` they are hidden.
        tags: Optional tags to filter components by in addition to names.

    Returns:
        A FastMCP ``Visibility`` transform instance.

    Example::

        # Hide the experimental agent from all clients
        xf = visibility_transform(["experimental_agent"], enabled=False)
        server = create_server(agents=agents, transforms=[xf])
    """
    from fastmcp.server.transforms import Visibility

    return Visibility(
        names=names if names else None,
        enabled=enabled,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# ResourcesAsTools transform helper
# ---------------------------------------------------------------------------


def resources_as_tools_transform(server: Any) -> Any:
    """Create a ``ResourcesAsTools`` transform bound to *server*.

    Exposes every MCP resource — including artifact resources registered
    by the gateway (``a2a://{task_id}/artifacts/{artifact_id}``) — as
    callable tools.  This lets clients that cannot read resources directly
    access artifact content through tool calls.

    **Important**: FastMCP's ``ResourcesAsTools`` must be initialised with
    a reference to the server it will query for resources.  Call this
    factory *after* the server is built, then apply the transform::

        server = create_server(agents=agents)
        server.add_transform(resources_as_tools_transform(server))

    Alternatively, pass ``resources_as_tools=True`` to ``create_server()``
    to apply the transform automatically.

    Args:
        server: The ``FastMCP`` server instance whose resources to expose.

    Returns:
        A FastMCP ``ResourcesAsTools`` transform instance.
    """
    from fastmcp.server.transforms import ResourcesAsTools

    return ResourcesAsTools(server)


# ---------------------------------------------------------------------------
# PromptsAsTools transform helper
# ---------------------------------------------------------------------------


def prompts_as_tools_transform(server: Any) -> Any:
    """Create a ``PromptsAsTools`` transform bound to *server*.

    Exposes every registered MCP prompt as a callable tool.  This lets
    clients compose complex workflows from agent-defined prompt templates
    without needing native prompt support.

    **Important**: Like ``ResourcesAsTools``, this transform requires a
    server reference at construction time.  Call this factory *after* the
    server is built::

        server = create_server(agents=agents)
        server.add_transform(prompts_as_tools_transform(server))

    Alternatively, pass ``prompts_as_tools=True`` to ``create_server()``
    to apply the transform automatically.

    Args:
        server: The ``FastMCP`` server instance whose prompts to expose.

    Returns:
        A FastMCP ``PromptsAsTools`` transform instance.
    """
    from fastmcp.server.transforms import PromptsAsTools

    return PromptsAsTools(server)


# ---------------------------------------------------------------------------
# Convenience: build a full transform stack for a single-agent gateway
# ---------------------------------------------------------------------------


def default_transform_stack(
    agent_name: str | None = None,
    *,
    server: Any = None,
    resources_as_tools: bool = False,
    prompts_as_tools: bool = False,
) -> list[Any]:
    """Build a sensible default transform stack.

    Args:
        agent_name: When given, adds a ``Namespace`` transform so all
            tools are prefixed with ``{agent_name}_``.
        server: Required when *resources_as_tools* or *prompts_as_tools*
            is ``True``.  Pass the ``FastMCP`` server instance.
        resources_as_tools: When ``True``, adds a ``ResourcesAsTools``
            transform so artifact resources are callable as tools.
            *server* must be provided.
        prompts_as_tools: When ``True``, adds a ``PromptsAsTools``
            transform.  *server* must be provided.

    Returns:
        A list of FastMCP transform instances.

    Example — namespace only (no server needed)::

        transforms = default_transform_stack("billing")
        server = create_server(agents=agents, transforms=transforms)

    Example — with resources as tools (server created first)::

        server = create_server(agents=agents)
        for xf in default_transform_stack(server=server, resources_as_tools=True):
            server.add_transform(xf)
    """
    transforms: list[Any] = []
    if agent_name:
        transforms.append(namespace_transform_for_agent(agent_name))
    if resources_as_tools:
        if server is None:
            raise ValueError(
                "default_transform_stack: 'server' is required when "
                "resources_as_tools=True"
            )
        transforms.append(resources_as_tools_transform(server))
    if prompts_as_tools:
        if server is None:
            raise ValueError(
                "default_transform_stack: 'server' is required when "
                "prompts_as_tools=True"
            )
        transforms.append(prompts_as_tools_transform(server))
    return transforms
