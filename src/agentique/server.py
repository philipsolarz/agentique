"""FastMCP server for A2A agent integration.

This module implements a transparent MCP-A2A bridge using FastMCP 3.0's
Provider architecture. The bridge allows MCP clients to interact with
A2A agents as if they were natively exposed through MCP.

Core principles (from mission):
- Transparent bridging, not black-box proxying
- Agent-centric design (agents are the stars, not the bridge)
- Real-time interactivity (streaming by default)
- Preserve agent semantics and structure visibility

Tools provided:
- agent: Send message to any A2A agent (streams response)
- agents: List available A2A agents
- task: Query task state for transparency
- inspect: View agent's internal structure (sub-agents, tools)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.tools.tool import ToolResult

from .bridge import RouterBridge
from .models import (
    AgentHierarchy,
    TaskState,
    TaskTracker,
)
from .provider import A2AAgentProvider
from .router import AgentRouter


@dataclass
class ServerConfig:
    """Configuration for the A2A MCP server.

    The server acts as a transparent bridge between MCP clients and A2A agents.
    Configuration options control which advanced features are enabled.
    """

    name: str = "Agentique"
    # Feature flags
    enable_background_tasks: bool = True
    enable_elicitation: bool = True
    enable_tool_confirmation: bool = True
    # Performance tuning
    cache_ttl: float = 300.0
    prefetch_cards: bool = True


def create_server(
    *,
    name: str = "Agentique Bridge",
    agents: list[AgentDescriptor] | None = None,
    router: AgentRouter | None = None,
    client_factory: Any | None = None,
    config: ServerConfig | None = None,
) -> FastMCP:
    """Create a FastMCP server that routes requests to A2A agents.

    This creates a server using FastMCP 3.0's Provider architecture for clean
    dynamic component sourcing from A2A agents.

    Args:
        name: Server name
        agents: List of agent descriptors
        router: Optional pre-configured router
        client_factory: Optional A2A client factory
        config: Optional server configuration

    Returns:
        Configured FastMCP server instance
    """
    config = config or ServerConfig(name=name)
    active_router = router or AgentRouter(agents or [])

    # Create the A2A Agent Provider
    provider = A2AAgentProvider(
        agents=active_router.list_agents(),
        client_factory=client_factory,
        cache_ttl=config.cache_ttl,
        prefetch_cards=config.prefetch_cards,
    )

    # Create the server with the provider
    mcp = FastMCP(config.name, providers=[provider])

    # Create bridge for direct access (used by core tools)
    bridge = RouterBridge(active_router, client_factory=client_factory)

    # Task tracking for state machine
    task_trackers: dict[str, TaskTracker] = {}
    task_lock = asyncio.Lock()

    # =========================================================================
    # Core Tools - Mission-Aligned, Generic Bridge Interface
    # =========================================================================
    #
    # These tools provide a transparent bridge to A2A agents:
    # - `agent`: Primary interaction (streams by default for real-time interactivity)
    # - `agents`: Discovery (what's available)
    # - `task`: State query (transparency into what's happening)
    # - `inspect`: Introspection (visibility into multi-agent structure)
    #
    # The naming is generic and agent-centric, not feature-specific.
    # =========================================================================

    # Track conversation contexts for continuity
    conversation_contexts: dict[str, list[dict[str, str]]] = {}

    @mcp.tool(name="agent")
    async def agent_tool(
        message: str,
        target: str | None = None,
        context_id: str | None = None,
        ctx: Context = CurrentContext(),
    ):
        """Send a message to an A2A agent and stream the response.

        This is the primary tool for interacting with A2A agents. Responses
        stream in real-time so you can see progress as the agent works.

        Args:
            message: The message to send to the agent
            target: Optional agent name to route to (auto-routes if not specified)
            context_id: Optional context ID for conversation continuity (reuse to continue a conversation)
        """
        task_id = str(uuid4())
        effective_context_id = context_id or task_id
        tracker = TaskTracker(task_id=task_id, context_id=effective_context_id)

        # Build hierarchy tracker for sub-agent visibility
        agent_hierarchy = AgentHierarchy(root=target or "auto")

        async with task_lock:
            task_trackers[task_id] = tracker

        # Handle conversation continuity
        metadata: dict[str, Any] = {}
        if context_id and context_id in conversation_contexts:
            history = conversation_contexts[context_id]
            if history:
                metadata["conversation_history"] = history[-10:]  # Last 10 turns
                await ctx.debug(f"Continuing conversation with {len(history)} previous turns")

        try:
            async for chunk in bridge.stream(
                message,
                ctx=ctx,
                agent=target,
                metadata=metadata if metadata else None,
            ):
                # Update task tracker
                from .models import AgentEvent
                event = AgentEvent(
                    kind=chunk.kind,
                    text=chunk.text,
                    task_id=chunk.task_id or task_id,
                    context_id=chunk.context_id or effective_context_id,
                    progress=chunk.progress,
                    branch=chunk.branch,
                    author=chunk.author,
                    state=chunk.state,
                    is_final=chunk.is_final,
                    requires_confirmation=chunk.requires_confirmation,
                    tool_call=chunk.tool_call,
                )
                tracker.add_event(event)

                # Build dynamic hierarchy from branch info (sub-agent visibility)
                if chunk.branch:
                    parts = chunk.branch.split(".")
                    for i, part in enumerate(parts):
                        parent = parts[i - 1] if i > 0 else None
                        if part not in agent_hierarchy.agents:
                            agent_hierarchy.add_agent(part, parent=parent)

                # Handle tool confirmation if enabled
                if config.enable_tool_confirmation and chunk.requires_confirmation:
                    if chunk.tool_call:
                        confirmed = await _handle_tool_confirmation(
                            ctx, chunk.tool_call, chunk.agent, config
                        )
                        if not confirmed:
                            await ctx.warning("Operation cancelled by user")
                            tracker.transition(TaskState.canceled, "User cancelled")
                            break

                # Handle input required (elicitation)
                if config.enable_elicitation and tracker.state == TaskState.input_required:
                    user_input = await _handle_elicitation(
                        ctx, tracker.message or "Agent needs input", config
                    )
                    if user_input:
                        await ctx.info("Continuing with user input")

                # Progress and status go to side-channel
                if chunk.kind in {"status", "task"}:
                    if chunk.text:
                        branch_info = f" [{chunk.branch}]" if chunk.branch else ""
                        await ctx.info(f"{branch_info} {chunk.text}")
                    if chunk.progress:
                        await ctx.report_progress(int(chunk.progress), 100)

                # Yield actual content
                if chunk.kind in {"message", "artifact"} and chunk.text:
                    yield chunk.text

        finally:
            if not tracker.state.is_terminal:
                tracker.transition(TaskState.completed)

            # Store conversation history for continuity
            if effective_context_id:
                if effective_context_id not in conversation_contexts:
                    conversation_contexts[effective_context_id] = []
                # Collect all text from this interaction
                all_text = " ".join(
                    e.text for e in tracker.events if e.text and e.kind in {"message", "artifact"}
                ).strip()
                if all_text:
                    conversation_contexts[effective_context_id].append(
                        {"role": "user", "content": message}
                    )
                    conversation_contexts[effective_context_id].append(
                        {"role": "agent", "content": all_text}
                    )
                    # Keep last 20 turns
                    conversation_contexts[effective_context_id] = \
                        conversation_contexts[effective_context_id][-20:]

            # Store hierarchy in tracker for later inspection
            if len(agent_hierarchy.agents) > 0:
                tracker.hierarchy = agent_hierarchy

    @mcp.tool(name="agents")
    def agents_tool() -> list[dict[str, Any]]:
        """List available A2A agents.

        Returns information about each agent including name, description,
        and available skills/capabilities.
        """
        return [agent.to_dict() for agent in active_router.list_agents()]

    @mcp.tool(name="task")
    async def task_tool(
        id: str,
        ctx: Context = CurrentContext(),
    ) -> dict[str, Any]:
        """Query the state of a task.

        Use this to check on the status of a previous interaction,
        see progress, retrieve results, or view the agent hierarchy that handled it.

        Args:
            id: The task ID to query
        """
        async with task_lock:
            tracker = task_trackers.get(id)
            if tracker:
                result = tracker.to_dict()
                # Include hierarchy if available
                if hasattr(tracker, 'hierarchy') and tracker.hierarchy:
                    result["hierarchy"] = tracker.hierarchy.to_dict()
                return result
        return {"error": f"Task {id} not found", "task_id": id}

    @mcp.tool(name="inspect")
    async def inspect_tool(
        name: str,
        ctx: Context = CurrentContext(),
    ) -> dict[str, Any]:
        """Inspect an agent's internal structure.

        Shows the sub-agents, tools, and capabilities of a multi-agent system.
        Useful for understanding how tasks are delegated internally.

        Args:
            name: The agent name to inspect
        """
        try:
            card = await provider.get_agent_card(name)
            if not card:
                return {"error": f"Agent '{name}' not found", "agent": name}

            hierarchy = _build_agent_hierarchy(name, card)
            return hierarchy.to_dict()

        except Exception as exc:
            await ctx.error(f"Failed to inspect agent: {exc}")
            return {"error": str(exc), "agent": name}

    # =========================================================================
    # Background Task Support (SEP-1686)
    # =========================================================================
    #
    # For long-running operations, we provide a background execution mode.
    # This uses FastMCP's task=True decorator which runs the tool in the
    # background and allows progress tracking via the `task` tool.
    #
    # NOTE: This is separate from `agent` because FastMCP's background tasks
    # have a fundamentally different execution model (non-streaming, returns
    # task ID immediately).
    # =========================================================================

    if config.enable_background_tasks:
        try:
            from fastmcp.dependencies import Progress
            from fastmcp.tools.tool import ToolResult

            @mcp.tool(name="agent_background", task=True)
            async def agent_background_tool(
                message: str,
                target: str | None = None,
                progress: Progress = Progress(),
            ) -> ToolResult:
                """Send a message to an A2A agent as a background task.

                Use this for long-running operations. Returns immediately with a
                task ID that you can query with the `task` tool.

                Args:
                    message: The message to send to the agent
                    target: Optional agent name to route to
                """
                from fastmcp.server.dependencies import get_context
                bg_ctx = get_context()

                await progress.set_total(100)
                await progress.set_message("Connecting to agent...")

                task_id = str(uuid4())
                tracker = TaskTracker(task_id=task_id)

                async with task_lock:
                    task_trackers[task_id] = tracker

                chunks = []
                index = 0

                try:
                    async for chunk in bridge.stream(
                        message,
                        ctx=bg_ctx,
                        agent=target,
                    ):
                        chunks.append(chunk)
                        index += 1

                        # Update progress
                        pct = min(10 + index * 5, 90)
                        await progress.set_progress(pct)

                        if chunk.text:
                            preview = chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text
                            await progress.set_message(f"Processing: {preview}")

                        # Track events
                        from .models import AgentEvent
                        event = AgentEvent(
                            kind=chunk.kind,
                            text=chunk.text,
                            task_id=task_id,
                            branch=chunk.branch,
                            state=chunk.state,
                        )
                        tracker.add_event(event)

                    await progress.set_progress(100)
                    await progress.set_message("Complete")

                    # Finalize
                    if not tracker.state.is_terminal:
                        tracker.transition(TaskState.completed)

                    all_text = " ".join(c.text for c in chunks if c.text).strip()

                    return ToolResult(
                        content=all_text,
                        structured_content={
                            "task_id": task_id,
                            "agent": target or "auto",
                            "state": tracker.state.value,
                            "event_count": len(chunks),
                        },
                    )

                except Exception as exc:
                    tracker.transition(TaskState.failed, str(exc))
                    raise

        except ImportError:
            # Background tasks not available (missing fastmcp[tasks])
            pass

    # =========================================================================
    # Resources for Agent Discovery
    # =========================================================================

    @mcp.resource("a2a://agents")
    def agents_catalog_resource() -> str:
        """Catalog of available A2A agents."""
        return json.dumps({
            "agents": [agent.to_dict() for agent in active_router.list_agents()]
        }, indent=2)

    return mcp


# =============================================================================
# Helper Functions
# =============================================================================


async def _handle_tool_confirmation(
    ctx: Context,
    tool_call: dict[str, Any],
    agent: str | None,
    config: ServerConfig,
) -> bool:
    """Handle tool confirmation via elicitation."""
    if not config.enable_elicitation:
        return True  # Auto-confirm if elicitation disabled

    try:
        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})

        message = (
            f"Agent '{agent or 'unknown'}' wants to execute tool '{tool_name}' "
            f"with arguments: {json.dumps(arguments, indent=2)}\n\n"
            "Approve this action?"
        )

        result = await ctx.elicit(message, response_type=None)
        return result.action == "accept"

    except Exception as exc:
        await ctx.warning(f"Elicitation failed, auto-approving: {exc}")
        return True


async def _handle_elicitation(
    ctx: Context,
    message: str,
    config: ServerConfig,
) -> str | None:
    """Handle elicitation request from agent."""
    if not config.enable_elicitation:
        return None

    try:
        result = await ctx.elicit(message, response_type=str)
        if result.action == "accept":
            return result.data
        return None

    except Exception as exc:
        await ctx.warning(f"Elicitation failed: {exc}")
        return None


def _build_agent_hierarchy(root_name: str, card: Any) -> AgentHierarchy:
    """Build agent hierarchy from agent card."""
    hierarchy = AgentHierarchy(root=root_name)

    # Extract skills and infer sub-agents
    if hasattr(card, "skills"):
        skills = getattr(card, "skills", []) or []
        for skill in skills:
            skill_name = getattr(skill, "name", None) or skill.get("name") if isinstance(skill, dict) else str(skill)
            skill_desc = getattr(skill, "description", None) or (skill.get("description") if isinstance(skill, dict) else None)
            if skill_name:
                hierarchy.add_agent(
                    skill_name,
                    description=skill_desc,
                    parent=root_name,
                )

    # Check for explicit sub-agents in metadata
    metadata = None
    if hasattr(card, "metadata"):
        metadata = getattr(card, "metadata", None)
    elif isinstance(card, dict):
        metadata = card.get("metadata")

    if metadata and isinstance(metadata, dict):
        sub_agents = metadata.get("sub_agents", [])
        for sub in sub_agents:
            if isinstance(sub, dict):
                hierarchy.add_agent(
                    sub.get("name", "unknown"),
                    description=sub.get("description"),
                    skills=sub.get("skills", []),
                    parent=root_name,
                )

    return hierarchy
