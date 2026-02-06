"""FastMCP server factory for agentique.

Creates a fully configured FastMCP server that bridges MCP clients to
A2A (or other protocol) agents. The server exposes:

    - ``agent``  — send a message and stream the response
    - ``agents`` — list available agents
    - ``task``   — query task state for transparency
    - ``inspect`` — view an agent's internal structure

Design principles:
    - Transparent bridging, not black-box proxying
    - Agent-centric (agents are the stars, not the bridge)
    - Real-time interactivity (streaming by default)
    - Preserve agent semantics and structure visibility
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from .core.config import AgentiqueConfig
from .core.events import AsyncEventEmitter
from .core.types import (
    AgentEvent,
    AgentHierarchy,
    AgentInfo,
    BridgeContext,
    StreamChunk,
    TaskState,
)
from .adapters.a2a import A2AAgentAdapter, A2AClientPool, A2ACardParser
from .bridge.provider import AgentProvider
from .bridge.router import AgentRouter
from .bridge.task_manager import TaskManager

logger = logging.getLogger(__name__)


def create_server(
    *,
    agents: list[AgentInfo] | None = None,
    router: AgentRouter | None = None,
    adapter: Any | None = None,
    client_factory: Any | None = None,
    config: AgentiqueConfig | None = None,
    events: AsyncEventEmitter | None = None,
) -> FastMCP:
    """Create a FastMCP server bridging MCP clients to agent backends.

    Args:
        agents: Agent descriptors to register.
        router: Pre-configured router (built from *agents* if absent).
        adapter: Pre-configured adapter (A2A adapter built if absent).
        client_factory: Legacy — passed to A2A adapter as client pool.
        config: Server configuration.
        events: Event emitter for lifecycle hooks.

    Returns:
        A fully configured ``FastMCP`` server instance.
    """
    config = config or AgentiqueConfig()
    emitter = events or AsyncEventEmitter()

    # Build router
    if router is None:
        router = AgentRouter(agents or [])
    agent_list = router.list_agents()

    # Build adapter
    if adapter is None:
        agent_map = {a.name: a for a in agent_list}
        pool: A2AClientPool | None = None
        if client_factory is not None:
            # Legacy support: wrap old-style factory
            pool = _wrap_legacy_factory(client_factory)
        else:
            pool = A2AClientPool(timeout=config.default_timeout)
        adapter = A2AAgentAdapter(agent_map, client_pool=pool)

    # Build provider
    card_parser = A2ACardParser()
    provider = AgentProvider(
        agent_list, adapter,
        card_parser=card_parser,
        cache_ttl=config.cache_ttl,
        prefetch_cards=config.prefetch_cards,
    )

    # Build server
    mcp = FastMCP(config.name, providers=[provider])

    # Task manager
    tasks = TaskManager()

    # ---- Core tools ----

    @mcp.tool(name="agent")
    async def agent_tool(
        message: str,
        target: str | None = None,
        context_id: str | None = None,
        ctx: Context = CurrentContext(),
    ):
        """Send a message to an A2A agent and stream the response.

        Args:
            message: The message to send to the agent
            target: Optional agent name (auto-routes if omitted)
            context_id: Optional context ID for conversation continuity
        """
        task_id = str(uuid4())
        effective_ctx_id = context_id or task_id
        tracker = await tasks.create(task_id, effective_ctx_id)
        hierarchy = AgentHierarchy(root=target or "auto")

        # Conversation history
        metadata: dict[str, Any] = {}
        if context_id:
            history = tasks.get_conversation_history(context_id)
            if history:
                metadata["conversation_history"] = history

        await emitter.emit("task.created", task_id=task_id)

        try:
            resolved = router.resolve(name=target, message=message)
            bridge_ctx = BridgeContext.from_fastmcp_context(ctx)
            if metadata.get("conversation_history"):
                bridge_ctx = BridgeContext(
                    session_id=bridge_ctx.session_id,
                    request_id=bridge_ctx.request_id,
                    client_id=bridge_ctx.client_id,
                    meta=bridge_ctx.meta,
                    conversation_history=metadata["conversation_history"],
                )

            index = 0
            async for event in adapter.stream_message(
                resolved.name, message, bridge_ctx,
            ):
                tracker.add_event(event)
                chunk = StreamChunk.from_event(resolved.name, index, event)
                index += 1

                # Build hierarchy from branch info
                if chunk.branch:
                    parts = chunk.branch.split(".")
                    for i, part in enumerate(parts):
                        parent = parts[i - 1] if i > 0 else None
                        if part not in hierarchy.agents:
                            hierarchy.add_agent(part, parent=parent)

                # Side-channel for status/progress
                if chunk.kind in {"status", "task"}:
                    if chunk.text:
                        branch = f" [{chunk.branch}]" if chunk.branch else ""
                        await ctx.info(f"{branch} {chunk.text}")
                    if chunk.progress:
                        await ctx.report_progress(int(chunk.progress), 100)

                # Yield content
                if chunk.kind in {"message", "artifact"} and chunk.text:
                    await emitter.emit("stream.chunk", chunk=chunk)
                    yield chunk.text

        finally:
            if not tracker.state.is_terminal:
                tracker.transition(TaskState.completed)

            # Persist conversation
            all_text = " ".join(
                e.text for e in tracker.events
                if e.text and e.kind in {"message", "artifact"}
            ).strip()
            if all_text and effective_ctx_id:
                tasks.append_conversation(effective_ctx_id, message, all_text)

            if hierarchy.agents:
                await tasks.set_hierarchy(task_id, hierarchy)

            await emitter.emit("task.completed", task_id=task_id)

    @mcp.tool(name="agents")
    def agents_tool() -> list[dict[str, Any]]:
        """List available A2A agents and their capabilities."""
        return [a.to_dict() for a in router.list_agents()]

    @mcp.tool(name="task")
    async def task_tool(
        id: str,
        ctx: Context = CurrentContext(),
    ) -> dict[str, Any]:
        """Query the state of a task.

        Args:
            id: The task ID to query
        """
        tracker = await tasks.get_or_none(id)
        if tracker:
            return tracker.to_dict()
        return {"error": f"Task {id} not found", "task_id": id}

    @mcp.tool(name="inspect")
    async def inspect_tool(
        name: str,
        ctx: Context = CurrentContext(),
    ) -> dict[str, Any]:
        """Inspect an agent's internal structure (sub-agents, tools).

        Args:
            name: The agent name to inspect
        """
        try:
            card = await provider.get_agent_card(name)
            if not card:
                return {"error": f"Agent '{name}' not found", "agent": name}
            hierarchy = card_parser.build_hierarchy(name, card)
            return hierarchy.to_dict()
        except Exception as exc:
            await ctx.error(f"Failed to inspect agent: {exc}")
            return {"error": str(exc), "agent": name}

    # ---- Background task support ----

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
                """Send a message as a background task.

                Returns immediately with a task ID queryable via ``task``.

                Args:
                    message: The message to send
                    target: Optional agent name
                """
                from fastmcp.server.dependencies import get_context
                bg_ctx = get_context()

                await progress.set_total(100)
                await progress.set_message("Connecting to agent...")

                task_id = str(uuid4())
                tracker = await tasks.create(task_id)
                resolved = router.resolve(name=target, message=message)
                bridge_ctx = BridgeContext.from_fastmcp_context(bg_ctx)

                chunks: list[str] = []
                idx = 0
                try:
                    async for event in adapter.stream_message(
                        resolved.name, message, bridge_ctx,
                    ):
                        tracker.add_event(event)
                        idx += 1
                        pct = min(10 + idx * 5, 90)
                        await progress.set_progress(pct)
                        if event.text:
                            chunks.append(event.text)
                            preview = event.text[:50]
                            await progress.set_message(f"Processing: {preview}")

                    await progress.set_progress(100)
                    await progress.set_message("Complete")
                    if not tracker.state.is_terminal:
                        tracker.transition(TaskState.completed)

                    return ToolResult(
                        content=" ".join(chunks).strip(),
                        structured_content={
                            "task_id": task_id,
                            "agent": resolved.name,
                            "state": tracker.state.value,
                            "event_count": idx,
                        },
                    )
                except Exception as exc:
                    tracker.transition(TaskState.failed, str(exc))
                    raise

        except ImportError:
            pass  # Background tasks require fastmcp[tasks]

    # ---- Catalog resource ----

    @mcp.resource("a2a://agents")
    def agents_catalog() -> str:
        """Catalog of available A2A agents."""
        return json.dumps(
            {"agents": [a.to_dict() for a in router.list_agents()]},
            indent=2,
        )

    return mcp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wrap_legacy_factory(factory: Any) -> A2AClientPool:
    """Wrap a legacy client factory into an ``A2AClientPool`` interface."""
    class _Wrapper(A2AClientPool):
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            super().__init__()

        async def get(self, base_url: str) -> Any:
            import inspect as _inspect
            result = self._inner.get(base_url)
            return await result if _inspect.isawaitable(result) else result

        async def close(self) -> None:
            close_fn = getattr(self._inner, "aclose", None) or getattr(self._inner, "close", None)
            if callable(close_fn):
                import inspect as _inspect
                result = close_fn()
                if _inspect.isawaitable(result):
                    await result

    return _Wrapper(factory)
