"""FastMCP server factory for agentique.

Creates a fully configured FastMCP server that bridges MCP clients to
A2A (or other protocol) agents. The server exposes:

    - ``agent``  — send a message and stream the response
    - ``agents`` — list available agents
    - ``task``   — query task state
    - ``inspect`` — view an agent's internal structure

Design principles:
    - Transparent bridging, not black-box proxying
    - Agent-centric (agents are the stars, not the bridge)
    - Real-time interactivity (streaming by default)
    - Preserve agent semantics and structure visibility

Integration features:
    - Dependency injection via ``Depends()`` for clean tool signatures
    - FastMCP Middleware (AgentiqueMiddleware) for server-level hooks
    - Transform support (Namespace, Visibility) for agent isolation
    - Composition via ``mount()`` for multi-bridge architectures
    - Elicitation for input-required A2A task states
    - Structured content via ToolResult
    - TaskConfig for fine-grained background task control
    - OpenTelemetry span attributes
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext, Depends
from fastmcp.tools.tool import ToolResult

from .core.config import AgentiqueConfig
from .core.events import AsyncEventEmitter
from .core.telemetry import set_span_attribute
from .core.types import (
    AgentEvent,
    AgentHierarchy,
    AgentInfo,
    BridgeContext,
    StreamChunk,
    TaskState,
)
from .adapters.a2a import A2AAgentAdapter, A2AClientPool, A2ACardParser
from .bridge.context_manager import ContextManager
from .bridge.dependencies import (
    clear as clear_deps,
    configure as configure_deps,
    get_adapter,
    get_config,
    get_context_manager,
    get_emitter,
    get_router,
    get_task_manager,
)
from .bridge.fastmcp_middleware import AgentiqueMiddleware
from .bridge.middleware import (
    ErrorMappingMiddleware,
    LoggingMiddleware,
    MiddlewareChain,
)
from .bridge.provider import AgentProvider
from .bridge.router import AgentRouter
from .bridge.storage import InMemoryTaskStore, TaskStore
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
    middleware: list[Any] | None = None,
    task_store: TaskStore | None = None,
    transforms: list[Any] | None = None,
    namespace: str | None = None,
    visibility: Any | None = None,
) -> FastMCP:
    """Create a FastMCP server bridging MCP clients to agent backends.

    Args:
        agents: Agent descriptors to register.
        router: Pre-configured router (built from *agents* if absent).
        adapter: Pre-configured adapter (A2A adapter built if absent).
        client_factory: Legacy — passed to A2A adapter as client pool.
        config: Server configuration.
        events: Event emitter for lifecycle hooks.
        middleware: List of middleware instances for the bridge chain.
        task_store: Pluggable storage backend for task persistence.
        transforms: List of FastMCP Transform instances to apply.
        namespace: Optional namespace prefix for all agent tools.
        visibility: Optional ``AgentVisibility`` instance. When provided,
            its tenant policy middleware is registered and
            ``apply_session_policy()`` is called at the start of each
            ``agent`` tool invocation.

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

    # Build bridge middleware chain
    chain = MiddlewareChain()
    chain.add(LoggingMiddleware())
    chain.add(ErrorMappingMiddleware())
    if middleware:
        for mw in middleware:
            chain.add(mw)

    # Build context manager
    context_mgr = ContextManager()

    # Task manager with pluggable storage
    tasks = TaskManager(store=task_store or InMemoryTaskStore())

    # Populate the dependency registry for Depends()-based injection
    configure_deps(
        router=router,
        adapter=adapter,
        task_manager=tasks,
        config=config,
        emitter=emitter,
        context_manager=context_mgr,
    )

    # Build server
    mcp = FastMCP(config.name, providers=[provider])

    # Add FastMCP-level middleware
    mcp.add_middleware(AgentiqueMiddleware(emitter=emitter))

    # Register tenant visibility middleware when a policy is configured
    if visibility is not None:
        _agent_tool_map = {a.name: ["agent"] for a in agent_list}
        vis_middleware = visibility.build_tenant_middleware(_agent_tool_map)
        if vis_middleware is not None:
            mcp.add_middleware(vis_middleware)

    # Apply transforms
    if namespace:
        from fastmcp.server.transforms import Namespace
        mcp.add_transform(Namespace(namespace))

    if transforms:
        for transform in transforms:
            mcp.add_transform(transform)

    # ---- Core tools (using Depends() for dependency injection) ----

    @mcp.tool(name="agent")
    async def agent_tool(
        message: str,
        target: str | None = None,
        context_id: str | None = None,
        ctx: Context = CurrentContext(),
        _router: AgentRouter = Depends(get_router),
        _adapter: Any = Depends(get_adapter),
        _tasks: TaskManager = Depends(get_task_manager),
        _config: AgentiqueConfig = Depends(get_config),
        _emitter: AsyncEventEmitter = Depends(get_emitter),
        _ctx_mgr: ContextManager = Depends(get_context_manager),
    ) -> str:
        """Send a message to an A2A agent and stream the response.

        Args:
            message: The message to send to the agent
            target: Optional agent name (auto-routes if omitted)
            context_id: Optional context ID for conversation continuity
        """
        task_id = str(uuid4())

        # Auto-apply tenant visibility policy at session initialisation
        if visibility is not None:
            session_meta: dict[str, Any] = {}
            _raw_meta = getattr(ctx, "meta", None) or getattr(ctx, "metadata", None)
            if isinstance(_raw_meta, dict):
                session_meta = _raw_meta
            try:
                await visibility.apply_session_policy(ctx, session_metadata=session_meta)
            except Exception as _vis_exc:
                logger.debug("Visibility policy application skipped: %s", _vis_exc)

        # Resolve context ID via the context manager
        session_id = getattr(ctx, "session_id", None)
        effective_ctx_id = await _ctx_mgr.resolve_context(
            session_id=session_id,
            context_id=context_id,
            task_id=task_id,
        )
        await _ctx_mgr.track_task(effective_ctx_id, task_id)

        tracker = await _tasks.create(task_id, effective_ctx_id)
        hierarchy = AgentHierarchy(root=target or "auto")

        # Conversation history
        metadata: dict[str, Any] = {}
        if context_id:
            history = _tasks.get_conversation_history(context_id)
            if history:
                metadata["conversation_history"] = history

        await _emitter.emit("task.created", task_id=task_id)

        # Collect text chunks (FastMCP 3.0b1 does not consume async generators
        # from tool functions — it stringifies the generator object instead.
        # We collect chunks and return the joined text.)
        text_parts: list[str] = []

        try:
            # Use aresolve for LLM-capable routing
            resolved = await _router.aresolve(
                name=target, message=message, ctx=ctx,
            )

            # OpenTelemetry tracing
            set_span_attribute("agentique.agent_name", resolved.name)
            set_span_attribute("agentique.task_id", task_id)

            bridge_ctx = BridgeContext.from_fastmcp_context(ctx)
            if metadata.get("conversation_history"):
                bridge_ctx = bridge_ctx.replace(
                    conversation_history=metadata["conversation_history"],
                )

            index = 0
            async for event in _adapter.stream_message(
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

                # Handle input-required state via elicitation
                if chunk.state == "input-required" and _config.enable_elicitation:
                    elicit_text = chunk.text or "The agent requires additional input."
                    try:
                        elicit_result = await ctx.elicit(
                            elicit_text,
                            response_type=None,
                        )
                        if hasattr(elicit_result, "data") and elicit_result.data:
                            user_input = str(elicit_result.data)
                            async for follow_event in _adapter.stream_message(
                                resolved.name, user_input, bridge_ctx,
                            ):
                                tracker.add_event(follow_event)
                                follow_chunk = StreamChunk.from_event(
                                    resolved.name, index, follow_event,
                                )
                                index += 1
                                if follow_chunk.kind in {"message", "artifact"} and follow_chunk.text:
                                    await _emitter.emit("stream.chunk", chunk=follow_chunk)
                                    text_parts.append(follow_chunk.text)
                    except Exception as elicit_exc:
                        logger.warning("Elicitation failed: %s", elicit_exc)

                # Handle auth-required state
                if chunk.state == "auth-required":
                    await ctx.warning(
                        "Agent requires authentication. "
                        "Provide credentials via context metadata."
                    )

                # Capture artifacts as MCP Resources
                if chunk.kind == "artifact" and chunk.text:
                    art_id = event.artifact_id or f"artifact-{index}"
                    art_name = event.artifact_name or art_id
                    uri = _tasks.capture_artifact(
                        task_id, art_id, chunk.text, name=art_name,
                    )
                    logger.debug("Artifact registered: %s", uri)
                    await ctx.info(f"Artifact available: {uri}")

                # Collect content
                if chunk.kind in {"message", "artifact"} and chunk.text:
                    await _emitter.emit("stream.chunk", chunk=chunk)
                    text_parts.append(chunk.text)

        finally:
            if not tracker.state.is_terminal:
                tracker.transition(TaskState.completed)

            set_span_attribute("agentique.task_state", tracker.state.value)

            # Persist conversation
            all_text = " ".join(
                e.text for e in tracker.events
                if e.text and e.kind in {"message", "artifact"}
            ).strip()
            if all_text and effective_ctx_id:
                _tasks.append_conversation(effective_ctx_id, message, all_text)

            if hierarchy.agents:
                await _tasks.set_hierarchy(task_id, hierarchy)

            await _emitter.emit("task.completed", task_id=task_id)

        return "".join(text_parts)

    @mcp.tool(name="agents")
    def agents_tool(
        _router: AgentRouter = Depends(get_router),
    ) -> ToolResult:
        """List available A2A agents and their capabilities."""
        agents_data = [a.to_dict() for a in _router.list_agents()]
        return ToolResult(
            content=json.dumps(agents_data, indent=2),
            structured_content={"agents": agents_data},
        )

    @mcp.tool(name="task")
    async def task_tool(
        id: str,
        ctx: Context = CurrentContext(),
        _tasks: TaskManager = Depends(get_task_manager),
    ) -> ToolResult:
        """Query the state of a task.

        Args:
            id: The task ID to query
        """
        tracker = await _tasks.get_or_none(id)
        if tracker:
            data = tracker.to_dict()
            return ToolResult(
                content=json.dumps(data, indent=2),
                structured_content=data,
            )
        error_data = {"error": f"Task {id} not found", "task_id": id}
        return ToolResult(
            content=json.dumps(error_data),
            structured_content=error_data,
        )

    @mcp.tool(name="inspect")
    async def inspect_tool(
        name: str,
        ctx: Context = CurrentContext(),
    ) -> ToolResult:
        """Inspect an agent's internal structure (sub-agents, tools).

        Args:
            name: The agent name to inspect
        """
        try:
            card = await provider.get_agent_card(name)
            if not card:
                error_data = {"error": f"Agent '{name}' not found", "agent": name}
                return ToolResult(
                    content=json.dumps(error_data),
                    structured_content=error_data,
                )
            hierarchy = card_parser.build_hierarchy(name, card)
            data = hierarchy.to_dict()
            return ToolResult(
                content=json.dumps(data, indent=2),
                structured_content=data,
            )
        except Exception as exc:
            await ctx.error(f"Failed to inspect agent: {exc}")
            error_data = {"error": str(exc), "agent": name}
            return ToolResult(
                content=json.dumps(error_data),
                structured_content=error_data,
            )

    # ---- Background task support with TaskConfig ----

    if config.enable_background_tasks:
        try:
            from fastmcp.server.tasks.config import TaskConfig

            @mcp.tool(
                name="agent_background",
                task=TaskConfig(mode="optional"),
            )
            async def agent_background_tool(
                message: str,
                target: str | None = None,
                _router: AgentRouter = Depends(get_router),
                _adapter: Any = Depends(get_adapter),
                _tasks: TaskManager = Depends(get_task_manager),
            ) -> ToolResult:
                """Send a message as a background task.

                Returns immediately with a task ID queryable via ``task``.

                Args:
                    message: The message to send
                    target: Optional agent name
                """
                from fastmcp.server.dependencies import get_context

                bg_ctx = get_context()

                task_id = str(uuid4())
                tracker = await _tasks.create(task_id)
                resolved = _router.resolve(name=target, message=message)

                set_span_attribute("agentique.agent_name", resolved.name)
                set_span_attribute("agentique.task_id", task_id)

                bridge_ctx = BridgeContext.from_fastmcp_context(bg_ctx)

                chunks: list[str] = []
                idx = 0
                try:
                    async for event in _adapter.stream_message(
                        resolved.name, message, bridge_ctx,
                    ):
                        tracker.add_event(event)
                        idx += 1
                        if event.text:
                            chunks.append(event.text)

                    if not tracker.state.is_terminal:
                        tracker.transition(TaskState.completed)

                    set_span_attribute("agentique.task_state", tracker.state.value)

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
                    set_span_attribute("agentique.task_state", "failed")
                    raise

        except ImportError:
            # Fallback: use task=True if TaskConfig is not available
            try:
                from fastmcp.dependencies import Progress

                @mcp.tool(name="agent_background", task=True)
                async def agent_background_tool(
                    message: str,
                    target: str | None = None,
                    progress: Any = Progress(),
                    _router: AgentRouter = Depends(get_router),
                    _adapter: Any = Depends(get_adapter),
                    _tasks: TaskManager = Depends(get_task_manager),
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
                    tracker = await _tasks.create(task_id)
                    resolved = _router.resolve(name=target, message=message)
                    bridge_ctx = BridgeContext.from_fastmcp_context(bg_ctx)

                    chunks: list[str] = []
                    idx = 0
                    try:
                        async for event in _adapter.stream_message(
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

    # ---- Artifact resources ----

    @mcp.resource("a2a://{task_id}/artifacts/{artifact_id}")
    async def artifact_resource(task_id: str, artifact_id: str) -> str:
        """Serve a captured agent artifact.

        Artifacts are registered automatically when an agent emits an
        artifact event. The URI scheme is::

            a2a://{task_id}/artifacts/{artifact_id}

        Args:
            task_id: The task that produced the artifact.
            artifact_id: The artifact identifier within that task.
        """
        content = tasks.get_artifact(task_id, artifact_id)
        if content is None:
            return json.dumps({
                "error": f"Artifact '{artifact_id}' not found in task '{task_id}'",
                "task_id": task_id,
                "artifact_id": artifact_id,
            })
        return content

    @mcp.resource("a2a://{task_id}/artifacts")
    async def task_artifacts_catalog(task_id: str) -> str:
        """List all artifacts captured for a task.

        Args:
            task_id: The task to list artifacts for.
        """
        artifacts = tasks.list_artifacts(task_id)
        return json.dumps(
            {"task_id": task_id, "artifacts": artifacts, "count": len(artifacts)},
            indent=2,
        )

    return mcp


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def mount_bridge(
    parent: FastMCP,
    *,
    agents: list[AgentInfo],
    adapter: Any,
    namespace: str,
    config: AgentiqueConfig | None = None,
    events: AsyncEventEmitter | None = None,
    task_store: TaskStore | None = None,
) -> FastMCP:
    """Create and mount a sub-bridge under *parent* with namespace isolation.

    This enables multi-protocol architectures where separate adapter
    bridges (A2A, HTTP, local) are composed under a single MCP server::

        main = FastMCP("Main")
        mount_bridge(main, agents=a2a_agents, adapter=a2a_adapter, namespace="a2a")
        mount_bridge(main, agents=http_agents, adapter=http_adapter, namespace="http")

    Args:
        parent: The parent ``FastMCP`` server to mount into.
        agents: Agent descriptors for this bridge.
        adapter: Pre-configured adapter for this bridge.
        namespace: Namespace prefix for all tools in this bridge.
        config: Optional server configuration override.
        events: Optional event emitter override.
        task_store: Optional storage backend override.

    Returns:
        The child ``FastMCP`` server that was mounted.
    """
    child = create_server(
        agents=agents,
        adapter=adapter,
        config=config or AgentiqueConfig(
            name=f"Agentique-{namespace}",
            enable_background_tasks=False,
            prefetch_cards=False,
        ),
        events=events,
        task_store=task_store,
    )

    parent.mount(child, namespace=namespace)
    return child


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
