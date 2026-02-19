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
from .bridge.output_models import (
    AgentInspectOutput,
    AgentListOutput,
    AgentMessageOutput,
    AgentSummary,
    ErrorOutput,
    SubAgentSummary,
    TaskListOutput,
    TaskStatusOutput,
    TaskSummary,
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
    tool_mapper: Any | None = None,
    resources_as_tools: bool = False,
    prompts_as_tools: bool = False,
    extra_providers: list[Any] | None = None,
    session_state_store: Any | None = None,
    sampling_handler: Any | None = None,
    sampling_handler_behavior: str = "fallback",
    health_check_interval: float | None = None,
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
        tool_mapper: Optional ``ToolMapper`` instance controlling how agent
            capabilities are mapped to MCP tool definitions. When ``None``,
            the default behaviour (one tool per agent) is used.
        resources_as_tools: When ``True``, adds a ``ResourcesAsTools``
            transform so clients that only support tools can still access
            all MCP resources (including artifact resources) via tool calls.
        prompts_as_tools: When ``True``, adds a ``PromptsAsTools`` transform
            so registered agent prompts are callable as tools.
        extra_providers: Optional list of additional FastMCP ``Provider``
            instances to compose alongside ``AgentProvider``. Use this to
            combine the A2A agent bridge with, for example, an
            ``OpenAPIProvider`` or a custom ``FileSystemProvider``::

                from fastmcp.server.openapi import OpenAPIProvider

                server = create_server(
                    agents=agents,
                    extra_providers=[OpenAPIProvider(spec_url="https://…")],
                )

        session_state_store: Pluggable ``AsyncKeyValue`` backend for FastMCP
            session state (``ctx.get_state``/``ctx.set_state``).  Defaults to
            an in-process ``MemoryStore``.  Pass a Redis or DynamoDB store to
            enable persistent, horizontally-scalable session state::

                from key_value.aio.stores.redis import RedisStore
                store = RedisStore(url="redis://localhost:6379")
                server = create_server(agents=agents, session_state_store=store)

        sampling_handler: Optional FastMCP sampling handler used when the
            connected MCP client does not support sampling
            (``sampling_handler_behavior="fallback"``, the default).  Pass an
            ``AnthropicSamplingHandler`` or ``OpenAISamplingHandler`` instance
            to ensure ``LLMRouter.aselect()`` always has access to an LLM::

                from fastmcp.client.sampling.handlers.anthropic import (
                    AnthropicSamplingHandler,
                )
                handler = AnthropicSamplingHandler(
                    default_model="claude-sonnet-4-6"
                )
                server = create_server(agents=agents, sampling_handler=handler)

        sampling_handler_behavior: Controls when *sampling_handler* is used.
            ``"fallback"`` (default) — use the handler only when the client
            doesn't support sampling.  ``"always"`` — bypass the client
            entirely and always use the handler.
        health_check_interval: When set, starts a background health-check
            loop via the composed lifespan, calling ``adapter.health_check()``
            every *health_check_interval* seconds.

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
    pool: A2AClientPool | None = None
    if adapter is None:
        agent_map = {a.name: a for a in agent_list}
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
        tool_mapper=tool_mapper,
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

    # Build server — compose AgentProvider with any extra providers
    all_providers: list[Any] = [provider]
    if extra_providers:
        all_providers.extend(extra_providers)

    # Build composed lifespan for resource management and cleanup
    from .bridge.lifespans import compose_lifespans, make_cleanup_lifespan
    from .bridge.dependencies import clear as _clear_deps

    cleanup_ls = make_cleanup_lifespan(pool, clear_deps_fn=_clear_deps)
    health_ls = None
    if health_check_interval is not None:
        from .bridge.lifespans import make_health_monitor_lifespan
        health_ls = make_health_monitor_lifespan(adapter, interval=health_check_interval)
    server_lifespan = compose_lifespans(cleanup_ls, health_ls)

    # Build server kwargs — pass optional FastMCP parameters only when set
    _server_kwargs: dict[str, Any] = {"providers": all_providers}
    if server_lifespan is not None:
        _server_kwargs["lifespan"] = server_lifespan
    if session_state_store is not None:
        _server_kwargs["session_state_store"] = session_state_store
    if sampling_handler is not None:
        _server_kwargs["sampling_handler"] = sampling_handler
        _server_kwargs["sampling_handler_behavior"] = sampling_handler_behavior

    mcp = FastMCP(config.name, **_server_kwargs)

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

    # Server-bound transforms (require the server instance to be built first)
    if resources_as_tools:
        try:
            from fastmcp.server.transforms import ResourcesAsTools
            mcp.add_transform(ResourcesAsTools(mcp))
        except Exception:
            logger.warning("ResourcesAsTools transform not available in this FastMCP version")

    if prompts_as_tools:
        try:
            from fastmcp.server.transforms import PromptsAsTools
            mcp.add_transform(PromptsAsTools(mcp))
        except Exception:
            logger.warning("PromptsAsTools transform not available in this FastMCP version")

    # ---- Core tools (using Depends() for dependency injection) ----

    # Compute TaskConfig for the main agent tool: use "optional" mode when
    # background tasks are enabled AND pydocket (fastmcp[tasks]) is installed.
    _agent_task_config: Any = False
    if config.enable_background_tasks:
        try:
            from fastmcp.server.tasks.config import TaskConfig as _AgentTC
            try:
                import docket as _docket_check  # noqa: F401
                _agent_task_config = _AgentTC(mode="optional")
            except ImportError:
                pass
        except ImportError:
            pass

    @mcp.tool(name="agent", task=_agent_task_config,
              output_schema=AgentMessageOutput.json_schema())
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
    ) -> ToolResult:
        """Send a message to an A2A agent and stream the response.

        Returns a structured payload including the task ID, resolved agent
        name, terminal task state, artifact URIs, and the response text.

        Args:
            message: The message to send to the agent
            target: Optional agent name (auto-routes if omitted)
            context_id: Optional context ID for conversation continuity
        """
        task_id = str(uuid4())
        resolved: Any = None

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
            ctx=ctx,
        )
        await _ctx_mgr.track_task(effective_ctx_id, task_id, ctx=ctx)

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

                # Handle auth-required state via elicitation
                if chunk.state == "auth-required":
                    auth_prompt = (
                        chunk.text
                        or "The agent requires authentication. "
                           "Please provide your credentials."
                    )
                    try:
                        elicit_result = await ctx.elicit(
                            auth_prompt,
                            response_type=None,
                        )
                        if hasattr(elicit_result, "data") and elicit_result.data:
                            credentials = str(elicit_result.data)
                            bridge_ctx = bridge_ctx.replace(
                                meta={
                                    **(bridge_ctx.meta or {}),
                                    "credentials": credentials,
                                }
                            )
                    except Exception as auth_exc:
                        logger.warning(
                            "Auth elicitation failed (%s); "
                            "continuing without credentials",
                            auth_exc,
                        )
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

        response_text = "".join(text_parts)
        artifact_uris = [
            f"a2a://{task_id}/artifacts/{a['artifact_id']}"
            for a in _tasks.list_artifacts(task_id)
        ]
        agent_name = resolved.name if resolved is not None else (target or "unknown")
        sc = AgentMessageOutput(
            agent=agent_name,
            task_id=task_id,
            context_id=effective_ctx_id,
            state=tracker.state.value,
            response=response_text,
            artifact_uris=artifact_uris,
            event_count=len(tracker.events),
            has_artifacts=bool(artifact_uris),
            mcp_related_task=task_id,
        ).model_dump()
        return ToolResult(
            content=response_text,
            structured_content=sc,
        )

    @mcp.tool(name="agents", output_schema=AgentListOutput.json_schema())
    def agents_tool(
        _router: AgentRouter = Depends(get_router),
    ) -> ToolResult:
        """List available A2A agents and their capabilities."""
        agent_list = _router.list_agents()
        sc = AgentListOutput(
            agents=[
                AgentSummary(
                    name=a.name,
                    base_url=a.base_url,
                    description=a.description,
                    skills=list(a.skills),
                )
                for a in agent_list
            ],
            count=len(agent_list),
        ).model_dump()
        return ToolResult(
            content=json.dumps(sc, indent=2),
            structured_content=sc,
        )

    @mcp.tool(name="task", output_schema=TaskStatusOutput.json_schema())
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
        if tracker is not None:
            sc = TaskStatusOutput(
                task_id=id,
                context_id=tracker.context_id,
                state=tracker.state.value,
                progress=tracker.progress,
                message=tracker.message,
                event_count=len(tracker.events),
                artifact_count=len(_tasks.list_artifacts(id)),
            ).model_dump()
            return ToolResult(
                content=json.dumps(sc, indent=2),
                structured_content=sc,
            )
        sc = ErrorOutput(error=f"Task {id} not found", code="TASK_NOT_FOUND").model_dump()
        return ToolResult(
            content=json.dumps(sc),
            structured_content=sc,
        )

    @mcp.tool(name="inspect", output_schema=AgentInspectOutput.json_schema())
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
                sc = ErrorOutput(
                    error=f"Agent '{name}' not found", code="AGENT_NOT_FOUND",
                ).model_dump()
                return ToolResult(
                    content=json.dumps(sc),
                    structured_content=sc,
                )
            hierarchy = card_parser.build_hierarchy(name, card)
            sc = AgentInspectOutput(
                root=hierarchy.root,
                agents={
                    k: SubAgentSummary(
                        name=v.name,
                        description=v.description,
                        skills=v.skills,
                        parent=v.parent,
                        depth=v.depth,
                    )
                    for k, v in hierarchy.agents.items()
                },
            ).model_dump()
            return ToolResult(
                content=json.dumps(sc, indent=2),
                structured_content=sc,
            )
        except Exception as exc:
            await ctx.error(f"Failed to inspect agent: {exc}")
            sc = ErrorOutput(error=str(exc), code="INSPECT_ERROR").model_dump()
            return ToolResult(
                content=json.dumps(sc),
                structured_content=sc,
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

    # ---- Capability advertisement resource ----

    @mcp.resource("a2a://capabilities")
    def capabilities_resource() -> str:
        """Gateway capability advertisement.

        Describes supported A2A extensions, transports, and feature flags.
        Since FastMCP 3.0 does not yet support ``capabilities.extensions``
        negotiation, this resource is the canonical way for MCP clients to
        discover what this gateway supports.
        """
        import agentique
        from .extensions import ALL_EXTENSION_URIS

        data: dict[str, Any] = {
            "gateway": config.name,
            "version": getattr(agentique, "__version__", "0.4.0"),
            "extensions": ALL_EXTENSION_URIS,
            "supported_transports": config.supported_transports or ["JSONRPC"],
            "features": {
                "background_tasks": config.enable_background_tasks,
                "elicitation": config.enable_elicitation,
                "tool_confirmation": config.enable_tool_confirmation,
                "push_notifications": True,
            },
            "session_state_persistent": session_state_store is not None,
        }
        return json.dumps(data, indent=2)

    # ---- Task catalog resource ----

    @mcp.resource("a2a://tasks")
    async def tasks_catalog() -> str:
        """List all tasks tracked in this gateway session.

        Returns a ``TaskListOutput`` JSON document with one ``TaskSummary``
        per tracked A2A task. Use the ``task`` tool for individual task details.
        """
        ids = await tasks.list_tasks()
        summaries: list[TaskSummary] = []
        for tid in ids:
            t = await tasks.get_or_none(tid)
            if t is not None:
                summaries.append(TaskSummary(
                    task_id=tid,
                    context_id=t.context_id,
                    state=t.state.value,
                    event_count=len(t.events),
                    artifact_count=len(tasks.list_artifacts(tid)),
                ))
        output = TaskListOutput(tasks=summaries, count=len(summaries))
        return output.model_dump_json(indent=2)

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
