"""FastMCP server factory for agentique.

Creates a fully configured FastMCP server that bridges MCP clients to
A2A (or other protocol) agents. The server exposes:

    - ``agent``  — send a message and return a structured response
    - ``agents`` — list available agents
    - ``task``   — query task state for transparency
    - ``inspect`` — view an agent's internal structure
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import inspect
import json
import logging
from typing import Any
from uuid import uuid4

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext, Depends
from fastmcp.tools.tool import TaskConfig, ToolResult
from pydantic import BaseModel

from .core.config import AgentiqueConfig
from .core.events import AsyncEventEmitter
from .core.types import (
    AgentHierarchy,
    AgentInfo,
    BridgeContext,
    StreamChunk,
    TaskState,
)
from .adapters.a2a import A2AAgentAdapter, A2AClientPool, A2ACardParser
from .bridge.context_manager import ContextManager
from .bridge.middleware import (
    ErrorMappingMiddleware,
    FastMCPBridgeMiddleware,
    LoggingMiddleware,
    MiddlewareChain,
)
from .bridge.provider import AgentProvider
from .bridge.router import AgentRouter
from .bridge.task_manager import TaskManager

logger = logging.getLogger(__name__)

try:
    from fastmcp.server.transforms import (
        Namespace,
        PromptsAsTools,
        ResourcesAsTools,
        ToolTransform,
        Visibility,
    )
except Exception:  # pragma: no cover - fastmcp is required
    Namespace = None  # type: ignore[assignment]
    ToolTransform = None  # type: ignore[assignment]
    Visibility = None  # type: ignore[assignment]
    ResourcesAsTools = None  # type: ignore[assignment]
    PromptsAsTools = None  # type: ignore[assignment]


class _ElicitedInput(BaseModel):
    response: str


def create_server(
    *,
    agents: list[AgentInfo] | None = None,
    router: AgentRouter | None = None,
    adapter: Any | None = None,
    client_factory: Any | None = None,
    config: AgentiqueConfig | None = None,
    events: AsyncEventEmitter | None = None,
    middleware: list[Any] | None = None,
    lifespan: Any | None = None,
) -> FastMCP:
    """Create a FastMCP server bridging MCP clients to agent backends.

    Args:
        agents: Agent descriptors to register.
        router: Pre-configured router (built from *agents* if absent).
        adapter: Pre-configured adapter (A2A adapter built if absent).
        client_factory: Legacy factory adapter (wrapped if provided).
        config: Server configuration.
        events: Event emitter for lifecycle hooks.
        middleware: Additional bridge middleware instances.
        lifespan: Optional external lifespan callable to compose.

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
            pool = A2AClientPool(
                timeout=config.default_timeout,
                card_path=config.a2a_card_path or None,
                supported_transports=config.parsed_a2a_supported_transports or None,
                use_client_preference=config.a2a_use_client_preference,
                extensions=config.parsed_a2a_extensions or None,
                push_notification_configs=_default_push_configs(config),
            )
        adapter = A2AAgentAdapter(
            agent_map,
            client_pool=pool,
            default_extensions=config.parsed_a2a_extensions,
            default_push_notification=_default_push_config(config),
        )

    # Build provider
    card_parser = A2ACardParser()
    provider = AgentProvider(
        agent_list,
        adapter,
        card_parser=card_parser,
        cache_ttl=config.cache_ttl,
        prefetch_cards=config.prefetch_cards,
    )

    # Build middleware chain
    chain = MiddlewareChain()
    chain.add(LoggingMiddleware())
    chain.add(ErrorMappingMiddleware())
    if middleware:
        for mw in middleware:
            chain.add(mw)

    # Build shared state managers
    context_mgr = ContextManager()
    tasks = TaskManager()

    # Dependency helpers for FastMCP Depends()
    def get_router() -> AgentRouter:
        return router

    def get_adapter() -> Any:
        return adapter

    def get_task_manager() -> TaskManager:
        return tasks

    def get_provider() -> AgentProvider:
        return provider

    def get_context_manager() -> ContextManager:
        return context_mgr

    def get_events() -> AsyncEventEmitter:
        return emitter

    @asynccontextmanager
    async def agentique_lifespan(_: Any):
        await emitter.emit("server.start")
        try:
            yield {}
        finally:
            await emitter.emit("server.stop")

    combined_lifespan = agentique_lifespan
    if callable(lifespan):
        try:
            from fastmcp.utilities.lifespan import combine_lifespans

            combined_lifespan = combine_lifespans(agentique_lifespan, lifespan)
        except Exception:
            logger.warning(
                "Failed to compose custom lifespan; using agentique lifespan only",
                exc_info=True,
            )

    # Build server
    mcp = FastMCP(
        config.name,
        providers=[provider],
        lifespan=combined_lifespan,
    )

    # Bridge middleware integration at the FastMCP layer.
    if config.enable_fastmcp_middleware:
        mcp.add_middleware(FastMCPBridgeMiddleware(chain))

    _configure_transforms(mcp, config, provider)

    # ---- Core tools ----

    @mcp.tool(
        name="agent",
        output_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "context_id": {"type": "string"},
                "agent": {"type": "string"},
                "state": {"type": "string"},
                "response_text": {"type": "string"},
                "event_count": {"type": "integer"},
            },
            "required": ["task_id", "context_id", "agent", "state", "response_text"],
        },
    )
    async def agent_tool(
        message: str,
        target: str | None = None,
        context_id: str | None = None,
        ctx: Context = CurrentContext(),
        router_dep: AgentRouter = Depends(get_router),
        adapter_dep: Any = Depends(get_adapter),
        tasks_dep: TaskManager = Depends(get_task_manager),
        context_dep: ContextManager = Depends(get_context_manager),
        emitter_dep: AsyncEventEmitter = Depends(get_events),
    ) -> ToolResult:
        """Send a message to an agent and return structured task output.

        Args:
            message: The message to send to the agent.
            target: Optional agent name (auto-routes if omitted).
            context_id: Optional context ID for conversation continuity.
        """
        task_id = str(uuid4())

        session_id = getattr(ctx, "session_id", None)
        effective_ctx_id = await context_dep.resolve_context(
            session_id=session_id,
            context_id=context_id,
            task_id=task_id,
        )
        await context_dep.track_task(effective_ctx_id, task_id)

        tracker = await tasks_dep.create(task_id, effective_ctx_id)
        hierarchy = AgentHierarchy(root=target or "auto")

        history = tasks_dep.get_conversation_history(effective_ctx_id)
        await emitter_dep.emit("task.created", task_id=task_id)

        resolved = await router_dep.aresolve(name=target, message=message, ctx=ctx)
        _set_span_attributes(
            **{
                "agentique.agent_name": resolved.name,
                "agentique.protocol": "a2a",
                "agentique.task_id": task_id,
            }
        )

        bridge_ctx = BridgeContext.from_fastmcp_context(ctx)
        if history:
            bridge_ctx = bridge_ctx.replace(conversation_history=history)

        text_chunks: list[str] = []
        elicitation: dict[str, Any] | None = None

        try:
            index = 0
            async for event in adapter_dep.stream_message(
                resolved.name,
                message,
                bridge_ctx,
            ):
                tracker.add_event(event)
                chunk = StreamChunk.from_event(resolved.name, index, event)
                index += 1

                if chunk.state:
                    _set_span_attributes(
                        **{
                            "agentique.task_state": chunk.state,
                        }
                    )

                if chunk.branch:
                    parts = chunk.branch.split(".")
                    for i, part in enumerate(parts):
                        parent = parts[i - 1] if i > 0 else None
                        if part not in hierarchy.agents:
                            hierarchy.add_agent(part, parent=parent)

                if chunk.kind in {"status", "task"}:
                    if chunk.text:
                        branch = f" [{chunk.branch}]" if chunk.branch else ""
                        await ctx.info(f"{branch} {chunk.text}")
                    if chunk.progress is not None:
                        await ctx.report_progress(int(chunk.progress), 100)

                if (
                    config.enable_elicitation
                    and chunk.state in {TaskState.input_required.value, TaskState.auth_required.value}
                    and hasattr(ctx, "elicit")
                ):
                    elicitation = await _run_elicitation(
                        ctx,
                        state=chunk.state,
                        prompt=chunk.text or "Agent requires additional input to continue.",
                    )
                    if chunk.state == TaskState.auth_required.value:
                        tracker.transition(TaskState.auth_required, chunk.text)
                    else:
                        tracker.transition(TaskState.input_required, chunk.text)

                if chunk.state == TaskState.rejected.value:
                    tracker.transition(TaskState.rejected, chunk.text)

                if chunk.kind in {"message", "artifact"} and chunk.text:
                    text_chunks.append(chunk.text)
                    await emitter_dep.emit("stream.chunk", chunk=chunk)

        except Exception as exc:
            tracker.transition(TaskState.failed, str(exc))
            await emitter_dep.emit("error", task_id=task_id, error=str(exc))
            raise
        finally:
            if not tracker.state.is_terminal and tracker.state not in {
                TaskState.input_required,
                TaskState.auth_required,
            }:
                tracker.transition(TaskState.completed)

            response_text = "".join(text_chunks).strip()
            if response_text and effective_ctx_id:
                tasks_dep.append_conversation(effective_ctx_id, message, response_text)

            if hierarchy.agents:
                await tasks_dep.set_hierarchy(task_id, hierarchy)

            await emitter_dep.emit("task.completed", task_id=task_id)

        response_text = "".join(text_chunks).strip()
        if not response_text and tracker.message:
            response_text = tracker.message

        structured: dict[str, Any] = {
            "task_id": task_id,
            "context_id": effective_ctx_id,
            "agent": resolved.name,
            "state": tracker.state.value,
            "response_text": response_text,
            "event_count": len(tracker.events),
        }
        if elicitation:
            structured["elicitation"] = elicitation

        return ToolResult(
            content=response_text,
            structured_content=structured,
        )

    @mcp.tool(name="agents")
    def agents_tool(
        router_dep: AgentRouter = Depends(get_router),
    ) -> list[dict[str, Any]]:
        """List available A2A agents and their capabilities."""
        return [a.to_dict() for a in router_dep.list_agents()]

    @mcp.tool(name="task")
    async def task_tool(
        id: str,
        tasks_dep: TaskManager = Depends(get_task_manager),
    ) -> dict[str, Any]:
        """Query the state of a task.

        Args:
            id: The task ID to query.
        """
        tracker = await tasks_dep.get_or_none(id)
        if tracker:
            return tracker.to_dict()
        return {"error": f"Task {id} not found", "task_id": id}

    @mcp.tool(name="inspect")
    async def inspect_tool(
        name: str,
        provider_dep: AgentProvider = Depends(get_provider),
    ) -> dict[str, Any]:
        """Inspect an agent's internal structure (sub-agents, tools).

        Args:
            name: The agent name to inspect.
        """
        card = await provider_dep.get_agent_card(name)
        if not card:
            return {"error": f"Agent '{name}' not found", "agent": name}
        hierarchy = card_parser.build_hierarchy(name, card)
        return hierarchy.to_dict()

    @mcp.tool(name="enable_components")
    async def enable_components_tool(
        names: list[str] | None = None,
        tags: list[str] | None = None,
        components: list[str] | None = None,
        ctx: Context = CurrentContext(),
    ) -> dict[str, Any]:
        """Enable components for the current session using visibility controls."""
        ctx.enable_components(
            names=set(names) if names else None,
            tags=set(tags) if tags else None,
            components=_component_set(components),
        )
        return {
            "status": "enabled",
            "names": names or [],
            "tags": tags or [],
            "components": components or [],
        }

    @mcp.tool(name="disable_components")
    async def disable_components_tool(
        names: list[str] | None = None,
        tags: list[str] | None = None,
        components: list[str] | None = None,
        ctx: Context = CurrentContext(),
    ) -> dict[str, Any]:
        """Disable components for the current session using visibility controls."""
        ctx.disable_components(
            names=set(names) if names else None,
            tags=set(tags) if tags else None,
            components=_component_set(components),
        )
        return {
            "status": "disabled",
            "names": names or [],
            "tags": tags or [],
            "components": components or [],
        }

    # ---- Background task support ----

    if config.enable_background_tasks:
        try:
            from fastmcp.dependencies import Progress

            @mcp.tool(
                name="agent_background",
                task=TaskConfig(
                    mode="optional",
                    poll_interval=config.background_task_poll_interval,
                ),
            )
            async def agent_background_tool(
                message: str,
                target: str | None = None,
                progress: Progress = Progress(),
                ctx: Context = CurrentContext(),
                router_dep: AgentRouter = Depends(get_router),
                adapter_dep: Any = Depends(get_adapter),
                tasks_dep: TaskManager = Depends(get_task_manager),
            ) -> ToolResult:
                """Send a message as a background task.

                Args:
                    message: The message to send.
                    target: Optional agent name.
                """
                await progress.set_total(3)
                await progress.set_message("Resolving agent...")

                task_id = str(uuid4())
                tracker = await tasks_dep.create(task_id)
                resolved = router_dep.resolve(name=target, message=message)
                await progress.increment()

                await progress.set_message("Sending request...")
                bridge_ctx = BridgeContext.from_fastmcp_context(ctx)

                chunks: list[str] = []
                idx = 0
                async for event in adapter_dep.stream_message(
                    resolved.name,
                    message,
                    bridge_ctx,
                ):
                    tracker.add_event(event)
                    idx += 1
                    if event.text:
                        chunks.append(event.text)

                await progress.increment()

                if not tracker.state.is_terminal:
                    tracker.transition(TaskState.completed)

                await progress.set_message("Complete")
                await progress.increment()

                content = "".join(chunks).strip()
                return ToolResult(
                    content=content,
                    structured_content={
                        "task_id": task_id,
                        "agent": resolved.name,
                        "state": tracker.state.value,
                        "event_count": idx,
                        "response_text": content,
                    },
                )

        except ImportError:
            pass

    # ---- Catalog resource ----

    @mcp.resource("a2a://agents")
    def agents_catalog(
        router_dep: AgentRouter = Depends(get_router),
    ) -> str:
        """Catalog of available A2A agents."""
        return json.dumps(
            {"agents": [a.to_dict() for a in router_dep.list_agents()]},
            indent=2,
        )

    return mcp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _configure_transforms(
    mcp: FastMCP,
    config: AgentiqueConfig,
    provider: AgentProvider,
) -> None:
    if Namespace is not None and config.component_namespace:
        mcp.add_transform(Namespace(config.component_namespace))

    if ToolTransform is not None and config.tool_transformations:
        mcp.add_transform(ToolTransform(config.tool_transformations))

    if Visibility is not None and config.parsed_disabled_component_names:
        mcp.add_transform(
            Visibility(False, names=config.parsed_disabled_component_names),
        )

    if ResourcesAsTools is not None and config.enable_resources_as_tools:
        mcp.add_transform(ResourcesAsTools(provider))

    if PromptsAsTools is not None and config.enable_prompts_as_tools:
        mcp.add_transform(PromptsAsTools(provider))


def _default_push_config(config: AgentiqueConfig) -> Any | None:
    if not config.a2a_push_notification_url:
        return None
    try:
        from a2a.types import PushNotificationConfig

        kwargs: dict[str, Any] = {
            "url": config.a2a_push_notification_url,
        }
        if config.a2a_push_notification_token:
            kwargs["token"] = config.a2a_push_notification_token
        if config.a2a_push_notification_id:
            kwargs["id"] = config.a2a_push_notification_id
        if config.a2a_push_notification_auth:
            kwargs["authentication"] = config.a2a_push_notification_auth
        return PushNotificationConfig(**kwargs)
    except Exception:
        logger.warning("Invalid default A2A push notification config", exc_info=True)
        return None


def _default_push_configs(config: AgentiqueConfig) -> list[Any] | None:
    cfg = _default_push_config(config)
    if cfg is None:
        return None
    return [cfg]


async def _run_elicitation(
    ctx: Context,
    *,
    state: str,
    prompt: str,
) -> dict[str, Any]:
    """Run structured elicitation for input/auth-required task states."""
    result = await ctx.elicit(
        f"[{state}] {prompt}",
        response_type=_ElicitedInput,
    )
    return {
        "action": getattr(result, "action", "unknown"),
        "content": getattr(result, "content", None),
    }


def _component_set(components: list[str] | None) -> set[str] | None:
    if not components:
        return None
    allowed = {"tool", "resource", "template", "prompt"}
    cleaned = {c for c in components if c in allowed}
    return cleaned or None


def _set_span_attributes(**attributes: Any) -> None:
    """Set OpenTelemetry span attributes if tracing is available."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:
        # Tracing is optional; never fail the tool flow.
        return


def _wrap_legacy_factory(factory: Any) -> A2AClientPool:
    """Wrap a legacy client factory into an ``A2AClientPool`` interface."""

    class _Wrapper(A2AClientPool):
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            super().__init__()

        async def get(self, base_url: str) -> Any:
            result = self._inner.get(base_url)
            return await result if inspect.isawaitable(result) else result

        async def close(self) -> None:
            close_fn = getattr(self._inner, "aclose", None) or getattr(self._inner, "close", None)
            if callable(close_fn):
                result = close_fn()
                if inspect.isawaitable(result):
                    await result

    return _Wrapper(factory)
