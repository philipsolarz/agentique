from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan
from fastmcp.tools.tool import ToolResult

from .bridge import RouterBridge
from .models import AgentDescriptor
from .router import AgentRouter


def create_server(
    *,
    name: str = "Agentique Bridge",
    agents: list[AgentDescriptor] | None = None,
    router: AgentRouter | None = None,
    client_factory: Any | None = None,
) -> FastMCP:
    """Create a FastMCP server that routes requests to A2A agents."""

    active_router = router or AgentRouter(agents or [])
    bridge = RouterBridge(active_router, client_factory=client_factory)

    @lifespan
    async def bridge_lifespan(server: FastMCP):
        # Pre-discover agent capabilities at startup (optional, best-effort)
        # This reduces latency on first tool invocation
        for agent_desc in active_router.list_agents():
            try:
                # Fetch agent card without context (no logging)
                card = await bridge.get_agent_card(agent_desc.name)
                # Pre-parse component definitions
                tool_defs, prompt_defs = _extract_component_defs(card)
                # Note: Actual registration happens lazily on first use with proper context
            except Exception:
                # Silently ignore discovery failures at startup
                # Discovery will be retried with proper error reporting on first use
                pass

        yield {"router": active_router}
        await bridge.aclose()

    mcp = FastMCP(name, lifespan=bridge_lifespan)

    registered_tool_names: set[str] = set()
    registered_prompt_names: set[str] = set()
    tool_registry: dict[tuple[str, str], str] = {}
    prompt_registry: dict[tuple[str, str], str] = {}
    registration_lock = asyncio.Lock()

    def _mark_tool(name: str) -> None:
        registered_tool_names.add(name)

    def _mark_prompt(name: str) -> None:
        registered_prompt_names.add(name)

    def _read_field(payload: Any, *names: str) -> Any:
        for name in names:
            if isinstance(payload, dict) and name in payload:
                return payload[name]
            if hasattr(payload, name):
                return getattr(payload, name)
        return None

    def _normalize_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _coerce_card_payload(card: Any) -> dict[str, Any]:
        if card is None:
            return {}
        if isinstance(card, dict):
            return dict(card)
        payload: dict[str, Any] = {}
        extra = getattr(card, "model_extra", None) or getattr(card, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if hasattr(card, "model_dump"):
            try:
                payload.update(card.model_dump())
            except Exception:
                pass
        elif hasattr(card, "dict"):
            try:
                payload.update(card.dict())
            except Exception:
                pass
        return payload

    def _extract_extensions(card: Any) -> list[Any]:
        if card is None:
            return []
        if isinstance(card, dict):
            capabilities = card.get("capabilities") or {}
            return _normalize_list(capabilities.get("extensions") or card.get("extensions"))
        capabilities = getattr(card, "capabilities", None)
        if capabilities is not None:
            extensions = getattr(capabilities, "extensions", None)
            if extensions:
                return list(extensions)
        extensions = getattr(card, "extensions", None)
        if extensions:
            return list(extensions)
        return []

    def _extract_component_defs(card: Any) -> tuple[list[Any], list[Any]]:
        payload = _coerce_card_payload(card)
        tools = _normalize_list(payload.get("mcp_tools") or payload.get("mcpTools"))
        prompts = _normalize_list(payload.get("mcp_prompts") or payload.get("mcpPrompts"))
        for extension in _extract_extensions(card):
            params = _read_field(extension, "params", "parameters")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = None
            if isinstance(params, dict):
                tools.extend(_normalize_list(params.get("mcp_tools") or params.get("mcpTools")))
                prompts.extend(_normalize_list(params.get("mcp_prompts") or params.get("mcpPrompts")))
        return tools, prompts

    def _json_schema_type(schema: Any) -> Any:
        if not isinstance(schema, dict):
            return Any
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            non_null = [item for item in schema_type if item != "null"]
            if not non_null:
                return Any
            base = _json_schema_type({"type": non_null[0]})
            return base | None
        if schema_type == "string":
            return str
        if schema_type == "integer":
            return int
        if schema_type == "number":
            return float
        if schema_type == "boolean":
            return bool
        if schema_type == "array":
            return list[Any]
        if schema_type == "object":
            return dict[str, Any]
        return Any

    def _parameters_from_schema(schema: Any) -> list[inspect.Parameter]:
        if not isinstance(schema, dict):
            return []
        schema_type = schema.get("type")
        if schema_type not in (None, "object"):
            return []
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict) or not properties:
            return []
        required = set(schema.get("required") or [])
        params: list[inspect.Parameter] = []
        for name, prop in properties.items():
            annotation = _json_schema_type(prop)
            if name in required:
                default = prop.get("default", inspect._empty)
            else:
                default = prop.get("default", None)
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=annotation,
                )
            )
        return params

    def _parameters_from_arguments(arguments: Any) -> list[inspect.Parameter]:
        if not isinstance(arguments, list):
            return []
        params: list[inspect.Parameter] = []
        for arg in arguments:
            if not isinstance(arg, dict):
                continue
            name = _read_field(arg, "name", "arg", "id")
            if not name:
                continue
            schema = _read_field(arg, "schema", "input_schema", "inputSchema")
            if schema is None:
                arg_type = _read_field(arg, "type")
                if arg_type:
                    schema = {"type": arg_type}
            annotation = _json_schema_type(schema or {})
            required = bool(_read_field(arg, "required"))
            if required:
                default = arg.get("default", inspect._empty)
            else:
                default = arg.get("default", None)
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=annotation,
                )
            )
        return params

    def _resolve_component_name(base: str, agent_name: str, used: set[str]) -> str:
        if base not in used:
            return base
        candidate = f"{agent_name}_{base}"
        if candidate not in used:
            return candidate
        index = 2
        while f"{candidate}_{index}" in used:
            index += 1
        return f"{candidate}_{index}"

    def _render_prompt(prompt_def: Any, prompt_name: str, agent_name: str, args: dict[str, Any]) -> str:
        template = _read_field(prompt_def, "template", "prompt", "text", "content")
        if isinstance(template, str):
            try:
                return template.format(**args)
            except Exception:
                return template
        messages = _read_field(prompt_def, "messages")
        if isinstance(messages, list):
            rendered: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role", "user")
                content = message.get("content") or message.get("text")
                if isinstance(content, str):
                    try:
                        content = content.format(**args)
                    except Exception:
                        pass
                    rendered.append(f"{role}: {content}")
            if rendered:
                return "\n".join(rendered)
        return f"Prompt '{prompt_name}' for agent '{agent_name}'."

    def _build_tool_function(
        *,
        agent_name: str,
        tool_name: str,
        registered_name: str,
        description: str | None,
        parameters: list[inspect.Parameter],
    ):
        async def tool_fn(*, ctx: Context = CurrentContext(), **kwargs: Any) -> dict[str, Any]:
            payload = {"tool": tool_name, "arguments": kwargs}
            message = json.dumps(payload, default=str)
            return await _route_message(
                message,
                agent=agent_name,
                metadata={"mcp_tool_call": payload},
                ctx=ctx,
            )

        tool_fn.__name__ = registered_name
        tool_fn.__doc__ = description or f"Proxy to agent '{agent_name}' tool '{tool_name}'."
        ctx_param = inspect.Parameter(
            "ctx",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=CurrentContext(),
            annotation=Context,
        )
        signature = inspect.Signature(parameters=[*parameters, ctx_param])
        tool_fn.__signature__ = signature
        annotations: dict[str, Any] = {"return": dict[str, Any]}
        for param in signature.parameters.values():
            if param.annotation is not inspect._empty:
                annotations[param.name] = param.annotation
        tool_fn.__annotations__ = annotations
        return tool_fn

    def _build_prompt_function(
        *,
        agent_name: str,
        prompt_name: str,
        registered_name: str,
        description: str | None,
        prompt_def: Any,
        parameters: list[inspect.Parameter],
    ):
        def prompt_fn(**kwargs: Any) -> str:
            return _render_prompt(prompt_def, prompt_name, agent_name, kwargs)

        prompt_fn.__name__ = registered_name
        prompt_fn.__doc__ = description or f"Prompt '{prompt_name}' from agent '{agent_name}'."
        signature = inspect.Signature(parameters=parameters)
        prompt_fn.__signature__ = signature
        annotations: dict[str, Any] = {"return": str}
        for param in signature.parameters.values():
            if param.annotation is not inspect._empty:
                annotations[param.name] = param.annotation
        prompt_fn.__annotations__ = annotations
        return prompt_fn

    async def _route_message(
        message: str,
        *,
        ctx: Context,
        agent: str | None = None,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> ToolResult:
        start_time = time.time()

        try:
            await ctx.debug(f"Routing request: agent={agent}, skill={skill}")
            await ctx.info("Routing message to A2A agent.")
            await ctx.report_progress(0, 100, "Connecting to agent")

            chunks = []
            task_updates = []
            has_received_content = False
            resolved_agent_name = None

            async for chunk in bridge.stream(
                message,
                ctx=ctx,
                agent=agent,
                skill=skill,
                metadata=metadata,
                configuration=configuration,
            ):
                chunks.append(chunk)

                # Capture the resolved agent name
                if resolved_agent_name is None:
                    resolved_agent_name = chunk.agent

                # Track different types of events for progress
                if chunk.kind == "task" and chunk.text:
                    task_updates.append(chunk.text)
                    # Progress from 10 to 90 based on task updates
                    progress = min(10 + len(task_updates) * 15, 90)
                    await ctx.report_progress(progress, 100, f"Agent task: {chunk.text}")
                    await ctx.info(f"Agent working: {chunk.text}")

                elif chunk.kind == "status" and chunk.text:
                    await ctx.report_progress(50, 100, f"Status: {chunk.text}")
                    await ctx.info(f"Agent status: {chunk.text}")

                elif chunk.kind in {"message", "artifact"}:
                    if not has_received_content:
                        await ctx.report_progress(80, 100, "Receiving response")
                        has_received_content = True

            if not chunks:
                await ctx.warning("Agent returned no response chunks")
                return ToolResult(
                    content="No response received from agent",
                    structured_content={
                        "agent": agent or "unknown",
                        "text": "",
                        "events": [],
                    },
                )

            final_agent = resolved_agent_name or agent or "unknown"
            all_text = " ".join(c.text for c in chunks if c.text).strip()
            elapsed_ms = int((time.time() - start_time) * 1000)

            await ctx.debug(f"Received {len(chunks)} chunks from agent '{final_agent}'")
            await ctx.report_progress(100, 100, "Response complete")

            # Get event type counts for structured content
            event_types = {}
            for chunk in chunks:
                event_types[chunk.kind] = event_types.get(chunk.kind, 0) + 1

            return ToolResult(
                content=all_text,  # What the LLM/user sees
                structured_content={  # For programmatic access
                    "agent": final_agent,
                    "event_count": len(chunks),
                    "event_types": event_types,
                    "events": [c.to_dict() for c in chunks],
                },
                meta={  # Runtime metadata
                    "execution_time_ms": elapsed_ms,
                    "chunk_count": len(chunks),
                    "task_updates": len(task_updates),
                    "requested_agent": agent,
                    "requested_skill": skill,
                },
            )

        except Exception as exc:
            await ctx.error(f"A2A agent error: {exc}")
            agent_name = agent or "unknown agent"
            raise ToolError(f"Failed to communicate with agent '{agent_name}': {exc}") from exc

    async def _discover_agent_components(agent_name: str, ctx: Context) -> tuple[list[str], list[str]]:
        await ctx.debug(f"Discovering components for agent '{agent_name}'")

        # Check session cache first
        cache_key = f"agent_card_{agent_name}"
        cached_card = await ctx.get_state(cache_key)

        if cached_card:
            await ctx.debug(f"Using cached agent card for '{agent_name}'")
            card = cached_card
        else:
            try:
                card = await bridge.get_agent_card(agent_name)
                # Cache for this session
                await ctx.set_state(cache_key, card)
                await ctx.debug(f"Cached agent card for '{agent_name}'")
            except Exception as exc:
                await ctx.warning(f"Failed to fetch agent card for '{agent_name}': {exc}")
                return [], []

        tool_defs, prompt_defs = _extract_component_defs(card)

        await ctx.debug(f"Found {len(tool_defs)} tool definitions and {len(prompt_defs)} prompt definitions")

        if not tool_defs and not prompt_defs:
            return [], []

        new_tools: list[str] = []
        new_prompts: list[str] = []

        async with registration_lock:
            for tool_def in tool_defs:
                tool_name = _read_field(tool_def, "name", "tool_name", "toolName", "id")
                if not tool_name:
                    continue
                key = (agent_name, tool_name)
                if key in tool_registry:
                    continue
                description = _read_field(tool_def, "description", "summary", "title")
                arguments = _read_field(tool_def, "arguments", "args")
                schema = _read_field(tool_def, "input_schema", "inputSchema", "schema", "parameters")
                params = _parameters_from_arguments(arguments) or _parameters_from_schema(schema)
                if not params:
                    params = [
                        inspect.Parameter(
                            "payload",
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            default=None,
                            annotation=dict[str, Any],
                        )
                    ]
                registered_name = _resolve_component_name(tool_name, agent_name, registered_tool_names)
                tool_fn = _build_tool_function(
                    agent_name=agent_name,
                    tool_name=tool_name,
                    registered_name=registered_name,
                    description=description,
                    parameters=params,
                )
                try:
                    mcp.add_tool(tool_fn)
                except Exception as exc:
                    await ctx.warning(
                        f"Failed to register tool '{registered_name}' from agent '{agent_name}': {exc}"
                    )
                    continue
                registered_tool_names.add(registered_name)
                tool_registry[key] = registered_name
                new_tools.append(registered_name)

            for prompt_def in prompt_defs:
                prompt_name = _read_field(prompt_def, "name", "prompt_name", "promptName", "id")
                if not prompt_name:
                    continue
                key = (agent_name, prompt_name)
                if key in prompt_registry:
                    continue
                description = _read_field(prompt_def, "description", "summary", "title")
                arguments = _read_field(prompt_def, "arguments", "args")
                schema = _read_field(prompt_def, "input_schema", "inputSchema", "schema", "parameters")
                params = _parameters_from_arguments(arguments) or _parameters_from_schema(schema)
                registered_name = _resolve_component_name(prompt_name, agent_name, registered_prompt_names)
                prompt_fn = _build_prompt_function(
                    agent_name=agent_name,
                    prompt_name=prompt_name,
                    registered_name=registered_name,
                    description=description,
                    prompt_def=prompt_def,
                    parameters=params,
                )
                try:
                    mcp.add_prompt(prompt_fn)
                except Exception as exc:
                    await ctx.warning(
                        f"Failed to register prompt '{registered_name}' from agent '{agent_name}': {exc}"
                    )
                    continue
                registered_prompt_names.add(registered_name)
                prompt_registry[key] = registered_name
                new_prompts.append(registered_name)

        return new_tools, new_prompts

    @mcp.tool
    async def a2a_send(
        message: str,
        agent: str | None = None,
        skill: str | None = None,
        continue_conversation: bool = False,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
        ctx: Context = CurrentContext(),
    ) -> ToolResult:
        """Route a message to an A2A agent and return its response.

        Uses streaming internally to keep the connection alive during long-running
        operations, while still returning the final aggregated result.

        Args:
            message: The message to send to the agent
            agent: Optional agent name to route to
            skill: Optional skill name for routing
            continue_conversation: If True, maintains conversation history across calls
            metadata: Optional metadata to include
            configuration: Optional configuration for the A2A request
            ctx: MCP context (injected automatically)

        Returns:
            Dictionary with agent response including text and events
        """
        # Handle conversation continuity
        conversation_key = f"conversation_{agent or 'default'}"
        message_to_send = message

        if continue_conversation:
            history = await ctx.get_state(conversation_key) or []
            if history:
                await ctx.debug(f"Continuing conversation with {len(history)} previous turns")
                # Add conversation context to metadata
                if metadata is None:
                    metadata = {}
                metadata["conversation_history"] = history[-10:]  # Last 10 turns

        response = await _route_message(
            message_to_send,
            ctx=ctx,
            agent=agent,
            skill=skill,
            metadata=metadata,
            configuration=configuration,
        )

        # Update conversation history if enabled
        if continue_conversation:
            history = await ctx.get_state(conversation_key) or []
            history.append({"role": "user", "content": message})
            # ToolResult stores content as a string or content blocks
            response_text = response.content if isinstance(response.content, str) else str(response.content)
            history.append({"role": "agent", "content": response_text})
            # Keep last 20 turns (10 exchanges)
            await ctx.set_state(conversation_key, history[-20:])
            await ctx.debug(f"Updated conversation history: {len(history)} total turns")

        return response

    @mcp.tool
    async def a2a_stream(
        message: str,
        agent: str | None = None,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
        ctx: Context = CurrentContext(),
    ):
        """Stream a response from an A2A agent as text.

        Progress and status updates are sent through context logging and progress reporting.
        Only actual agent text responses are yielded as streaming content.
        """

        await ctx.info("Starting A2A streaming response.")
        index = 0
        total_chunks = 0

        async for chunk in bridge.stream(
            message,
            ctx=ctx,
            agent=agent,
            skill=skill,
            metadata=metadata,
            configuration=configuration,
        ):
            total_chunks += 1

            # Progress and status go to side-channel
            if chunk.kind in {"status", "task"}:
                if chunk.text:
                    await ctx.info(f"[{chunk.kind.upper()}] {chunk.text}")
                await ctx.report_progress(index, index + 1, chunk.kind)

            # Only yield actual text content that should be part of the response
            if chunk.kind in {"message", "artifact"} and chunk.text:
                yield chunk.text

            index += 1

        await ctx.report_progress(total_chunks, total_chunks, "A2A streaming completed")

    @mcp.tool
    def a2a_list_agents() -> list[dict[str, Any]]:
        """List known A2A agents."""

        return [agent.to_dict() for agent in active_router.list_agents()]

    _mark_tool("a2a_send")
    _mark_tool("a2a_stream")
    _mark_tool("a2a_list_agents")

    def _register_agent_tool(descriptor: AgentDescriptor) -> None:
        agent_name = descriptor.name
        description = descriptor.description or f"Send a message to the '{agent_name}' agent."

        async def agent_tool(
            message: str,
            metadata: dict[str, Any] | None = None,
            configuration: dict[str, Any] | None = None,
            ctx: Context = CurrentContext(),
        ) -> ToolResult:
            response_task = asyncio.create_task(
                _route_message(
                    message,
                    ctx=ctx,
                    agent=agent_name,
                    metadata=metadata,
                    configuration=configuration,
                )
            )
            discovery_task = asyncio.create_task(_discover_agent_components(agent_name, ctx))
            response = await response_task
            new_tools, new_prompts = await discovery_task

            # Send discovery notifications through proper channel - don't pollute response text
            if new_tools:
                await ctx.info(f"Discovered new tools: {', '.join(new_tools)}")
            if new_prompts:
                await ctx.info(f"Discovered new prompts: {', '.join(new_prompts)}")

            # Return clean agent response - NO text pollution
            return response

        agent_tool.__name__ = agent_name
        agent_tool.__doc__ = description
        if agent_name in registered_tool_names:
            raise RuntimeError(
                f"Cannot register agent tool '{agent_name}': tool name already in use."
            )
        mcp.add_tool(agent_tool)
        registered_tool_names.add(agent_name)

    for agent_descriptor in active_router.list_agents():
        _register_agent_tool(agent_descriptor)

    @mcp.resource("a2a://agents")
    async def a2a_agents_resource(ctx: Context = CurrentContext()) -> str:
        """Provide agent catalog data to MCP clients."""

        return json.dumps({
            "agents": [agent.to_dict() for agent in active_router.list_agents()],
            "session_id": getattr(ctx, "session_id", None),
        })

    @mcp.resource("a2a://agents/{agent}")
    async def a2a_agent_resource(agent: str, ctx: Context = CurrentContext()) -> str:
        descriptor = active_router.describe(agent)
        return json.dumps({
            "agent": descriptor.to_dict(),
            "session_id": getattr(ctx, "session_id", None),
        })

    @mcp.resource("a2a://agents/{agent}/card")
    async def a2a_agent_card_resource(agent: str) -> str:
        card = await bridge.get_agent_card(agent)
        if card is None:
            return json.dumps({"agent": agent, "card": None})
        if hasattr(card, "model_dump"):
            return json.dumps({"agent": agent, "card": card.model_dump()})
        if hasattr(card, "dict"):
            return json.dumps({"agent": agent, "card": card.dict()})
        return json.dumps({"agent": agent, "card": card})

    @mcp.prompt
    async def a2a_routing_prompt(goal: str, ctx: Context = CurrentContext()) -> str:
        agent_lines = []
        for agent in active_router.list_agents():
            skills = ", ".join(agent.skills) if agent.skills else "(no skills listed)"
            agent_lines.append(f"- {agent.name}: {skills}")
        agent_block = "\n".join(agent_lines) if agent_lines else "- No agents registered"
        session = getattr(ctx, "session_id", None)
        return (
            "You are an MCP client that can route work to A2A agents.\n"
            f"Goal: {goal}\n\n"
            "Available agents:\n"
            f"{agent_block}\n\n"
            "Pick the best agent and call the tool named after that agent with the goal as the message.\n"
            "Use `a2a_send` only if you need manual routing or debugging.\n"
            f"Session: {session}\n"
        )

    _mark_prompt("a2a_routing_prompt")

    return mcp
