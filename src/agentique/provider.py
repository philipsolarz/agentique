"""A2A Agent Provider for FastMCP 3.0.

This module implements the Provider architecture to dynamically source
tools, resources, and prompts from A2A agents. It replaces the manual
tool registration approach with a clean, FastMCP-native pattern.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from fastmcp.server.context import Context
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool
from fastmcp.tools.function_tool import FunctionTool
from fastmcp.resources import Resource
from fastmcp.prompts import Prompt
from fastmcp.prompts.prompt import FunctionPrompt

from .a2a_adapter import A2ABridge, A2AClientFactory, A2ATranslator
from .models import AgentDescriptor, McpContextSnapshot


@dataclass
class AgentCardCache:
    """Cached agent card with parsed component definitions."""

    card: Any
    tool_defs: list[dict[str, Any]] = field(default_factory=list)
    prompt_defs: list[dict[str, Any]] = field(default_factory=list)
    resource_defs: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0


class A2AAgentProvider(Provider):
    """Provider that sources MCP components from A2A agents.

    This provider dynamically discovers and exposes tools, resources, and
    prompts from A2A agents via their agent cards. It implements the FastMCP
    Provider interface for clean integration with the server lifecycle.

    Features:
        - Lazy agent card fetching with caching
        - Dynamic tool creation from agent card definitions
        - Proper lifecycle management via lifespan()
        - Support for multiple A2A agents
        - Automatic tool routing to appropriate agents
    """

    def __init__(
        self,
        agents: list[AgentDescriptor],
        *,
        client_factory: A2AClientFactory | None = None,
        translator: A2ATranslator | None = None,
        cache_ttl: float = 300.0,  # 5 minutes default
        prefetch_cards: bool = True,
    ) -> None:
        super().__init__()
        self._agents = {agent.name: agent for agent in agents}
        self._translator = translator or A2ATranslator()
        self._client_factory = client_factory or A2AClientFactory()
        self._bridge = A2ABridge(self._client_factory, self._translator)
        self._cache: dict[str, AgentCardCache] = {}
        self._cache_ttl = cache_ttl
        self._prefetch_cards = prefetch_cards
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Manage provider lifecycle with optional card prefetching."""
        if self._prefetch_cards:
            # Pre-fetch agent cards concurrently at startup
            await self._prefetch_all_cards()
        try:
            yield
        finally:
            await self._bridge.aclose()

    async def _prefetch_all_cards(self) -> None:
        """Prefetch all agent cards concurrently."""
        tasks = [
            self._fetch_and_cache_card(name)
            for name in self._agents
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_and_cache_card(self, agent_name: str) -> AgentCardCache | None:
        """Fetch and cache an agent card."""
        async with self._lock:
            if agent_name in self._cache:
                import time
                cache = self._cache[agent_name]
                if time.time() - cache.fetched_at < self._cache_ttl:
                    return cache

        try:
            descriptor = self._agents.get(agent_name)
            if not descriptor:
                return None

            card = await self._bridge.get_agent_card(descriptor.base_url)
            if card is None:
                return None

            tool_defs, prompt_defs, resource_defs = self._extract_component_defs(card)

            import time
            cache = AgentCardCache(
                card=card,
                tool_defs=tool_defs,
                prompt_defs=prompt_defs,
                resource_defs=resource_defs,
                fetched_at=time.time(),
            )

            async with self._lock:
                self._cache[agent_name] = cache

            return cache

        except Exception:
            return None

    def _extract_component_defs(
        self, card: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract tool, prompt, and resource definitions from agent card."""
        payload = self._coerce_card_payload(card)

        tools = self._normalize_list(
            payload.get("mcp_tools") or payload.get("mcpTools")
        )
        prompts = self._normalize_list(
            payload.get("mcp_prompts") or payload.get("mcpPrompts")
        )
        resources = self._normalize_list(
            payload.get("mcp_resources") or payload.get("mcpResources")
        )

        # Also check extensions for component definitions
        for extension in self._extract_extensions(card):
            params = self._read_field(extension, "params", "parameters")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = None
            if isinstance(params, dict):
                tools.extend(self._normalize_list(
                    params.get("mcp_tools") or params.get("mcpTools")
                ))
                prompts.extend(self._normalize_list(
                    params.get("mcp_prompts") or params.get("mcpPrompts")
                ))
                resources.extend(self._normalize_list(
                    params.get("mcp_resources") or params.get("mcpResources")
                ))

        return tools, prompts, resources

    def _coerce_card_payload(self, card: Any) -> dict[str, Any]:
        """Convert agent card to dict format."""
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

    def _extract_extensions(self, card: Any) -> list[Any]:
        """Extract extensions from agent card."""
        if card is None:
            return []
        if isinstance(card, dict):
            capabilities = card.get("capabilities") or {}
            return self._normalize_list(
                capabilities.get("extensions") or card.get("extensions")
            )
        capabilities = getattr(card, "capabilities", None)
        if capabilities is not None:
            extensions = getattr(capabilities, "extensions", None)
            if extensions:
                return list(extensions)
        extensions = getattr(card, "extensions", None)
        if extensions:
            return list(extensions)
        return []

    def _normalize_list(self, value: Any) -> list[Any]:
        """Normalize value to list."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _read_field(self, payload: Any, *names: str) -> Any:
        """Read field from payload by trying multiple names."""
        for name in names:
            if isinstance(payload, dict) and name in payload:
                return payload[name]
            if hasattr(payload, name):
                return getattr(payload, name)
        return None

    # =========================================================================
    # Provider Interface Implementation
    # =========================================================================

    async def _list_tools(self) -> Sequence[Tool]:
        """List all tools from all registered A2A agents."""
        tools: list[Tool] = []

        # First, add base agent tools (one tool per agent for direct access)
        for agent_name, descriptor in self._agents.items():
            tool = self._create_agent_tool(descriptor)
            tools.append(tool)

        # Then, add discovered tools from agent cards
        for agent_name in self._agents:
            cache = await self._fetch_and_cache_card(agent_name)
            if cache:
                for tool_def in cache.tool_defs:
                    tool = self._create_tool_from_def(agent_name, tool_def)
                    if tool:
                        tools.append(tool)

        return tools

    async def _list_resources(self) -> Sequence[Resource]:
        """List all resources from A2A agents."""
        resources: list[Resource] = []

        # Add agent catalog resource
        resources.append(self._create_agent_catalog_resource())

        # Add per-agent resources
        for agent_name, descriptor in self._agents.items():
            resources.append(self._create_agent_resource(descriptor))
            resources.append(self._create_agent_card_resource(descriptor))

        return resources

    async def _list_prompts(self) -> Sequence[Prompt]:
        """List all prompts from A2A agents."""
        prompts: list[Prompt] = []

        # Add routing prompt
        prompts.append(self._create_routing_prompt())

        # Add discovered prompts from agent cards
        for agent_name in self._agents:
            cache = await self._fetch_and_cache_card(agent_name)
            if cache:
                for prompt_def in cache.prompt_defs:
                    prompt = self._create_prompt_from_def(agent_name, prompt_def)
                    if prompt:
                        prompts.append(prompt)

        return prompts

    # =========================================================================
    # Tool Creation
    # =========================================================================

    def _create_agent_tool(self, descriptor: AgentDescriptor) -> Tool:
        """Create a tool for direct agent access."""
        agent_name = descriptor.name
        description = descriptor.description or f"Send a message to the '{agent_name}' agent."

        async def agent_tool_fn(
            message: str,
            metadata: dict[str, Any] | None = None,
            configuration: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Send a message to this A2A agent."""
            from fastmcp.server.dependencies import get_context
            ctx = get_context()

            snapshot = McpContextSnapshot.from_context(ctx)
            response = await self._bridge.send_message(
                descriptor.base_url,
                message,
                context=snapshot,
                metadata=metadata,
                configuration=configuration,
            )

            return {
                "agent": agent_name,
                "text": response.text,
                "events": [e.to_dict() for e in response.events],
            }

        agent_tool_fn.__name__ = agent_name
        agent_tool_fn.__doc__ = description

        return FunctionTool.from_function(
            agent_tool_fn,
            name=agent_name,
            description=description,
        )

    def _create_tool_from_def(
        self, agent_name: str, tool_def: dict[str, Any]
    ) -> Tool | None:
        """Create a tool from an agent card tool definition."""
        tool_name = self._read_field(tool_def, "name", "tool_name", "toolName", "id")
        if not tool_name:
            return None

        description = self._read_field(tool_def, "description", "summary", "title")
        registered_name = f"{agent_name}_{tool_name}"

        # Build function signature from schema
        schema = self._read_field(
            tool_def, "input_schema", "inputSchema", "schema", "parameters"
        )
        arguments = self._read_field(tool_def, "arguments", "args")

        async def proxy_tool_fn(**kwargs: Any) -> dict[str, Any]:
            """Proxy tool call to A2A agent."""
            from fastmcp.server.dependencies import get_context
            ctx = get_context()

            descriptor = self._agents[agent_name]
            payload = {"tool": tool_name, "arguments": kwargs}
            message = json.dumps(payload, default=str)

            snapshot = McpContextSnapshot.from_context(ctx)
            response = await self._bridge.send_message(
                descriptor.base_url,
                message,
                context=snapshot,
                metadata={"mcp_tool_call": payload},
            )

            return {
                "agent": agent_name,
                "tool": tool_name,
                "text": response.text,
                "events": [e.to_dict() for e in response.events],
            }

        proxy_tool_fn.__name__ = registered_name
        proxy_tool_fn.__doc__ = description or f"Proxy to agent '{agent_name}' tool '{tool_name}'."

        # Apply signature from schema
        params = self._build_parameters(schema, arguments)
        if params:
            proxy_tool_fn.__signature__ = inspect.Signature(parameters=params)
            annotations: dict[str, Any] = {"return": dict[str, Any]}
            for param in params:
                if param.annotation is not inspect._empty:
                    annotations[param.name] = param.annotation
            proxy_tool_fn.__annotations__ = annotations

        return FunctionTool.from_function(
            proxy_tool_fn,
            name=registered_name,
            description=description,
        )

    def _build_parameters(
        self,
        schema: Any,
        arguments: Any,
    ) -> list[inspect.Parameter]:
        """Build function parameters from schema or arguments."""
        params = self._parameters_from_arguments(arguments)
        if not params:
            params = self._parameters_from_schema(schema)
        return params

    def _json_schema_type(self, schema: Any) -> Any:
        """Convert JSON schema type to Python type."""
        if not isinstance(schema, dict):
            return Any
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            non_null = [item for item in schema_type if item != "null"]
            if not non_null:
                return Any
            base = self._json_schema_type({"type": non_null[0]})
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

    def _parameters_from_schema(self, schema: Any) -> list[inspect.Parameter]:
        """Build parameters from JSON schema."""
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
            annotation = self._json_schema_type(prop)
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

    def _parameters_from_arguments(self, arguments: Any) -> list[inspect.Parameter]:
        """Build parameters from arguments list."""
        if not isinstance(arguments, list):
            return []
        params: list[inspect.Parameter] = []
        for arg in arguments:
            if not isinstance(arg, dict):
                continue
            name = self._read_field(arg, "name", "arg", "id")
            if not name:
                continue
            schema = self._read_field(arg, "schema", "input_schema", "inputSchema")
            if schema is None:
                arg_type = self._read_field(arg, "type")
                if arg_type:
                    schema = {"type": arg_type}
            annotation = self._json_schema_type(schema or {})
            required = bool(self._read_field(arg, "required"))
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

    # =========================================================================
    # Resource Creation
    # =========================================================================

    def _create_agent_catalog_resource(self) -> Resource:
        """Create resource for agent catalog."""
        async def read_catalog() -> str:
            return json.dumps({
                "agents": [
                    agent.to_dict() for agent in self._agents.values()
                ]
            })

        return Resource.from_function(
            read_catalog,
            uri="a2a://agents",
            name="A2A Agent Catalog",
            description="List of all registered A2A agents",
            mime_type="application/json",
        )

    def _create_agent_resource(self, descriptor: AgentDescriptor) -> Resource:
        """Create resource for individual agent."""
        async def read_agent() -> str:
            return json.dumps({"agent": descriptor.to_dict()})

        return Resource.from_function(
            read_agent,
            uri=f"a2a://agents/{descriptor.name}",
            name=f"Agent: {descriptor.name}",
            description=descriptor.description or f"Details for agent '{descriptor.name}'",
            mime_type="application/json",
        )

    def _create_agent_card_resource(self, descriptor: AgentDescriptor) -> Resource:
        """Create resource for agent card."""
        async def read_card() -> str:
            cache = await self._fetch_and_cache_card(descriptor.name)
            if cache is None:
                return json.dumps({"agent": descriptor.name, "card": None})

            card = cache.card
            if hasattr(card, "model_dump"):
                return json.dumps({"agent": descriptor.name, "card": card.model_dump()})
            if hasattr(card, "dict"):
                return json.dumps({"agent": descriptor.name, "card": card.dict()})
            return json.dumps({"agent": descriptor.name, "card": card})

        return Resource.from_function(
            read_card,
            uri=f"a2a://agents/{descriptor.name}/card",
            name=f"Agent Card: {descriptor.name}",
            description=f"A2A agent card for '{descriptor.name}'",
            mime_type="application/json",
        )

    # =========================================================================
    # Prompt Creation
    # =========================================================================

    def _create_routing_prompt(self) -> Prompt:
        """Create the routing prompt."""
        def routing_prompt(goal: str) -> str:
            agent_lines = []
            for agent in self._agents.values():
                skills = ", ".join(agent.skills) if agent.skills else "(no skills listed)"
                agent_lines.append(f"- {agent.name}: {skills}")
            agent_block = "\n".join(agent_lines) if agent_lines else "- No agents registered"
            return (
                "You are an MCP client that can route work to A2A agents.\n"
                f"Goal: {goal}\n\n"
                "Available agents:\n"
                f"{agent_block}\n\n"
                "Pick the best agent and call the tool named after that agent with the goal as the message.\n"
            )

        routing_prompt.__name__ = "a2a_routing_prompt"
        routing_prompt.__doc__ = "Generate a routing prompt for A2A agent selection."

        return FunctionPrompt.from_function(
            routing_prompt,
            name="a2a_routing_prompt",
            description="Generate a routing prompt for A2A agent selection.",
        )

    def _create_prompt_from_def(
        self, agent_name: str, prompt_def: dict[str, Any]
    ) -> Prompt | None:
        """Create a prompt from an agent card prompt definition."""
        prompt_name = self._read_field(prompt_def, "name", "prompt_name", "promptName", "id")
        if not prompt_name:
            return None

        description = self._read_field(prompt_def, "description", "summary", "title")
        registered_name = f"{agent_name}_{prompt_name}"
        template = self._read_field(prompt_def, "template", "prompt", "text", "content")

        def render_prompt(**kwargs: Any) -> str:
            if isinstance(template, str):
                try:
                    return template.format(**kwargs)
                except Exception:
                    return template
            return f"Prompt '{prompt_name}' for agent '{agent_name}'."

        render_prompt.__name__ = registered_name
        render_prompt.__doc__ = description or f"Prompt '{prompt_name}' from agent '{agent_name}'."

        # Build signature from schema
        schema = self._read_field(
            prompt_def, "input_schema", "inputSchema", "schema", "parameters"
        )
        arguments = self._read_field(prompt_def, "arguments", "args")
        params = self._build_parameters(schema, arguments)
        if params:
            render_prompt.__signature__ = inspect.Signature(parameters=params)

        return FunctionPrompt.from_function(
            render_prompt,
            name=registered_name,
            description=description,
        )

    # =========================================================================
    # Public Methods
    # =========================================================================

    def list_agents(self) -> list[AgentDescriptor]:
        """List all registered agents."""
        return list(self._agents.values())

    def get_agent(self, name: str) -> AgentDescriptor | None:
        """Get agent descriptor by name."""
        return self._agents.get(name)

    async def get_agent_card(self, name: str) -> Any | None:
        """Get cached agent card."""
        cache = await self._fetch_and_cache_card(name)
        return cache.card if cache else None

    @property
    def bridge(self) -> A2ABridge:
        """Access the underlying A2A bridge."""
        return self._bridge
