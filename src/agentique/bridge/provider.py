"""FastMCP 3.0 Provider that dynamically sources MCP components from agents.

This provider bridges the FastMCP server to any ``AgentAdapter`` backend.
It fetches agent cards, creates tools/resources/prompts, and handles
the full lifecycle via FastMCP's provider protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastmcp.prompts.function_prompt import FunctionPrompt
from fastmcp.resources import Resource
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool
from fastmcp.tools.function_tool import FunctionTool

from agentique.core.types import AgentInfo, BridgeContext

logger = logging.getLogger(__name__)


class _CardCache:
    """Cached agent card with parsed component definitions."""

    __slots__ = ("card", "tools", "prompts", "resources", "fetched_at")

    def __init__(
        self,
        card: Any,
        tools: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
        resources: list[dict[str, Any]],
    ) -> None:
        self.card = card
        self.tools = tools
        self.prompts = prompts
        self.resources = resources
        self.fetched_at = time.monotonic()


class AgentProvider(Provider):
    """Provider that sources MCP components from agent adapters.

    Features:
        - Lazy agent card fetching with TTL-based caching
        - Dynamic tool creation from agent card MCP extensions
        - One base tool per agent for direct access
        - Proper lifecycle management via ``lifespan()``
    """

    def __init__(
        self,
        agents: list[AgentInfo],
        adapter: Any,  # AgentAdapter protocol
        *,
        card_parser: Any | None = None,
        cache_ttl: float = 300.0,
        prefetch_cards: bool = True,
        tool_mapper: Any | None = None,  # optional ToolMapper protocol
    ) -> None:
        super().__init__()
        self._agents = {a.name: a for a in agents}
        self._adapter = adapter
        self._cache_ttl = cache_ttl
        self._prefetch = prefetch_cards
        self._cache: dict[str, _CardCache] = {}
        self._lock = asyncio.Lock()
        self._tool_mapper = tool_mapper  # None → use _make_agent_tool default

        if card_parser is None:
            from agentique.adapters.a2a.card_parser import A2ACardParser
            card_parser = A2ACardParser()
        self._parser = card_parser

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        if self._prefetch:
            await self._prefetch_all()
        try:
            yield
        finally:
            await self._adapter.close()

    async def _prefetch_all(self) -> None:
        tasks = [self._fetch_card(name) for name in self._agents]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_card(self, agent_name: str) -> _CardCache | None:
        async with self._lock:
            cached = self._cache.get(agent_name)
            if cached and (time.monotonic() - cached.fetched_at) < self._cache_ttl:
                return cached

        try:
            card = await self._adapter.get_agent_card(agent_name)
            if card is None:
                return None
            tools, prompts, resources = self._parser.extract_components(card)
            entry = _CardCache(card, tools, prompts, resources)
            async with self._lock:
                self._cache[agent_name] = entry
            return entry
        except Exception:
            logger.exception("Failed to fetch card for agent '%s'", agent_name)
            return None

    # ---- Provider interface ----

    async def _list_tools(self) -> Sequence[Tool]:
        tools: list[Tool] = []
        for name, info in self._agents.items():
            if self._tool_mapper is not None:
                # ToolMapper controls the base tool name/description/structure
                mapper_defs = self._tool_mapper.map_tools(info)
                for tdef in mapper_defs:
                    tool = self._make_tool_from_mapper_def(name, tdef)
                    if tool:
                        tools.append(tool)
            else:
                # Default: one tool per agent (named after the agent)
                tools.append(self._make_agent_tool(info))

            # Card-derived proxy tools are always included alongside base tools
            cached = await self._fetch_card(name)
            if cached:
                for tdef in cached.tools:
                    tool = self._make_proxy_tool(name, tdef)
                    if tool:
                        tools.append(tool)
        return tools

    async def _list_resources(self) -> Sequence[Resource]:
        resources: list[Resource] = [self._make_catalog_resource()]
        for name, info in self._agents.items():
            resources.append(self._make_agent_resource(info))
        return resources

    async def _list_prompts(self) -> Sequence[FunctionPrompt]:
        prompts: list[FunctionPrompt] = [self._make_routing_prompt()]

        for name in self._agents:
            cached = await self._fetch_card(name)
            if cached:
                for pdef in cached.prompts:
                    prompt = self._make_prompt_from_def(name, pdef)
                    if prompt:
                        prompts.append(prompt)
        return prompts

    # ---- Tool builders ----

    def _make_agent_tool(self, info: AgentInfo) -> Tool:
        agent_name = info.name
        desc = info.description or f"Send a message to the '{agent_name}' agent."
        adapter = self._adapter

        async def handler(
            message: str,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            from fastmcp.server.dependencies import get_context
            ctx = get_context()
            bridge_ctx = BridgeContext.from_fastmcp_context(ctx)
            response = await adapter.send_message(agent_name, message, bridge_ctx)
            return {
                "agent": agent_name,
                "text": response.text,
                "events": [e.to_dict() for e in response.events],
            }

        handler.__name__ = agent_name
        handler.__doc__ = desc
        return FunctionTool.from_function(handler, name=agent_name, description=desc)

    def _make_tool_from_mapper_def(
        self, agent_name: str, tool_def: dict[str, Any],
    ) -> Tool | None:
        """Create a FunctionTool from a ToolMapper-produced definition.

        Like ``_make_agent_tool`` but driven by a mapper dict so users can
        control the tool name, description, and multi-tool-per-agent layouts
        (e.g. one tool per skill via ``PerSkillToolMapper``).

        All mapper tools accept a ``message: str`` parameter and delegate
        to ``adapter.send_message(agent_name, message, ctx)``.
        """
        tool_name = _read(tool_def, "name", "tool_name", "id")
        if not tool_name:
            return None
        desc = _read(tool_def, "description", "summary", "title")
        adapter = self._adapter

        async def handler(message: str) -> dict[str, Any]:
            from fastmcp.server.dependencies import get_context
            ctx = get_context()
            bridge_ctx = BridgeContext.from_fastmcp_context(ctx)
            response = await adapter.send_message(agent_name, message, bridge_ctx)
            return {
                "agent": agent_name,
                "text": response.text,
                "events": [e.to_dict() for e in response.events],
            }

        handler.__name__ = tool_name
        handler.__doc__ = desc or f"Send a message via '{agent_name}' (tool: {tool_name})."
        return FunctionTool.from_function(
            handler, name=tool_name, description=desc
        )

    def _make_proxy_tool(
        self, agent_name: str, tool_def: dict[str, Any],
    ) -> Tool | None:
        tool_name = _read(tool_def, "name", "tool_name", "toolName", "id")
        if not tool_name:
            return None
        desc = _read(tool_def, "description", "summary", "title")
        registered = f"{agent_name}_{tool_name}"
        schema = _read(tool_def, "input_schema", "inputSchema", "schema", "parameters")
        arguments = _read(tool_def, "arguments", "args")
        adapter = self._adapter

        async def proxy(**kwargs: Any) -> dict[str, Any]:
            from fastmcp.server.dependencies import get_context
            ctx = get_context()
            bridge_ctx = BridgeContext.from_fastmcp_context(ctx)
            payload = {"tool": tool_name, "arguments": kwargs}
            response = await adapter.send_message(
                agent_name, json.dumps(payload, default=str), bridge_ctx,
            )
            return {
                "agent": agent_name, "tool": tool_name,
                "text": response.text,
                "events": [e.to_dict() for e in response.events],
            }

        proxy.__name__ = registered
        proxy.__doc__ = desc or f"Proxy to '{agent_name}' tool '{tool_name}'."

        params = _build_params(schema, arguments)
        if params:
            proxy.__signature__ = inspect.Signature(parameters=params)
            proxy.__annotations__ = {
                p.name: p.annotation
                for p in params if p.annotation is not inspect.Parameter.empty
            }
            proxy.__annotations__["return"] = dict[str, Any]

        return FunctionTool.from_function(proxy, name=registered, description=desc)

    # ---- Resource builders ----

    def _make_catalog_resource(self) -> Resource:
        agents = self._agents

        async def read() -> str:
            return json.dumps({"agents": [a.to_dict() for a in agents.values()]})

        return Resource.from_function(
            read, uri="a2a://agents", name="A2A Agent Catalog",
            description="List of all registered A2A agents",
            mime_type="application/json",
        )

    def _make_agent_resource(self, info: AgentInfo) -> Resource:
        async def read() -> str:
            return json.dumps({"agent": info.to_dict()})

        return Resource.from_function(
            read, uri=f"a2a://agents/{info.name}",
            name=f"Agent: {info.name}",
            description=info.description or f"Details for agent '{info.name}'",
            mime_type="application/json",
        )

    # ---- Prompt builders ----

    def _make_routing_prompt(self) -> FunctionPrompt:
        agents = self._agents

        def routing(goal: str) -> str:
            lines = []
            for a in agents.values():
                skills = ", ".join(a.skills) if a.skills else "(none)"
                lines.append(f"- {a.name}: {skills}")
            block = "\n".join(lines) or "- No agents registered"
            return (
                "You are an MCP client routing work to A2A agents.\n"
                f"Goal: {goal}\n\nAvailable agents:\n{block}\n\n"
                "Pick the best agent and call it with the goal as message.\n"
            )

        routing.__name__ = "a2a_routing_prompt"
        routing.__doc__ = "Generate a routing prompt for A2A agent selection."
        return FunctionPrompt.from_function(
            routing, name="a2a_routing_prompt",
            description="Generate a routing prompt for A2A agent selection.",
        )

    def _make_prompt_from_def(
        self, agent_name: str, prompt_def: dict[str, Any],
    ) -> FunctionPrompt | None:
        name = _read(prompt_def, "name", "prompt_name", "promptName", "id")
        if not name:
            return None
        desc = _read(prompt_def, "description", "summary", "title")
        registered = f"{agent_name}_{name}"
        template = _read(prompt_def, "template", "prompt", "text", "content")

        def render(**kwargs: Any) -> str:
            if isinstance(template, str):
                try:
                    return template.format(**kwargs)
                except Exception:
                    return template
            return f"Prompt '{name}' for agent '{agent_name}'."

        render.__name__ = registered
        render.__doc__ = desc or f"Prompt '{name}' from agent '{agent_name}'."

        schema = _read(prompt_def, "input_schema", "inputSchema", "schema", "parameters")
        arguments = _read(prompt_def, "arguments", "args")
        params = _build_params(schema, arguments)
        if params:
            render.__signature__ = inspect.Signature(parameters=params)

        return FunctionPrompt.from_function(render, name=registered, description=desc)

    # ---- Public accessors ----

    def list_agents(self) -> list[AgentInfo]:
        return list(self._agents.values())

    def get_agent(self, name: str) -> AgentInfo | None:
        return self._agents.get(name)

    async def get_agent_card(self, name: str) -> Any | None:
        cached = await self._fetch_card(name)
        return cached.card if cached else None

    @property
    def adapter(self) -> Any:
        return self._adapter


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _read(payload: Any, *names: str) -> Any:
    for name in names:
        if isinstance(payload, dict) and name in payload:
            return payload[name]
        val = getattr(payload, name, None)
        if val is not None:
            return val
    return None


def _json_type(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return Any
    t = schema.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        base = _json_type({"type": non_null[0]} if non_null else {})
        return base | None
    mapping = {
        "string": str, "integer": int, "number": float,
        "boolean": bool, "array": list[Any], "object": dict[str, Any],
    }
    return mapping.get(t, Any)


def _build_params(
    schema: Any, arguments: Any,
) -> list[inspect.Parameter]:
    params = _params_from_args(arguments)
    if not params:
        params = _params_from_schema(schema)
    return params


def _params_from_schema(schema: Any) -> list[inspect.Parameter]:
    if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
        return []
    properties = schema.get("properties") or {}
    if not properties:
        return []
    required = set(schema.get("required") or [])
    params: list[inspect.Parameter] = []
    for name, prop in properties.items():
        annotation = _json_type(prop)
        default = prop.get("default", inspect.Parameter.empty if name in required else None)
        params.append(inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default, annotation=annotation,
        ))
    return params


def _params_from_args(arguments: Any) -> list[inspect.Parameter]:
    if not isinstance(arguments, list):
        return []
    params: list[inspect.Parameter] = []
    for arg in arguments:
        if not isinstance(arg, dict):
            continue
        name = _read(arg, "name", "arg", "id")
        if not name:
            continue
        schema = _read(arg, "schema", "input_schema", "inputSchema")
        if schema is None:
            arg_type = _read(arg, "type")
            schema = {"type": arg_type} if arg_type else {}
        annotation = _json_type(schema or {})
        req = bool(_read(arg, "required"))
        default = arg.get("default", inspect.Parameter.empty if req else None)
        params.append(inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default, annotation=annotation,
        ))
    return params
