"""MCP proxy adapter implementing the ``AgentAdapter`` protocol.

Wraps remote MCP servers as agent-like adapters, enabling MCP-to-MCP
bridging.  Each remote MCP server is treated as a single "agent" whose
tools, resources, and prompts are accessible through agentique's bridge
layer.

Uses FastMCP 3.0's ``create_proxy()`` internally to manage connections
and proxying.

Usage::

    from agentique.adapters.mcp import MCPProxyAdapter
    from agentique.core.types import AgentInfo

    agents = {
        "remote-tools": AgentInfo(
            name="remote-tools",
            base_url="http://remote-mcp-server:8000/mcp",
            description="Remote MCP tool server",
        ),
    }
    adapter = MCPProxyAdapter(agents)
    response = await adapter.send_message("remote-tools", "call tool X", ctx)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator
from uuid import uuid4

from agentique.core.errors import AgentNotFoundError
from agentique.core.types import (
    AgentEvent,
    AgentInfo,
    AgentResponse,
    BridgeContext,
)

logger = logging.getLogger(__name__)

try:
    from fastmcp.server import create_proxy
    from fastmcp.server.providers.proxy import FastMCPProxy
except ImportError:
    create_proxy = None  # type: ignore[assignment]
    FastMCPProxy = None  # type: ignore[assignment,misc]


class MCPProxyAdapter:
    """Adapter that bridges agentique to remote MCP servers.

    Each registered agent maps to a remote MCP server endpoint.
    The adapter uses FastMCP's ``create_proxy()`` to create proxy
    servers that forward tool calls to the remote backend.

    Implements the ``AgentAdapter`` protocol via structural subtyping.

    Args:
        agents: Mapping of agent name to ``AgentInfo``.  The
            ``base_url`` of each agent should point to the remote
            MCP server's endpoint (HTTP URL, or path to a script).
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo],
    ) -> None:
        if create_proxy is None:
            raise RuntimeError(
                "FastMCP 3.0+ is required for the MCP proxy adapter."
            )
        self._agents = dict(agents)
        self._proxies: dict[str, FastMCPProxy] = {}

    def _get_proxy(self, agent_id: str) -> FastMCPProxy:
        """Get or create a proxy for the given agent."""
        if agent_id in self._proxies:
            return self._proxies[agent_id]

        info = self._agents.get(agent_id)
        if info is None:
            raise AgentNotFoundError(
                f"Unknown agent '{agent_id}'. "
                f"Available: {sorted(self._agents)}"
            )

        proxy = create_proxy(info.base_url, name=f"agentique-proxy-{agent_id}")
        self._proxies[agent_id] = proxy
        return proxy

    # ---- AgentAdapter protocol ----

    async def discover_agents(self) -> list[AgentInfo]:
        """Return the agents reachable through this adapter."""
        return list(self._agents.values())

    async def send_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AgentResponse:
        """Forward a message to the remote MCP server.

        The *message* is interpreted as a tool call request.  The
        adapter calls the first available tool on the remote server
        with the message as input and returns the result.
        """
        proxy = self._get_proxy(agent_id)
        task_id = str(uuid4())

        events: list[AgentEvent] = []
        try:
            # Use the proxy's test client to call tools
            async with proxy.test_client() as client:
                tools = await client.list_tools()

                if not tools:
                    event = AgentEvent(
                        kind="message",
                        text=f"No tools available on remote MCP server for agent '{agent_id}'.",
                        task_id=task_id,
                    )
                    events.append(event)
                else:
                    # Call the first tool with the message as input
                    tool = tools[0]
                    try:
                        result = await client.call_tool(
                            tool.name,
                            arguments={"message": message},
                        )
                        text = _extract_text(result)
                    except Exception:
                        # Retry without arguments if the tool doesn't take 'message'
                        try:
                            result = await client.call_tool(tool.name, arguments={})
                            text = _extract_text(result)
                        except Exception as exc:
                            text = f"Error calling tool '{tool.name}': {exc}"

                    event = AgentEvent(
                        kind="message",
                        text=text,
                        task_id=task_id,
                    )
                    events.append(event)
        except Exception as exc:
            logger.warning(
                "MCP proxy send_message failed for agent %s: %s",
                agent_id, exc,
            )
            event = AgentEvent(
                kind="message",
                text=f"Error communicating with remote MCP server: {exc}",
                task_id=task_id,
            )
            events.append(event)

        all_text = " ".join(e.text for e in events if e.text).strip()
        return AgentResponse(agent=agent_id, text=all_text, events=tuple(events))

    async def stream_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a message to the remote MCP server.

        Since standard MCP tool calls are request/response (not streaming),
        this yields a single event with the full result.
        """
        response = await self.send_message(agent_id, message, context)
        for event in response.events:
            yield event

    async def get_agent_card(self, agent_id: str) -> dict[str, Any] | None:
        """Build a synthetic agent card from the remote server's tools."""
        proxy = self._get_proxy(agent_id)
        info = self._agents[agent_id]

        try:
            async with proxy.test_client() as client:
                tools = await client.list_tools()
                return {
                    "name": info.name,
                    "description": info.description or f"MCP proxy to {info.base_url}",
                    "version": "1.0.0",
                    "url": info.base_url,
                    "capabilities": {
                        "streaming": False,
                        "pushNotifications": False,
                    },
                    "skills": [
                        {
                            "id": t.name,
                            "name": t.name,
                            "description": t.description or f"Tool: {t.name}",
                        }
                        for t in tools
                    ],
                    "metadata": {"protocol": "mcp", "tool_count": len(tools)},
                }
        except Exception:
            logger.debug(
                "Failed to build agent card for MCP proxy %s",
                agent_id, exc_info=True,
            )
            return None

    async def close(self) -> None:
        """Release proxy resources."""
        self._proxies.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(result: Any) -> str:
    """Extract text content from an MCP tool call result."""
    if isinstance(result, str):
        return result

    # Handle CallToolResult or similar
    content = getattr(result, "content", None)
    if content is None:
        return str(result)

    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif hasattr(block, "text"):
                texts.append(str(block.text))
            elif isinstance(block, dict) and "text" in block:
                texts.append(str(block["text"]))
        return " ".join(texts) if texts else str(result)

    return str(content)
