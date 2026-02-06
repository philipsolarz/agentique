"""Generic HTTP agent adapter.

Connects to agents that expose a simple JSON-over-HTTP API.
Expected agent API contract:

    POST /message
        Body: {"message": str, "metadata": dict}
        Response: {"text": str, "metadata": dict}

    POST /stream  (optional)
        Body: {"message": str, "metadata": dict}
        Response: SSE stream of {"text": str, "done": bool}

    GET /health   (optional)
        Response: {"status": "ok"}

This adapter validates the ``AgentAdapter`` protocol abstraction by
providing a second backend alongside the A2A adapter.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from agentique.core.errors import (
    AdapterError,
    AgentNotFoundError,
    AgentUnavailableError,
)
from agentique.core.types import (
    AgentEvent,
    AgentInfo,
    AgentResponse,
    BridgeContext,
)

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError as exc:
    raise RuntimeError(
        "httpx is required for the HTTP adapter. "
        "Install with: pip install httpx"
    ) from exc


class HttpAgentAdapter:
    """Adapter for agents exposing a simple HTTP JSON API.

    Implements the ``AgentAdapter`` protocol via structural subtyping.

    Args:
        agents: Mapping of agent name to ``AgentInfo``.
        timeout: HTTP request timeout in seconds.
        message_path: URL path for sending messages.
        stream_path: URL path for streaming messages.
        health_path: URL path for health checks.
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo],
        *,
        timeout: float = 60.0,
        message_path: str = "/message",
        stream_path: str = "/stream",
        health_path: str = "/health",
    ) -> None:
        self._agents = dict(agents)
        self._timeout = timeout
        self._message_path = message_path
        self._stream_path = stream_path
        self._health_path = health_path
        self._clients: dict[str, httpx.AsyncClient] = {}

    async def _get_client(self, base_url: str) -> httpx.AsyncClient:
        if base_url not in self._clients:
            self._clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._clients[base_url]

    # ---- AgentAdapter protocol ----

    async def discover_agents(self) -> list[AgentInfo]:
        """Return known agents, optionally filtered by health check."""
        healthy: list[AgentInfo] = []
        for info in self._agents.values():
            try:
                client = await self._get_client(info.base_url)
                resp = await client.get(self._health_path)
                if resp.status_code == 200:
                    healthy.append(info)
                else:
                    logger.warning(
                        "Agent '%s' health check returned %d",
                        info.name, resp.status_code,
                    )
                    healthy.append(info)  # Include anyway
            except Exception:
                logger.warning(
                    "Agent '%s' health check failed", info.name, exc_info=True,
                )
                healthy.append(info)  # Include anyway, let send_message fail
        return healthy

    async def send_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AgentResponse:
        info = self._require_agent(agent_id)
        client = await self._get_client(info.base_url)

        payload = {
            "message": message,
            "metadata": context.to_metadata(),
        }
        if context.conversation_history:
            payload["conversation_history"] = context.conversation_history

        try:
            resp = await client.post(self._message_path, json=payload)
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise AgentUnavailableError(
                f"Cannot connect to agent '{agent_id}' at {info.base_url}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise AdapterError(
                f"Agent '{agent_id}' returned HTTP {exc.response.status_code}"
            ) from exc

        data = resp.json()
        text = data.get("text", "")
        event = AgentEvent(
            kind="message",
            text=text,
            raw=data,
            task_id=data.get("task_id"),
            context_id=data.get("context_id"),
        )
        return AgentResponse(agent=agent_id, text=text, events=(event,))

    async def stream_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AsyncIterator[AgentEvent]:
        info = self._require_agent(agent_id)
        client = await self._get_client(info.base_url)

        payload = {
            "message": message,
            "metadata": context.to_metadata(),
        }
        if context.conversation_history:
            payload["conversation_history"] = context.conversation_history

        try:
            async with client.stream(
                "POST",
                self._stream_path,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    # Process complete SSE events
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        event = self._parse_sse_event(event_str)
                        if event:
                            yield event

                # Process remaining buffer
                if buffer.strip():
                    event = self._parse_sse_event(buffer)
                    if event:
                        yield event

        except httpx.ConnectError as exc:
            raise AgentUnavailableError(
                f"Cannot connect to agent '{agent_id}' at {info.base_url}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            # Streaming not supported — fall back to non-streaming
            if exc.response.status_code == 404:
                response = await self.send_message(agent_id, message, context)
                for event in response.events:
                    yield event
                return
            raise AdapterError(
                f"Agent '{agent_id}' returned HTTP {exc.response.status_code}"
            ) from exc

    async def close(self) -> None:
        """Close all HTTP clients."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    # ---- internal helpers ----

    def _require_agent(self, agent_id: str) -> AgentInfo:
        info = self._agents.get(agent_id)
        if info is None:
            raise AgentNotFoundError(
                f"Unknown agent '{agent_id}'. "
                f"Available: {sorted(self._agents)}"
            )
        return info

    def _parse_sse_event(self, event_str: str) -> AgentEvent | None:
        """Parse a Server-Sent Events formatted string."""
        data_lines: list[str] = []
        for line in event_str.strip().split("\n"):
            if line.startswith("data: "):
                data_lines.append(line[6:])
            elif line.startswith("data:"):
                data_lines.append(line[5:])

        if not data_lines:
            # Try parsing as raw JSON
            try:
                data = json.loads(event_str.strip())
            except (json.JSONDecodeError, ValueError):
                return None
        else:
            raw = "\n".join(data_lines)
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # Plain text event
                return AgentEvent(kind="message", text=raw)

        if isinstance(data, dict):
            text = data.get("text") or data.get("content") or data.get("message")
            is_final = data.get("done", False) or data.get("is_final", False)
            return AgentEvent(
                kind="message",
                text=str(text) if text else None,
                raw=data,
                task_id=data.get("task_id"),
                state=data.get("state"),
                is_final=is_final,
            )

        return AgentEvent(kind="message", text=str(data))
