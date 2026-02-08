"""MCP test client wrapping fastmcp.Client with event capture hooks."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp import Client

from .models import EventType, SessionEvent
from .recorder import SessionRecorder

logger = logging.getLogger(__name__)

# Type alias for the real-time event callback
OnEventCallback = Callable[[SessionEvent], Awaitable[None]] | None


class MCPTestClient:
    """Wraps fastmcp.Client with event capture and real-time callbacks.

    Provides three callback hooks for real-time streaming visibility:
    - log_handler: captures ctx.info() / ctx.warning() from the server
    - progress_handler: captures ctx.report_progress() calls
    - elicitation_handler: responds to ctx.elicit() calls via asyncio.Future
    """

    def __init__(
        self,
        mcp_url: str,
        recorder: SessionRecorder | None = None,
        on_event: OnEventCallback = None,
    ) -> None:
        self.mcp_url = mcp_url
        self.recorder = recorder or SessionRecorder()
        self.on_event = on_event
        self._client: Client | None = None
        self._elicitation_future: asyncio.Future[dict[str, Any]] | None = None

    async def connect(self) -> None:
        """Connect to the MCP server."""
        self._client = Client(
            self.mcp_url,
            log_handler=self._handle_log,
            progress_handler=self._handle_progress,
            elicitation_handler=self._handle_elicitation,
        )
        await self._client.__aenter__()
        await self._emit(SessionEvent(type=EventType.CONNECTED, data={"url": self.mcp_url}))

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None
            await self._emit(SessionEvent(type=EventType.DISCONNECTED))

    async def __aenter__(self) -> MCPTestClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools."""
        assert self._client, "Not connected"
        tools = await self._client.list_tools()
        tool_list = [
            {"name": t.name, "description": getattr(t, "description", "")}
            for t in tools
        ]
        await self._emit(SessionEvent(
            type=EventType.TOOLS_LISTED,
            data={"tools": [t["name"] for t in tool_list]},
        ))
        return tool_list

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Call an MCP tool and capture all events.

        Returns a dict with keys: text, is_error, structured_content, meta.
        """
        assert self._client, "Not connected"

        await self._emit(SessionEvent(
            type=EventType.TOOL_CALL,
            data={"tool": name, "arguments": arguments or {}},
        ))

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._client.call_tool(name, arguments, raise_on_error=False),
                timeout=timeout,
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            # Extract text from content blocks
            text_parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            text = "\n".join(text_parts)

            result_data = {
                "tool": name,
                "text": text,
                "is_error": result.is_error,
                "structured_content": result.structured_content,
                "meta": result.meta,
                "response_time_ms": elapsed_ms,
            }

            await self._emit(SessionEvent(
                type=EventType.TOOL_RESULT,
                data=result_data,
            ))
            return result_data

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            error_data = {
                "tool": name,
                "text": f"Timeout after {timeout}s",
                "is_error": True,
                "response_time_ms": elapsed_ms,
            }
            await self._emit(SessionEvent(
                type=EventType.ERROR,
                data={"error": f"Tool call '{name}' timed out after {timeout}s"},
            ))
            return error_data

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            error_data = {
                "tool": name,
                "text": str(exc),
                "is_error": True,
                "response_time_ms": elapsed_ms,
            }
            await self._emit(SessionEvent(
                type=EventType.ERROR,
                data={"error": str(exc)},
            ))
            return error_data

    async def send_message(
        self,
        message: str,
        target: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a message via the 'agent' tool.

        This is the primary way to interact with A2A agents through the bridge.
        """
        args: dict[str, Any] = {"message": message}
        if target:
            args["agent"] = target
        return await self.call_tool("agent", args, timeout=timeout)

    async def resolve_elicitation(self, response: dict[str, Any]) -> None:
        """Resolve a pending elicitation request from the UI or scenario runner."""
        if self._elicitation_future and not self._elicitation_future.done():
            self._elicitation_future.set_result(response)
            await self._emit(SessionEvent(
                type=EventType.ELICITATION_RESPONSE,
                data={"response": response},
            ))

    async def _handle_log(self, message: Any) -> None:
        """Handle log notifications from the MCP server."""
        level = getattr(message, "level", "info")
        data = getattr(message, "data", str(message))
        logger.debug("MCP log [%s]: %s", level, data)
        await self._emit(SessionEvent(
            type=EventType.LOG_MESSAGE,
            data={"level": str(level), "data": data},
        ))

    async def _handle_progress(
        self,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        """Handle progress notifications from the MCP server."""
        logger.debug("MCP progress: %s/%s %s", progress, total, message)
        await self._emit(SessionEvent(
            type=EventType.PROGRESS,
            data={"progress": progress, "total": total, "message": message},
        ))

    async def _handle_elicitation(
        self,
        message: str,
        response_type: type | None,
        params: Any,
        context: Any,
    ) -> dict[str, Any]:
        """Handle elicitation requests from the MCP server.

        Creates a Future that WebSocket or scenario runner can resolve.
        """
        logger.info("MCP elicitation request: %s", message)
        await self._emit(SessionEvent(
            type=EventType.ELICITATION_REQUEST,
            data={"message": message},
        ))

        # Create a future for the response
        loop = asyncio.get_running_loop()
        self._elicitation_future = loop.create_future()

        try:
            # Wait up to 120s for a response
            response = await asyncio.wait_for(self._elicitation_future, timeout=120.0)
            return response
        except asyncio.TimeoutError:
            return {"action": "cancel"}
        finally:
            self._elicitation_future = None

    async def _emit(self, event: SessionEvent) -> None:
        """Record event and forward to callback."""
        self.recorder.record(event)
        if self.on_event:
            try:
                await self.on_event(event)
            except Exception:
                logger.exception("Error in on_event callback")
