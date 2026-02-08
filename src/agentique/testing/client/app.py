"""Starlette ASGI app — serves UI, handles WebSocket, relays to MCP client."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from .mcp_client import MCPTestClient
from .models import SessionEvent
from .recorder import SessionRecorder
from .scenario import ScenarioRunner, discover_scenarios

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class TestHarness:
    """Manages the MCP client and WebSocket connections."""

    def __init__(self, mcp_url: str) -> None:
        self.mcp_url = mcp_url
        self.recorder = SessionRecorder()
        self.client: MCPTestClient | None = None
        self.scenario_runner: ScenarioRunner | None = None
        self._ws_connections: set[WebSocket] = set()

    async def start(self) -> None:
        """Initialize the MCP client connection."""
        self.client = MCPTestClient(
            self.mcp_url,
            recorder=self.recorder,
            on_event=self._broadcast_event,
        )
        try:
            await self.client.connect()
            self.scenario_runner = ScenarioRunner(self.client, self.recorder)
        except Exception:
            logger.exception("Failed to connect to MCP server at %s", self.mcp_url)
            self.client = None

    async def stop(self) -> None:
        """Disconnect the MCP client."""
        if self.client:
            await self.client.disconnect()
            self.client = None

    def add_ws(self, ws: WebSocket) -> None:
        self._ws_connections.add(ws)

    def remove_ws(self, ws: WebSocket) -> None:
        self._ws_connections.discard(ws)

    async def _broadcast_event(self, event: SessionEvent) -> None:
        """Broadcast an event to all connected WebSocket clients."""
        msg = {
            "type": event.type.value,
            "data": event.data,
            "timestamp": event.timestamp,
        }
        payload = json.dumps(msg)
        disconnected = set()
        for ws in self._ws_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        self._ws_connections -= disconnected

    async def _send_ws(self, ws: WebSocket, msg: dict[str, Any]) -> None:
        """Send a message to a single WebSocket client."""
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            pass

    async def handle_ws_message(self, ws: WebSocket, raw: str) -> None:
        """Handle an incoming WebSocket message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_ws(ws, {"type": "error", "data": {"error": "Invalid JSON"}})
            return

        msg_type = msg.get("type", "")
        data = msg.get("data", {})

        if not self.client:
            await self._send_ws(ws, {
                "type": "error",
                "data": {"error": "MCP client not connected"},
            })
            return

        match msg_type:
            case "send_message":
                asyncio.create_task(self._handle_send_message(data))

            case "call_tool":
                asyncio.create_task(self._handle_call_tool(data))

            case "list_tools":
                asyncio.create_task(self._handle_list_tools())

            case "list_scenarios":
                await self._handle_list_scenarios(ws)

            case "run_scenario":
                asyncio.create_task(self._handle_run_scenario(data))

            case "run_all_scenarios":
                asyncio.create_task(self._handle_run_all_scenarios())

            case "elicitation_response":
                response = data.get("response", {"action": "cancel"})
                await self.client.resolve_elicitation(response)

            case _:
                await self._send_ws(ws, {
                    "type": "error",
                    "data": {"error": f"Unknown message type: {msg_type}"},
                })

    async def _handle_send_message(self, data: dict[str, Any]) -> None:
        assert self.client
        message = data.get("message", "")
        target = data.get("target")
        timeout = data.get("timeout", 30)
        await self.client.send_message(message, target=target, timeout=timeout)

    async def _handle_call_tool(self, data: dict[str, Any]) -> None:
        assert self.client
        tool = data.get("tool", "")
        arguments = data.get("arguments")
        timeout = data.get("timeout", 30)
        await self.client.call_tool(tool, arguments, timeout=timeout)

    async def _handle_list_tools(self) -> None:
        assert self.client
        await self.client.list_tools()

    async def _handle_list_scenarios(self, ws: WebSocket) -> None:
        scenarios = list(discover_scenarios().keys())
        await self._send_ws(ws, {
            "type": "scenario_list",
            "data": {"scenarios": scenarios},
        })

    async def _handle_run_scenario(self, data: dict[str, Any]) -> None:
        if not self.scenario_runner:
            return
        name = data.get("name", "")
        self.recorder.clear()
        result = await self.scenario_runner.run_scenario(name)
        # Broadcast result to all WS clients
        msg = {
            "type": "scenario_result",
            "data": result.model_dump(),
        }
        payload = json.dumps(msg, default=str)
        for ws in self._ws_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                pass

    async def _handle_run_all_scenarios(self) -> None:
        if not self.scenario_runner:
            return
        discovered = discover_scenarios()
        from .scenario import load_scenario

        for name, path in discovered.items():
            scenario = load_scenario(path)
            self.recorder.clear()
            result = await self.scenario_runner.run_scenario(scenario)
            msg = {
                "type": "scenario_result",
                "data": result.model_dump(),
            }
            payload = json.dumps(msg, default=str)
            for ws in self._ws_connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    pass


# --- Global harness instance (set during app creation) ---
_harness: TestHarness | None = None


async def homepage(request: Request) -> HTMLResponse:
    """Serve the main HTML page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connections."""
    assert _harness is not None
    await websocket.accept()
    _harness.add_ws(websocket)

    # Send initial connection info
    await websocket.send_text(json.dumps({
        "type": "connected",
        "data": {"mcp_url": _harness.mcp_url},
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            await _harness.handle_ws_message(websocket, raw)
    except WebSocketDisconnect:
        pass
    finally:
        _harness.remove_ws(websocket)


def create_app(mcp_url: str) -> Starlette:
    """Create the Starlette ASGI application.

    Args:
        mcp_url: The MCP server URL to connect to (e.g. http://localhost:8000/mcp).
    """
    global _harness
    harness = TestHarness(mcp_url)
    _harness = harness

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await harness.start()
        yield
        await harness.stop()

    app = Starlette(
        routes=[
            Route("/", homepage),
            WebSocketRoute("/ws", websocket_endpoint),
            Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
        ],
        lifespan=lifespan,
    )
    return app
