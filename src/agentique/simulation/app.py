"""SimulationHarness + Starlette ASGI app for the simulation observer UI."""

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
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from agentique.testing.client.mcp_client import MCPTestClient
from agentique.testing.client.recorder import SessionRecorder

from .models import (
    EventType,
    SessionEvent,
    SimulationConfig,
    SimulationObjective,
    SimulationPersona,
    SimulationState,
)
from .simulation_agent import SimulationAgent

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class SimulationHarness:
    """Manages simulation lifecycle, WebSocket broadcasting, and REST API state."""

    def __init__(self, mcp_url: str) -> None:
        self.mcp_url = mcp_url
        self._simulations: dict[str, SimulationAgent] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._ws_connections: set[WebSocket] = set()

    async def start_simulation(self, config: SimulationConfig) -> str:
        """Start a new simulation and return its ID."""
        config.mcp_url = self.mcp_url

        recorder = SessionRecorder()
        client = MCPTestClient(
            self.mcp_url,
            recorder=recorder,
            on_event=self._make_mcp_event_forwarder(),
        )

        # Create reasoning client for Cogito mode
        reasoning_client = None
        if config.use_cogito_reasoning:
            reasoning_mcp_url = config.reasoning_mcp_url or self.mcp_url
            reasoning_recorder = SessionRecorder()
            reasoning_client = MCPTestClient(
                reasoning_mcp_url,
                recorder=reasoning_recorder,
                on_event=self._make_mcp_event_forwarder(),
            )

        agent = SimulationAgent(
            client=client,
            config=config,
            on_event=self._make_event_handler(),
            reasoning_client=reasoning_client,
        )

        sim_id = agent.simulation_id
        self._simulations[sim_id] = agent
        self._events[sim_id] = []

        # Run simulation in background task
        task = asyncio.create_task(
            self._run_simulation(sim_id, client, agent, reasoning_client)
        )
        self._tasks[sim_id] = task

        return sim_id

    async def _run_simulation(
        self,
        sim_id: str,
        client: MCPTestClient,
        agent: SimulationAgent,
        reasoning_client: MCPTestClient | None = None,
    ) -> None:
        """Execute a simulation with proper client lifecycle."""
        try:
            await client.connect()
            if reasoning_client:
                await reasoning_client.connect()
            result = await agent.run()
            self._results[sim_id] = result.model_dump()
        except Exception as exc:
            logger.exception("Simulation %s failed", sim_id)
            self._results[sim_id] = {
                "simulation_id": sim_id,
                "state": SimulationState.FAILED.value,
                "summary": f"Simulation failed: {exc}",
                "conversation": [t.model_dump() for t in agent.conversation],
                "insights": [i.model_dump() for i in agent.insights],
            }
        finally:
            if reasoning_client:
                await reasoning_client.disconnect()
            await client.disconnect()
            self._tasks.pop(sim_id, None)

    def stop_simulation(self, sim_id: str) -> bool:
        """Stop a running simulation."""
        agent = self._simulations.get(sim_id)
        if not agent:
            return False
        agent.stop()
        return True

    async def pause_simulation(self, sim_id: str) -> bool:
        """Pause a running simulation."""
        agent = self._simulations.get(sim_id)
        if not agent:
            return False
        await agent.pause()
        return True

    async def resume_simulation(self, sim_id: str) -> bool:
        """Resume a paused simulation."""
        agent = self._simulations.get(sim_id)
        if not agent:
            return False
        await agent.resume()
        return True

    def get_status(self, sim_id: str) -> dict[str, Any] | None:
        """Get simulation status."""
        agent = self._simulations.get(sim_id)
        if not agent:
            result = self._results.get(sim_id)
            if result:
                return {
                    "simulation_id": sim_id,
                    "state": result.get("state", SimulationState.COMPLETED.value),
                    "turns": len(result.get("conversation", [])) // 2,
                    "insights_count": len(result.get("insights", [])),
                }
            return None

        return {
            "simulation_id": sim_id,
            "state": agent.state.value,
            "turns": len(agent.conversation) // 2,
            "insights_count": len(agent.insights),
            "persona": agent.config.persona.name,
            "objective": agent.config.objective.goal,
        }

    def get_events(
        self, sim_id: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get events for a simulation with pagination."""
        events = self._events.get(sim_id, [])
        return events[offset : offset + limit]

    def get_result(self, sim_id: str) -> dict[str, Any] | None:
        """Get simulation result (after completion)."""
        return self._results.get(sim_id)

    def get_insights(self, sim_id: str) -> list[dict[str, Any]]:
        """Get insights from a simulation."""
        agent = self._simulations.get(sim_id)
        if agent:
            return [i.model_dump() for i in agent.insights]
        result = self._results.get(sim_id)
        if result:
            return result.get("insights", [])
        return []

    def list_simulations(self) -> list[dict[str, Any]]:
        """List all simulations."""
        sims = []
        # Active simulations
        for sim_id, agent in self._simulations.items():
            sims.append({
                "simulation_id": sim_id,
                "state": agent.state.value,
                "persona": agent.config.persona.name,
                "objective": agent.config.objective.goal,
                "turns": len(agent.conversation) // 2,
            })
        # Completed simulations not in active
        for sim_id, result in self._results.items():
            if sim_id not in self._simulations:
                sims.append({
                    "simulation_id": sim_id,
                    "state": result.get("state", SimulationState.COMPLETED.value),
                    "turns": len(result.get("conversation", [])) // 2,
                })
        return sims

    def _make_event_handler(self):
        """Create an event handler that records and broadcasts simulation events."""

        async def handler(event: SessionEvent) -> None:
            event_data = {
                "type": event.type.value,
                "data": event.data,
                "timestamp": event.timestamp,
                "simulation_id": event.simulation_id,
            }
            sim_id = event.simulation_id
            if sim_id and sim_id in self._events:
                self._events[sim_id].append(event_data)
            await self._broadcast(event_data)

        return handler

    def _make_mcp_event_forwarder(self):
        """Forward MCP protocol events (tool calls, logs, etc.) to WebSocket."""
        from agentique.testing.client.models import SessionEvent as TestingEvent

        async def forwarder(event: TestingEvent) -> None:
            event_data = {
                "type": event.type.value,
                "data": event.data,
                "timestamp": event.timestamp,
            }
            await self._broadcast(event_data)

        return forwarder

    async def _broadcast(self, msg: dict[str, Any]) -> None:
        """Send to all WebSocket clients."""
        payload = json.dumps(msg, default=str)
        disconnected = set()
        for ws in self._ws_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        self._ws_connections -= disconnected

    def add_ws(self, ws: WebSocket) -> None:
        self._ws_connections.add(ws)

    def remove_ws(self, ws: WebSocket) -> None:
        self._ws_connections.discard(ws)


# --- Global harness (set during app creation) ---
_harness: SimulationHarness | None = None


# --- HTTP Handlers ---

async def homepage(request: Request) -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


async def api_start_simulation(request: Request) -> JSONResponse:
    assert _harness is not None
    body = await request.json()

    persona_data = body.get("persona", {})
    objective_data = body.get("objective", {})

    config = SimulationConfig(
        persona=SimulationPersona(
            name=persona_data.get("name", "Alex"),
            role=persona_data.get("role", "curious user"),
            personality_traits=persona_data.get("personality_traits", ["friendly", "inquisitive"]),
            conversation_style=persona_data.get("conversation_style", "casual and direct"),
            typing_speed_ms=persona_data.get("typing_speed_ms", 80.0),
            reading_speed_ms=persona_data.get("reading_speed_ms", 20.0),
        ),
        objective=SimulationObjective(
            goal=objective_data.get("goal", "Explore the AI agent's capabilities"),
            max_turns=objective_data.get("max_turns", 10),
            max_duration_seconds=objective_data.get("max_duration_seconds", 300.0),
            topics=objective_data.get("topics", []),
            stop_conditions=objective_data.get("stop_conditions", []),
        ),
        llm_provider=body.get("llm_provider", "gemini"),
        llm_model=body.get("llm_model", "gemini-2.0-flash"),
    )

    sim_id = await _harness.start_simulation(config)
    return JSONResponse({"simulation_id": sim_id, "status": "started"})


async def api_stop_simulation(request: Request) -> JSONResponse:
    assert _harness is not None
    sim_id = request.path_params["sim_id"]
    success = _harness.stop_simulation(sim_id)
    if not success:
        return JSONResponse({"error": "Simulation not found"}, status_code=404)
    return JSONResponse({"simulation_id": sim_id, "status": "stopping"})


async def api_pause_simulation(request: Request) -> JSONResponse:
    assert _harness is not None
    sim_id = request.path_params["sim_id"]
    success = await _harness.pause_simulation(sim_id)
    if not success:
        return JSONResponse({"error": "Simulation not found"}, status_code=404)
    return JSONResponse({"simulation_id": sim_id, "status": "paused"})


async def api_resume_simulation(request: Request) -> JSONResponse:
    assert _harness is not None
    sim_id = request.path_params["sim_id"]
    success = await _harness.resume_simulation(sim_id)
    if not success:
        return JSONResponse({"error": "Simulation not found"}, status_code=404)
    return JSONResponse({"simulation_id": sim_id, "status": "resumed"})


async def api_list_simulations(request: Request) -> JSONResponse:
    assert _harness is not None
    return JSONResponse({"simulations": _harness.list_simulations()})


async def api_get_simulation(request: Request) -> JSONResponse:
    assert _harness is not None
    sim_id = request.path_params["sim_id"]
    status = _harness.get_status(sim_id)
    if not status:
        return JSONResponse({"error": "Simulation not found"}, status_code=404)
    return JSONResponse(status)


async def api_get_events(request: Request) -> JSONResponse:
    assert _harness is not None
    sim_id = request.path_params["sim_id"]
    limit = int(request.query_params.get("limit", "100"))
    offset = int(request.query_params.get("offset", "0"))
    events = _harness.get_events(sim_id, limit=limit, offset=offset)
    return JSONResponse({"events": events, "total": len(_harness._events.get(sim_id, []))})


async def api_get_insights(request: Request) -> JSONResponse:
    assert _harness is not None
    sim_id = request.path_params["sim_id"]
    insights = _harness.get_insights(sim_id)
    return JSONResponse({"insights": insights})


async def api_get_result(request: Request) -> JSONResponse:
    assert _harness is not None
    sim_id = request.path_params["sim_id"]
    result = _harness.get_result(sim_id)
    if not result:
        return JSONResponse({"error": "Result not available yet"}, status_code=404)
    return JSONResponse(result)


async def websocket_endpoint(websocket: WebSocket) -> None:
    assert _harness is not None
    await websocket.accept()
    _harness.add_ws(websocket)

    await websocket.send_text(json.dumps({
        "type": "connected",
        "data": {"mcp_url": _harness.mcp_url, "mode": "simulation"},
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            # Observer UI - handle minimal controls only
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")
                if msg_type == "pause_simulation":
                    sim_id = msg.get("simulation_id", "")
                    await _harness.pause_simulation(sim_id)
                elif msg_type == "resume_simulation":
                    sim_id = msg.get("simulation_id", "")
                    await _harness.resume_simulation(sim_id)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _harness.remove_ws(websocket)


def create_app(mcp_url: str) -> Starlette:
    """Create the Starlette ASGI application for the simulation observer UI."""
    global _harness
    harness = SimulationHarness(mcp_url)
    _harness = harness

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        logger.info("Simulation UI starting, MCP URL: %s", mcp_url)
        yield
        logger.info("Simulation UI shutting down")

    app = Starlette(
        routes=[
            Route("/", homepage),
            # REST API
            Route("/api/simulations", api_list_simulations, methods=["GET"]),
            Route("/api/simulations", api_start_simulation, methods=["POST"]),
            Route("/api/simulations/{sim_id}", api_get_simulation, methods=["GET"]),
            Route("/api/simulations/{sim_id}", api_stop_simulation, methods=["DELETE"]),
            Route("/api/simulations/{sim_id}/pause", api_pause_simulation, methods=["POST"]),
            Route("/api/simulations/{sim_id}/resume", api_resume_simulation, methods=["POST"]),
            Route("/api/simulations/{sim_id}/events", api_get_events, methods=["GET"]),
            Route("/api/simulations/{sim_id}/insights", api_get_insights, methods=["GET"]),
            Route("/api/simulations/{sim_id}/result", api_get_result, methods=["GET"]),
            # WebSocket
            WebSocketRoute("/ws", websocket_endpoint),
            # Static files
            Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
        ],
        lifespan=lifespan,
    )
    return app
