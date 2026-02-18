"""Simulator MCP Server — exposes simulation controls as MCP tools.

Operates in two modes:
1. REST mode: Calls the Simulation UI's REST API (when simulation_ui_url is set)
2. Headless mode: Creates its own internal SimulationHarness (no UI)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def create_simulator_mcp_server(
    simulation_ui_url: str | None = None,
    mcp_url: str = "http://localhost:8000/mcp",
) -> FastMCP:
    """Create an MCP server that controls simulations.

    Args:
        simulation_ui_url: URL of the Simulation UI (e.g., http://localhost:8080).
            If None, runs in headless mode with an internal harness.
        mcp_url: MCP server URL for headless mode.

    Returns:
        FastMCP server with simulation tools
    """
    mcp = FastMCP("Agentique Simulator")

    _http_client = None
    _harness = None

    async def _get_http_client():
        nonlocal _http_client
        if _http_client is None:
            import httpx
            _http_client = httpx.AsyncClient(timeout=120.0)
        return _http_client

    async def _get_harness():
        nonlocal _harness
        if _harness is None:
            from ..app import SimulationHarness
            _harness = SimulationHarness(mcp_url)
        return _harness

    async def _api_call(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a REST API call to the simulation UI."""
        try:
            client = await _get_http_client()
            url = f"{simulation_ui_url}{path}"

            if method == "GET":
                resp = await client.get(url, params=params)
            elif method == "POST":
                resp = await client.post(url, json=body or {})
            elif method == "DELETE":
                resp = await client.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if resp.status_code >= 400:
                return {"error": f"HTTP {resp.status_code}: {resp.text}", "status_code": resp.status_code}

            return resp.json()
        except httpx.ConnectError as exc:
            logger.error("Connection error calling %s %s: %s", method, path, exc)
            return {"error": f"Connection failed: {exc}", "status_code": 0}
        except httpx.TimeoutException as exc:
            logger.error("Timeout calling %s %s: %s", method, path, exc)
            return {"error": f"Request timed out: {exc}", "status_code": 0}

    @mcp.tool()
    async def start_simulation(
        objective: str = "Explore the AI agent's capabilities through natural conversation",
        persona_name: str = "Alex",
        persona_role: str = "curious user",
        conversation_style: str = "casual and direct",
        max_turns: int = 10,
        topics: list[str] | None = None,
        persona_preset: str | None = None,
        use_cogito_reasoning: bool = False,
    ) -> str:
        """Start a simulated human conversation with the AI agent.

        An LLM-powered agent will emulate a real human user, typing messages
        naturally and having a genuine conversation through the full MCP pipeline.

        Args:
            objective: What the simulated human should try to accomplish
            persona_name: Name of the simulated person
            persona_role: Role/background of the simulated person
            conversation_style: How the person talks (e.g., "casual", "formal", "technical")
            max_turns: Maximum conversation turns (each turn = human message + AI response)
            topics: Specific topics to explore during the conversation
            persona_preset: Use a preset persona ("cogito" for full Cogito mirror mode)
            use_cogito_reasoning: Route thinking through MCP/Cogito instead of direct LLM

        Returns:
            JSON with simulation_id and status
        """
        # Apply persona preset
        if persona_preset == "cogito":
            use_cogito_reasoning = True

        config_body = {
            "persona": {
                "name": persona_name,
                "role": persona_role,
                "conversation_style": conversation_style,
            },
            "objective": {
                "goal": objective,
                "max_turns": max_turns,
                "topics": topics or [],
            },
            "use_cogito_reasoning": use_cogito_reasoning,
        }

        if simulation_ui_url:
            result = await _api_call("POST", "/api/simulations", body=config_body)
        else:
            from ..models import SimulationConfig, SimulationObjective, SimulationPersona

            if persona_preset == "cogito":
                persona = SimulationPersona.cogito(persona_name)
            else:
                persona = SimulationPersona(
                    name=persona_name,
                    role=persona_role,
                    conversation_style=conversation_style,
                )

            harness = await _get_harness()
            config = SimulationConfig(
                persona=persona,
                objective=SimulationObjective(
                    goal=objective,
                    max_turns=max_turns,
                    topics=topics or [],
                ),
                use_cogito_reasoning=use_cogito_reasoning,
            )
            sim_id = await harness.start_simulation(config)
            result = {"simulation_id": sim_id, "status": "started"}

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def stop_simulation(simulation_id: str) -> str:
        """Stop a running simulation.

        Args:
            simulation_id: The ID of the simulation to stop

        Returns:
            JSON with status
        """
        if simulation_ui_url:
            result = await _api_call("DELETE", f"/api/simulations/{simulation_id}")
        else:
            harness = await _get_harness()
            success = harness.stop_simulation(simulation_id)
            result = {"simulation_id": simulation_id, "status": "stopping" if success else "not_found"}

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def pause_simulation(simulation_id: str) -> str:
        """Pause a running simulation.

        Pauses after the current turn completes. Use this to make code edits,
        then call resume_simulation to continue.

        Args:
            simulation_id: The ID of the simulation to pause

        Returns:
            JSON with status
        """
        if simulation_ui_url:
            result = await _api_call("POST", f"/api/simulations/{simulation_id}/pause")
        else:
            harness = await _get_harness()
            success = await harness.pause_simulation(simulation_id)
            result = {"simulation_id": simulation_id, "status": "paused" if success else "not_found"}

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def resume_simulation(simulation_id: str) -> str:
        """Resume a paused simulation.

        Call this after making code edits. The simulation continues from where it paused.

        Args:
            simulation_id: The ID of the simulation to resume

        Returns:
            JSON with status
        """
        if simulation_ui_url:
            result = await _api_call("POST", f"/api/simulations/{simulation_id}/resume")
        else:
            harness = await _get_harness()
            success = await harness.resume_simulation(simulation_id)
            result = {"simulation_id": simulation_id, "status": "resumed" if success else "not_found"}

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def get_simulation_status(simulation_id: str) -> str:
        """Get the current status of a simulation.

        Args:
            simulation_id: The ID of the simulation

        Returns:
            JSON with simulation state, turns completed, insights count
        """
        if simulation_ui_url:
            result = await _api_call("GET", f"/api/simulations/{simulation_id}")
        else:
            harness = await _get_harness()
            result = harness.get_status(simulation_id) or {"error": "Simulation not found"}

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def get_simulation_events(
        simulation_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """Get events from a simulation with pagination.

        Shows the detailed event log including typing, messages, responses, and insights.

        Args:
            simulation_id: The ID of the simulation
            limit: Maximum events to return (default: 50)
            offset: Skip this many events (default: 0)

        Returns:
            JSON with events list and total count
        """
        if simulation_ui_url:
            result = await _api_call(
                "GET",
                f"/api/simulations/{simulation_id}/events",
                params={"limit": limit, "offset": offset},
            )
        else:
            harness = await _get_harness()
            events = harness.get_events(simulation_id, limit=limit, offset=offset)
            result = {"events": events, "total": len(harness._events.get(simulation_id, []))}

        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def get_simulation_insights(simulation_id: str) -> str:
        """Get insights discovered during a simulation.

        Insights are categorized by severity and include evidence from the conversation.

        Args:
            simulation_id: The ID of the simulation

        Returns:
            JSON with list of insights
        """
        if simulation_ui_url:
            result = await _api_call("GET", f"/api/simulations/{simulation_id}/insights")
        else:
            harness = await _get_harness()
            insights = harness.get_insights(simulation_id)
            result = {"insights": insights}

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def list_simulations() -> str:
        """List all simulations (active and completed).

        Returns:
            JSON with list of simulation summaries
        """
        if simulation_ui_url:
            result = await _api_call("GET", "/api/simulations")
        else:
            harness = await _get_harness()
            result = {"simulations": harness.list_simulations()}

        return json.dumps(result, indent=2)

    return mcp
