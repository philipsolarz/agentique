"""MCP server interface for the test client.

This exposes the test suite as MCP tools so that other agents (including Claude)
can trigger autonomous testing, run scenarios, and get insights programmatically.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp import FastMCP

from ..autonomous_agent import AutonomousTestAgent, TestObjective
from ..mcp_client import MCPTestClient
from ..recorder import SessionRecorder
from ..scenario import ScenarioRunner, discover_scenarios

logger = logging.getLogger(__name__)


def create_test_mcp_server(mcp_url: str) -> FastMCP:
    """Create an MCP server that exposes the test suite as tools.

    Args:
        mcp_url: The MCP server URL to test (e.g., http://localhost:8000/mcp)

    Returns:
        FastMCP server with test tools
    """
    mcp = FastMCP("Agentique Test Suite")

    # Global state (initialized on first use)
    _client: MCPTestClient | None = None
    _autonomous_agent: AutonomousTestAgent | None = None
    _scenario_runner: ScenarioRunner | None = None
    _recorder = SessionRecorder()
    _last_report: dict[str, Any] | None = None

    async def _ensure_connected() -> MCPTestClient:
        """Ensure test client is connected."""
        nonlocal _client, _autonomous_agent, _scenario_runner

        if _client is None:
            client = MCPTestClient(mcp_url, recorder=_recorder)
            await client.connect()
            _scenario_runner = ScenarioRunner(client, _recorder)
            try:
                _autonomous_agent = AutonomousTestAgent(
                    client,
                    llm_provider="gemini",
                    simulate=False,  # No visual simulation for programmatic use
                )
            except Exception:
                logger.warning(
                    "Autonomous agent failed to initialize (missing GOOGLE_API_KEY?). "
                    "Autonomous testing will be unavailable, but other tools will work."
                )
            _client = client
            logger.info("Test client connected to %s", mcp_url)

        return _client

    @mcp.tool()
    async def run_autonomous_test(
        objective: str = "Comprehensively test all MCP server capabilities",
        focus_areas: list[str] | None = None,
        max_actions: int = 15,
    ) -> str:
        """Run autonomous AI-powered testing on the MCP server.

        The autonomous agent will:
        1. Discover all MCP tools and capabilities
        2. Generate an intelligent test plan using an LLM
        3. Execute test actions and analyze responses
        4. Discover bugs, edge cases, and issues
        5. Return a comprehensive report with insights

        Args:
            objective: High-level testing goal
            focus_areas: Specific areas to focus on (e.g., ["tool calling", "error handling"])
            max_actions: Maximum number of test actions to execute

        Returns:
            JSON string with test report including:
            - actions_executed: Number of test actions run
            - success_rate: Percentage of successful actions
            - insights_discovered: Number of bugs/issues found
            - insights: Detailed list of discovered issues
            - coverage: Tool coverage statistics
        """
        nonlocal _last_report

        await _ensure_connected()

        if _autonomous_agent is None:
            return json.dumps({"error": "Autonomous agent not initialized"})

        test_objective = TestObjective(
            goal=objective,
            focus_areas=focus_areas or [
                "tool calling",
                "agent routing",
                "error handling",
                "streaming responses",
            ],
        )

        logger.info("Running autonomous test with objective: %s", objective)

        report = await _autonomous_agent.autonomous_exploration(
            test_objective,
            max_actions=max_actions,
        )

        _last_report = report

        return json.dumps(report, indent=2, default=str)

    @mcp.tool()
    async def get_last_test_report() -> str:
        """Get the most recent autonomous test report.

        Returns:
            JSON string with the last test report, or error if no tests have been run
        """
        nonlocal _last_report

        if _last_report is None:
            return json.dumps({"error": "No tests have been run yet"})

        return json.dumps(_last_report, indent=2, default=str)

    @mcp.tool()
    async def list_test_scenarios() -> str:
        """List all available test scenarios.

        Returns:
            JSON string with list of scenario names
        """
        scenarios = discover_scenarios()
        return json.dumps({"scenarios": list(scenarios.keys())}, indent=2)

    @mcp.tool()
    async def run_test_scenario(name: str) -> str:
        """Run a specific test scenario.

        Args:
            name: Name of the scenario to run (e.g., "03_math")

        Returns:
            JSON string with scenario results
        """
        await _ensure_connected()

        if _scenario_runner is None:
            return json.dumps({"error": "Scenario runner not initialized"})

        _recorder.clear()
        result = await _scenario_runner.run_scenario(name)

        return json.dumps(result.model_dump(), indent=2, default=str)

    @mcp.tool()
    async def get_test_insights() -> str:
        """Get all insights discovered by the autonomous agent.

        Returns:
            JSON string with list of insights, categorized by severity
        """
        await _ensure_connected()

        if _autonomous_agent is None:
            return json.dumps({"error": "Autonomous agent not initialized"})

        insights_by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": [],
        }

        for insight in _autonomous_agent.insights:
            insights_by_severity[insight.severity].append(insight.model_dump())

        return json.dumps({
            "total_insights": len(_autonomous_agent.insights),
            "by_severity": insights_by_severity,
        }, indent=2)

    @mcp.tool()
    async def list_mcp_tools() -> str:
        """List all tools available on the MCP server being tested.

        Returns:
            JSON string with list of tool names and descriptions
        """
        await _ensure_connected()

        if _client is None:
            return json.dumps({"error": "Client not connected"})

        tools = await _client.list_tools()
        return json.dumps({"tools": tools}, indent=2)

    return mcp


if __name__ == "__main__":
    # For standalone testing
    import asyncio

    async def main():
        server = create_test_mcp_server("http://localhost:8000/mcp")
        server.run(transport="stdio")

    asyncio.run(main())
