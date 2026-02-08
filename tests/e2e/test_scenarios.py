"""E2E tests driven by YAML scenarios via the HeadlessRunner.

These tests require a running Docker Compose stack:
    docker compose -f docker-compose.interactive.yml up -d --wait

Run with:
    uv run pytest tests/e2e/test_scenarios.py -v
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agentique.testing.client import HeadlessRunner, discover_scenarios

MCP_URL = "http://localhost:8000/mcp"


@pytest_asyncio.fixture(scope="module")
async def runner():
    """Create a HeadlessRunner connected to the live MCP server."""
    async with HeadlessRunner(MCP_URL) as r:
        yield r


@pytest.mark.e2e
async def test_tool_discovery(runner: HeadlessRunner):
    """Verify core MCP tools are discoverable."""
    tools = await runner.list_tools()
    tool_names = {t["name"] for t in tools}
    assert "agent" in tool_names, f"Expected 'agent' tool, got: {tool_names}"
    assert "agents" in tool_names, f"Expected 'agents' tool, got: {tool_names}"


@pytest.mark.e2e
async def test_list_agents(runner: HeadlessRunner):
    """Verify the agents tool returns agent data."""
    result = await runner.call_tool("agents")
    assert not result.get("is_error"), f"agents tool returned error: {result.get('text')}"
    assert result.get("text"), "agents tool returned empty response"


@pytest.mark.e2e
async def test_scenario_discovery(runner: HeadlessRunner):
    """Verify built-in YAML scenarios are discoverable."""
    scenarios = discover_scenarios()
    assert len(scenarios) >= 6, f"Expected at least 6 scenarios, found: {list(scenarios.keys())}"
    assert "01_discovery" in scenarios
    assert "03_math" in scenarios


@pytest.mark.e2e
@pytest.mark.parametrize(
    "scenario_name",
    sorted(discover_scenarios().keys()),
)
async def test_scenario(runner: HeadlessRunner, scenario_name: str):
    """Run each YAML scenario and assert it passes."""
    result = await runner.run_scenario(scenario_name)
    failures = [
        f"  {sr.step_name}: {sr.error or '; '.join(ar.message for ar in sr.assertion_results if not ar.passed)}"
        for sr in result.step_results
        if not sr.passed
    ]
    assert result.passed, (
        f"Scenario '{scenario_name}' failed ({result.total_time_ms:.0f}ms):\n"
        + "\n".join(failures)
    )
