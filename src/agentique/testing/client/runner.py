"""Headless runner for pytest — no browser needed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .mcp_client import MCPTestClient
from .models import ScenarioResult
from .recorder import SessionRecorder
from .scenario import ScenarioRunner, discover_scenarios

logger = logging.getLogger(__name__)


class HeadlessRunner:
    """Async context manager for running scenarios without a browser.

    Usage::

        async with HeadlessRunner("http://localhost:8000/mcp") as runner:
            result = await runner.run_scenario("03_math")
            assert result.passed
    """

    def __init__(self, mcp_url: str) -> None:
        self.mcp_url = mcp_url
        self.recorder = SessionRecorder()
        self.client = MCPTestClient(mcp_url, recorder=self.recorder)
        self.scenario_runner = ScenarioRunner(self.client, self.recorder)

    async def __aenter__(self) -> HeadlessRunner:
        await self.client.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.client.disconnect()

    async def run_scenario(
        self,
        name_or_path: str | Path,
    ) -> ScenarioResult:
        """Run a scenario by name or file path."""
        self.recorder.clear()
        return await self.scenario_runner.run_scenario(name_or_path)

    async def run_all_scenarios(
        self,
        directory: str | Path | None = None,
        tags: list[str] | None = None,
    ) -> list[ScenarioResult]:
        """Run all discovered scenarios, optionally filtered by tags."""
        from .scenario import load_scenario

        discovered = discover_scenarios(directory)
        results: list[ScenarioResult] = []

        for name, path in discovered.items():
            scenario = load_scenario(path)
            if tags:
                if not any(t in scenario.tags for t in tags):
                    continue
            self.recorder.clear()
            result = await self.scenario_runner.run_scenario(scenario)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            logger.info("%s: %s (%.0fms)", status, name, result.total_time_ms)

        return results

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools."""
        return await self.client.list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Call a tool directly."""
        return await self.client.call_tool(name, arguments, timeout=timeout)

    async def send_message(
        self,
        message: str,
        target: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a message to an agent."""
        return await self.client.send_message(message, target=target, timeout=timeout)
