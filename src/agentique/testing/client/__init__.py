"""Agentique MCP Test Client — web-based visual chat UI and headless scenario runner.

Usage:
    # Visual mode (browser)
    python -m agentique.testing.client --mcp-url http://localhost:8000/mcp

    # Headless mode (pytest)
    async with HeadlessRunner("http://localhost:8000/mcp") as runner:
        result = await runner.run_scenario("03_math")
"""

from .app import create_app
from .mcp_client import MCPTestClient
from .models import (
    Assertion,
    AssertionType,
    EventType,
    Scenario,
    ScenarioResult,
    ScenarioStep,
    SessionEvent,
    StepAction,
    StepResult,
)
from .recorder import SessionRecorder
from .runner import HeadlessRunner
from .scenario import ScenarioRunner, discover_scenarios, load_scenario

__all__ = [
    "Assertion",
    "AssertionType",
    "EventType",
    "HeadlessRunner",
    "MCPTestClient",
    "Scenario",
    "ScenarioResult",
    "ScenarioRunner",
    "ScenarioStep",
    "SessionEvent",
    "SessionRecorder",
    "StepAction",
    "StepResult",
    "create_app",
    "discover_scenarios",
    "load_scenario",
]
