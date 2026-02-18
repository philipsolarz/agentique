"""Scenario runner — loads YAML scenarios and executes steps."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .models import (
    EventType,
    Scenario,
    ScenarioResult,
    SessionEvent,
    StepAction,
    StepResult,
)

if TYPE_CHECKING:
    from .mcp_client import MCPTestClient
    from .recorder import SessionRecorder

logger = logging.getLogger(__name__)

# Default directory for built-in scenario YAML files
SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file."""
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return Scenario.model_validate(data)


def discover_scenarios(directory: str | Path | None = None) -> dict[str, Path]:
    """Discover all YAML scenario files in a directory.

    Returns a mapping of scenario name to file path.
    """
    search_dir = Path(directory) if directory else SCENARIOS_DIR
    scenarios: dict[str, Path] = {}
    if search_dir.exists():
        for p in sorted(search_dir.glob("*.yaml")):
            scenario = load_scenario(p)
            scenarios[scenario.name] = p
    return scenarios


class ScenarioRunner:
    """Executes test scenarios against an MCPTestClient."""

    def __init__(
        self,
        client: MCPTestClient,
        recorder: SessionRecorder,
    ) -> None:
        self.client = client
        self.recorder = recorder
        self.simulate = False  # Enable visual simulation mode

    async def run_scenario(
        self,
        scenario: Scenario | str | Path,
    ) -> ScenarioResult:
        """Run a complete scenario and return results.

        Accepts a Scenario object, a file path, or a scenario name
        (looked up from the built-in scenarios directory).
        """
        if isinstance(scenario, (str, Path)):
            path = Path(scenario)
            if not path.exists():
                # Try looking up by name in scenarios dir
                discovered = discover_scenarios()
                if str(scenario) in discovered:
                    path = discovered[str(scenario)]
                else:
                    return ScenarioResult(
                        scenario_name=str(scenario),
                        passed=False,
                        error=f"Scenario not found: {scenario}",
                    )
            scenario = load_scenario(path)

        logger.info("Running scenario: %s", scenario.name)
        await self.client._emit(SessionEvent(
            type=EventType.SCENARIO_START,
            data={"name": scenario.name, "description": scenario.description},
        ))

        overall_start = time.monotonic()
        step_results: list[StepResult] = []
        all_passed = True

        for step in scenario.steps:
            result = await self._run_step(step)
            step_results.append(result)
            if not result.passed:
                all_passed = False

        total_time = (time.monotonic() - overall_start) * 1000

        scenario_result = ScenarioResult(
            scenario_name=scenario.name,
            passed=all_passed,
            step_results=step_results,
            total_time_ms=total_time,
        )

        await self.client._emit(SessionEvent(
            type=EventType.SCENARIO_END,
            data={
                "name": scenario.name,
                "passed": all_passed,
                "total_time_ms": total_time,
            },
        ))

        return scenario_result

    async def _run_step(self, step: "StepResult | object") -> StepResult:
        """Execute a single scenario step."""
        from .models import ScenarioStep

        assert isinstance(step, ScenarioStep)

        logger.info("Running step: %s (%s)", step.name, step.action)
        self.recorder.mark_step_start(step.name)

        await self.client._emit(SessionEvent(
            type=EventType.STEP_START,
            data={"name": step.name, "action": step.action.value},
        ))

        start = time.monotonic()
        response_text = ""
        error = None

        try:
            match step.action:
                case StepAction.SEND_MESSAGE:
                    # Emit simulation events for visual animation
                    if self.simulate:
                        await self.client._emit(SessionEvent(
                            type=EventType.SIMULATION_EVENT,
                            data={
                                "action": "type_message",
                                "message": step.message or "",
                            },
                        ))
                        # Give UI time to animate typing
                        message = step.message or ""
                        typing_time = len(message) * 0.08 + 0.5  # Approximate typing time
                        await asyncio.sleep(typing_time)

                        await self.client._emit(SessionEvent(
                            type=EventType.SIMULATION_EVENT,
                            data={"action": "show_thinking"},
                        ))

                    result = await self.client.send_message(
                        message=step.message or "",
                        timeout=step.timeout,
                    )
                    response_text = result.get("text", "")

                    if self.simulate:
                        await self.client._emit(SessionEvent(
                            type=EventType.SIMULATION_EVENT,
                            data={"action": "hide_thinking"},
                        ))
                        await self.client._emit(SessionEvent(
                            type=EventType.SIMULATION_EVENT,
                            data={"action": "read_response"},
                        ))
                        # Simulate reading time
                        reading_time = min(3.0, len(response_text) * 0.02)
                        await asyncio.sleep(reading_time)

                case StepAction.CALL_TOOL:
                    result = await self.client.call_tool(
                        name=step.tool or "",
                        arguments=step.arguments,
                        timeout=step.timeout,
                    )
                    response_text = result.get("text", "")

                case StepAction.LIST_TOOLS:
                    tools = await self.client.list_tools()
                    response_text = ", ".join(t["name"] for t in tools)

                case StepAction.WAIT:
                    wait_time = step.wait_seconds or 1.0
                    await asyncio.sleep(wait_time)
                    response_text = f"Waited {wait_time}s"

                case StepAction.ELICIT_RESPONSE:
                    await self.client.resolve_elicitation(
                        step.arguments or {"action": "submit"}
                    )
                    response_text = "Elicitation resolved"

        except Exception as exc:
            error = str(exc)
            logger.exception("Step '%s' failed", step.name)

        elapsed_ms = (time.monotonic() - start) * 1000
        self.recorder.mark_step_end(step.name)

        # Evaluate assertions
        step_events = self.recorder.get_step_events(step.name)
        assertion_results = [
            self.recorder.evaluate_assertion(a, response_text, elapsed_ms, step_events)
            for a in step.assertions
        ]

        passed = error is None and all(ar.passed for ar in assertion_results)

        await self.client._emit(SessionEvent(
            type=EventType.STEP_END,
            data={"name": step.name, "passed": passed},
        ))

        return StepResult(
            step_name=step.name,
            passed=passed,
            response_text=response_text,
            response_time_ms=elapsed_ms,
            assertion_results=assertion_results,
            events=step_events,
            error=error,
        )
