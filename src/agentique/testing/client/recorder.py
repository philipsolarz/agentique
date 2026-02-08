"""Session recorder that captures MCP events and evaluates assertions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .models import (
    AssertionResult,
    AssertionType,
    EventType,
    SessionEvent,
)

if TYPE_CHECKING:
    from .models import Assertion


class SessionRecorder:
    """Captures all MCP events during a test session.

    Supports step-scoped event tracking and assertion evaluation.
    """

    def __init__(self) -> None:
        self.events: list[SessionEvent] = []
        self._current_step: str | None = None
        self._step_boundaries: dict[str, tuple[int, int | None]] = {}

    def record(self, event: SessionEvent) -> None:
        """Record an event, tagging it with the current step if active."""
        if self._current_step:
            event.step_name = self._current_step
        self.events.append(event)

    def mark_step_start(self, step_name: str) -> None:
        """Mark the beginning of a scenario step."""
        self._current_step = step_name
        self._step_boundaries[step_name] = (len(self.events), None)

    def mark_step_end(self, step_name: str) -> None:
        """Mark the end of a scenario step."""
        if step_name in self._step_boundaries:
            start, _ = self._step_boundaries[step_name]
            self._step_boundaries[step_name] = (start, len(self.events))
        self._current_step = None

    def get_step_events(self, step_name: str) -> list[SessionEvent]:
        """Get all events recorded during a specific step."""
        return [e for e in self.events if e.step_name == step_name]

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()
        self._step_boundaries.clear()
        self._current_step = None

    def evaluate_assertion(
        self,
        assertion: Assertion,
        response_text: str,
        response_time_ms: float,
        step_events: list[SessionEvent],
    ) -> AssertionResult:
        """Evaluate a single assertion against the step context."""
        match assertion.type:
            case AssertionType.CONTAINS:
                value = str(assertion.value)
                passed = value.lower() in response_text.lower()
                msg = (
                    f"Response contains '{value}'"
                    if passed
                    else f"Expected '{value}' in response, got: {response_text[:200]}"
                )

            case AssertionType.NOT_CONTAINS:
                value = str(assertion.value)
                passed = value.lower() not in response_text.lower()
                msg = (
                    f"Response does not contain '{value}'"
                    if passed
                    else f"Unexpected '{value}' found in response"
                )

            case AssertionType.MATCHES:
                pattern = str(assertion.value)
                passed = bool(re.search(pattern, response_text, re.IGNORECASE))
                msg = (
                    f"Response matches /{pattern}/"
                    if passed
                    else f"Response does not match /{pattern}/: {response_text[:200]}"
                )

            case AssertionType.TOOL_CALLED:
                tool_name = str(assertion.value)
                called_tools = {
                    e.data.get("tool", "")
                    for e in step_events
                    if e.type == EventType.TOOL_CALL
                }
                passed = tool_name in called_tools
                msg = (
                    f"Tool '{tool_name}' was called"
                    if passed
                    else f"Tool '{tool_name}' was not called (called: {called_tools})"
                )

            case AssertionType.TOOL_NOT_CALLED:
                tool_name = str(assertion.value)
                called_tools = {
                    e.data.get("tool", "")
                    for e in step_events
                    if e.type == EventType.TOOL_CALL
                }
                passed = tool_name not in called_tools
                msg = (
                    f"Tool '{tool_name}' was not called"
                    if passed
                    else f"Tool '{tool_name}' was unexpectedly called"
                )

            case AssertionType.HAS_PROGRESS:
                has_progress = any(
                    e.type == EventType.PROGRESS for e in step_events
                )
                passed = has_progress
                msg = (
                    "Progress events received"
                    if passed
                    else "No progress events received"
                )

            case AssertionType.HAS_LOG:
                has_log = any(
                    e.type == EventType.LOG_MESSAGE for e in step_events
                )
                passed = has_log
                msg = (
                    "Log messages received"
                    if passed
                    else "No log messages received"
                )

            case AssertionType.RESPONSE_TIME_LT:
                threshold = float(assertion.value)  # type: ignore[arg-type]
                passed = response_time_ms < threshold
                msg = (
                    f"Response time {response_time_ms:.0f}ms < {threshold:.0f}ms"
                    if passed
                    else f"Response time {response_time_ms:.0f}ms >= {threshold:.0f}ms threshold"
                )

            case _:
                passed = False
                msg = f"Unknown assertion type: {assertion.type}"

        return AssertionResult(assertion=assertion, passed=passed, message=msg)
