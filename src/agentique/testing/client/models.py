"""Pydantic models for the MCP test client."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AssertionType(str, Enum):
    """Types of assertions that can be evaluated against step results."""

    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"
    TOOL_CALLED = "tool_called"
    TOOL_NOT_CALLED = "tool_not_called"
    HAS_PROGRESS = "has_progress"
    HAS_LOG = "has_log"
    RESPONSE_TIME_LT = "response_time_lt"


class StepAction(str, Enum):
    """Types of actions a scenario step can perform."""

    SEND_MESSAGE = "send_message"
    CALL_TOOL = "call_tool"
    LIST_TOOLS = "list_tools"
    WAIT = "wait"
    ELICIT_RESPONSE = "elicit_response"


class EventType(str, Enum):
    """Types of events captured during a session."""

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LOG_MESSAGE = "log_message"
    PROGRESS = "progress"
    ELICITATION_REQUEST = "elicitation_request"
    ELICITATION_RESPONSE = "elicitation_response"
    ERROR = "error"
    TOOLS_LISTED = "tools_listed"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SCENARIO_START = "scenario_start"
    SCENARIO_END = "scenario_end"
    STEP_START = "step_start"
    STEP_END = "step_end"


class Assertion(BaseModel):
    """A single assertion to evaluate against step results."""

    type: AssertionType
    value: str | float | None = None


class ScenarioStep(BaseModel):
    """A single step in a test scenario."""

    name: str
    action: StepAction
    message: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    timeout: int = 30
    assertions: list[Assertion] = Field(default_factory=list)
    wait_seconds: float | None = None


class Scenario(BaseModel):
    """A complete test scenario loaded from YAML."""

    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    steps: list[ScenarioStep]


class SessionEvent(BaseModel):
    """An event captured during a test session."""

    type: EventType
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)
    step_name: str | None = None


class AssertionResult(BaseModel):
    """Result of evaluating a single assertion."""

    assertion: Assertion
    passed: bool
    message: str = ""


class StepResult(BaseModel):
    """Result of executing a single scenario step."""

    step_name: str
    passed: bool
    response_text: str = ""
    response_time_ms: float = 0.0
    assertion_results: list[AssertionResult] = Field(default_factory=list)
    events: list[SessionEvent] = Field(default_factory=list)
    error: str | None = None


class ScenarioResult(BaseModel):
    """Result of executing a complete scenario."""

    scenario_name: str
    passed: bool
    step_results: list[StepResult] = Field(default_factory=list)
    total_time_ms: float = 0.0
    error: str | None = None
