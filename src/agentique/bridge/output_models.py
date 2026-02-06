"""Structured tool output models using Pydantic.

Defines Pydantic models that map to MCP's ``outputSchema`` and
``structuredContent`` for type-safe tool responses. These models
can be used with FastMCP's ``output_schema`` parameter and
``ToolResult.structured_content`` for validated responses.

Usage::

    from agentique.bridge.output_models import (
        AgentListOutput,
        AgentMessageOutput,
        TaskStatusOutput,
    )
    from fastmcp.tools.tool import ToolResult

    # In a tool function:
    output = AgentMessageOutput(
        agent="my-agent",
        text="Hello!",
        task_id="abc-123",
        state="completed",
    )
    return ToolResult(
        content=output.model_dump_json(),
        structured_content=output.model_dump(),
    )
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent message / response output
# ---------------------------------------------------------------------------


class AgentMessageOutput(BaseModel):
    """Structured output for the ``agent`` tool response."""

    agent: str = Field(description="Name of the agent that responded")
    text: str = Field(description="The agent's response text")
    task_id: str | None = Field(None, description="Task ID for this interaction")
    context_id: str | None = Field(None, description="Context ID for conversation continuity")
    state: str = Field("completed", description="Final task state")
    event_count: int = Field(0, description="Number of events received")
    has_artifacts: bool = Field(False, description="Whether the response includes artifacts")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for MCP outputSchema."""
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# Agent list output
# ---------------------------------------------------------------------------


class AgentSummary(BaseModel):
    """Summary of a single agent."""

    name: str = Field(description="Agent name")
    base_url: str = Field(description="Agent endpoint URL")
    description: str | None = Field(None, description="Agent description")
    skills: list[str] = Field(default_factory=list, description="Agent skills")


class AgentListOutput(BaseModel):
    """Structured output for the ``agents`` tool response."""

    agents: list[AgentSummary] = Field(
        default_factory=list,
        description="List of available agents",
    )
    count: int = Field(0, description="Total number of agents")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for MCP outputSchema."""
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# Task status output
# ---------------------------------------------------------------------------


class TaskStatusOutput(BaseModel):
    """Structured output for the ``task`` tool response."""

    task_id: str = Field(description="Task identifier")
    context_id: str | None = Field(None, description="Associated context ID")
    state: str = Field(description="Current task state")
    progress: float = Field(0.0, description="Progress percentage (0-100)")
    message: str | None = Field(None, description="Status message")
    event_count: int = Field(0, description="Number of events received")
    artifact_count: int = Field(0, description="Number of artifacts")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for MCP outputSchema."""
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# Agent inspection output
# ---------------------------------------------------------------------------


class SubAgentSummary(BaseModel):
    """Summary of a sub-agent in a hierarchy."""

    name: str = Field(description="Sub-agent name")
    description: str | None = Field(None, description="Sub-agent description")
    skills: list[str] = Field(default_factory=list, description="Sub-agent skills")
    parent: str | None = Field(None, description="Parent agent name")
    depth: int = Field(0, description="Depth in hierarchy")


class AgentInspectOutput(BaseModel):
    """Structured output for the ``inspect`` tool response."""

    root: str = Field(description="Root agent name")
    agents: dict[str, SubAgentSummary] = Field(
        default_factory=dict,
        description="Sub-agents in the hierarchy",
    )

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for MCP outputSchema."""
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# Health check output
# ---------------------------------------------------------------------------


class AgentHealthOutput(BaseModel):
    """Structured output for agent health checks."""

    agent_id: str = Field(description="Agent identifier")
    healthy: bool = Field(description="Whether the agent is healthy")
    last_check: float = Field(0.0, description="Unix timestamp of last check")
    consecutive_failures: int = Field(0, description="Number of consecutive failures")
    last_error: str | None = Field(None, description="Last error message")
    latency_ms: float | None = Field(None, description="Last check latency in ms")


class HealthCheckOutput(BaseModel):
    """Structured output for health check results."""

    agents: dict[str, AgentHealthOutput] = Field(
        default_factory=dict,
        description="Health status per agent",
    )
    healthy_count: int = Field(0, description="Number of healthy agents")
    unhealthy_count: int = Field(0, description="Number of unhealthy agents")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for MCP outputSchema."""
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# Webhook notification output
# ---------------------------------------------------------------------------


class WebhookNotificationOutput(BaseModel):
    """Structured output for webhook notification receipt."""

    status: str = Field("received", description="Processing status")
    notification_id: str = Field(description="Unique notification ID")
    agent_id: str = Field(description="Source agent")
    task_id: str | None = Field(None, description="Related task ID")
    kind: str = Field("status", description="Notification kind")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for MCP outputSchema."""
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# Error output
# ---------------------------------------------------------------------------


class ErrorOutput(BaseModel):
    """Structured error response."""

    error: str = Field(description="Error message")
    code: str | None = Field(None, description="Error code")
    details: dict[str, Any] | None = Field(None, description="Additional error details")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for MCP outputSchema."""
        return cls.model_json_schema()
