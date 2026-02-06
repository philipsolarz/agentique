"""Tests for structured output models."""

from __future__ import annotations

import json
import pytest

from agentique.bridge.output_models import (
    AgentHealthOutput,
    AgentInspectOutput,
    AgentListOutput,
    AgentMessageOutput,
    AgentSummary,
    ErrorOutput,
    HealthCheckOutput,
    SubAgentSummary,
    TaskStatusOutput,
    WebhookNotificationOutput,
)


class TestAgentMessageOutput:
    def test_basic(self):
        output = AgentMessageOutput(
            agent="test-agent",
            text="Hello world",
            task_id="t1",
            state="completed",
        )
        assert output.agent == "test-agent"
        assert output.text == "Hello world"
        assert output.state == "completed"

    def test_model_dump(self):
        output = AgentMessageOutput(agent="a", text="b")
        data = output.model_dump()
        assert data["agent"] == "a"
        assert data["text"] == "b"
        assert data["state"] == "completed"  # default

    def test_json_schema(self):
        schema = AgentMessageOutput.json_schema()
        assert "properties" in schema
        assert "agent" in schema["properties"]
        assert "text" in schema["properties"]

    def test_model_dump_json(self):
        output = AgentMessageOutput(agent="a", text="b")
        raw = output.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["agent"] == "a"


class TestAgentListOutput:
    def test_with_agents(self):
        output = AgentListOutput(
            agents=[
                AgentSummary(name="a", base_url="http://localhost:8001"),
                AgentSummary(name="b", base_url="http://localhost:8002", description="Agent B"),
            ],
            count=2,
        )
        assert len(output.agents) == 2
        assert output.count == 2
        assert output.agents[1].description == "Agent B"

    def test_empty(self):
        output = AgentListOutput()
        assert output.agents == []
        assert output.count == 0

    def test_json_schema(self):
        schema = AgentListOutput.json_schema()
        assert "properties" in schema


class TestTaskStatusOutput:
    def test_basic(self):
        output = TaskStatusOutput(
            task_id="t1",
            state="working",
            progress=50.0,
            event_count=10,
        )
        assert output.task_id == "t1"
        assert output.state == "working"
        assert output.progress == 50.0

    def test_json_schema(self):
        schema = TaskStatusOutput.json_schema()
        assert "task_id" in schema["properties"]


class TestAgentInspectOutput:
    def test_with_hierarchy(self):
        output = AgentInspectOutput(
            root="main",
            agents={
                "sub-1": SubAgentSummary(name="sub-1", depth=1),
                "sub-2": SubAgentSummary(name="sub-2", parent="sub-1", depth=2),
            },
        )
        assert output.root == "main"
        assert len(output.agents) == 2
        assert output.agents["sub-2"].parent == "sub-1"


class TestHealthCheckOutput:
    def test_basic(self):
        output = HealthCheckOutput(
            agents={
                "a": AgentHealthOutput(agent_id="a", healthy=True),
                "b": AgentHealthOutput(agent_id="b", healthy=False, consecutive_failures=3),
            },
            healthy_count=1,
            unhealthy_count=1,
        )
        assert output.healthy_count == 1
        assert output.agents["b"].consecutive_failures == 3


class TestWebhookNotificationOutput:
    def test_basic(self):
        output = WebhookNotificationOutput(
            notification_id="n1",
            agent_id="agent-a",
            task_id="t1",
            kind="message",
        )
        assert output.notification_id == "n1"
        assert output.agent_id == "agent-a"


class TestErrorOutput:
    def test_basic(self):
        output = ErrorOutput(error="Not found", code="NOT_FOUND")
        assert output.error == "Not found"
        assert output.code == "NOT_FOUND"

    def test_json_schema(self):
        schema = ErrorOutput.json_schema()
        assert "error" in schema["properties"]
