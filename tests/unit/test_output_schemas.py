"""Tests for output model schemas and output_schema wiring on server tools.

Covers:
  - Each output model has a valid JSON Schema via json_schema()
  - AgentMessageOutput has the correct fields matching agent_tool output
  - TaskListOutput / TaskSummary behave correctly
  - create_server() tools carry output_schema metadata
"""

from __future__ import annotations

import json

import pytest

from agentique.bridge.output_models import (
    AgentInspectOutput,
    AgentListOutput,
    AgentMessageOutput,
    AgentSummary,
    ErrorOutput,
    SubAgentSummary,
    TaskListOutput,
    TaskStatusOutput,
    TaskSummary,
)
from agentique.core.types import AgentInfo


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def is_valid_json_schema(schema: dict) -> bool:
    """Basic check that a dict looks like a JSON Schema."""
    return isinstance(schema, dict) and "type" in schema or "properties" in schema


# ---------------------------------------------------------------------------
# AgentMessageOutput
# ---------------------------------------------------------------------------


def test_agent_message_output_schema_is_dict():
    schema = AgentMessageOutput.json_schema()
    assert isinstance(schema, dict)


def test_agent_message_output_schema_has_properties():
    schema = AgentMessageOutput.json_schema()
    assert "properties" in schema


def test_agent_message_output_has_response_field():
    schema = AgentMessageOutput.json_schema()
    assert "response" in schema["properties"]


def test_agent_message_output_has_artifact_uris_field():
    schema = AgentMessageOutput.json_schema()
    assert "artifact_uris" in schema["properties"]


def test_agent_message_output_has_mcp_related_task_field():
    schema = AgentMessageOutput.json_schema()
    assert "mcp_related_task" in schema["properties"]


def test_agent_message_output_no_text_field():
    """Verify old 'text' field was removed in favour of 'response'."""
    schema = AgentMessageOutput.json_schema()
    assert "text" not in schema.get("properties", {})


def test_agent_message_output_model_dump_has_all_keys():
    obj = AgentMessageOutput(
        agent="test-agent",
        task_id="t1",
        context_id="c1",
        state="completed",
        response="Hello",
        artifact_uris=["a2a://t1/artifacts/a1"],
        event_count=3,
        has_artifacts=True,
        mcp_related_task="t1",
    )
    d = obj.model_dump()
    assert d["agent"] == "test-agent"
    assert d["response"] == "Hello"
    assert d["artifact_uris"] == ["a2a://t1/artifacts/a1"]
    assert d["mcp_related_task"] == "t1"
    assert d["event_count"] == 3
    assert d["has_artifacts"] is True


def test_agent_message_output_defaults():
    obj = AgentMessageOutput(agent="bot")
    d = obj.model_dump()
    assert d["response"] == ""
    assert d["artifact_uris"] == []
    assert d["has_artifacts"] is False
    assert d["mcp_related_task"] is None
    assert d["task_id"] is None


# ---------------------------------------------------------------------------
# AgentListOutput
# ---------------------------------------------------------------------------


def test_agent_list_output_schema_valid():
    schema = AgentListOutput.json_schema()
    assert "properties" in schema
    assert "agents" in schema["properties"]
    assert "count" in schema["properties"]


def test_agent_list_output_count_matches_agents():
    obj = AgentListOutput(
        agents=[AgentSummary(name="a1", base_url="http://a1"), AgentSummary(name="a2", base_url="http://a2")],
        count=2,
    )
    assert obj.count == 2
    assert len(obj.agents) == 2


def test_agent_list_output_empty():
    obj = AgentListOutput()
    assert obj.count == 0
    assert obj.agents == []


# ---------------------------------------------------------------------------
# TaskStatusOutput
# ---------------------------------------------------------------------------


def test_task_status_output_schema_valid():
    schema = TaskStatusOutput.json_schema()
    assert "properties" in schema
    for field in ("task_id", "state", "progress", "event_count", "artifact_count"):
        assert field in schema["properties"]


def test_task_status_output_model_dump():
    obj = TaskStatusOutput(
        task_id="t1",
        context_id="c1",
        state="completed",
        progress=100.0,
        message="Done",
        event_count=5,
        artifact_count=2,
    )
    d = obj.model_dump()
    assert d["task_id"] == "t1"
    assert d["state"] == "completed"
    assert d["event_count"] == 5
    assert d["artifact_count"] == 2


# ---------------------------------------------------------------------------
# TaskSummary / TaskListOutput
# ---------------------------------------------------------------------------


def test_task_summary_fields():
    ts = TaskSummary(task_id="t1", state="working", event_count=2, artifact_count=0)
    assert ts.task_id == "t1"
    assert ts.state == "working"
    assert ts.context_id is None


def test_task_summary_with_context_id():
    ts = TaskSummary(task_id="t1", context_id="c1", state="completed")
    assert ts.context_id == "c1"


def test_task_list_output_schema_valid():
    schema = TaskListOutput.json_schema()
    assert "properties" in schema
    assert "tasks" in schema["properties"]
    assert "count" in schema["properties"]


def test_task_list_output_empty():
    obj = TaskListOutput()
    assert obj.count == 0
    assert obj.tasks == []
    dumped = json.loads(obj.model_dump_json())
    assert dumped["count"] == 0


def test_task_list_output_with_tasks():
    obj = TaskListOutput(
        tasks=[
            TaskSummary(task_id="t1", state="completed"),
            TaskSummary(task_id="t2", state="working"),
        ],
        count=2,
    )
    assert obj.count == 2
    assert obj.tasks[0].task_id == "t1"


def test_task_list_output_model_dump_json_round_trip():
    obj = TaskListOutput(
        tasks=[TaskSummary(task_id="t1", state="completed", event_count=1)],
        count=1,
    )
    json_str = obj.model_dump_json(indent=2)
    parsed = json.loads(json_str)
    assert parsed["count"] == 1
    assert parsed["tasks"][0]["task_id"] == "t1"


# ---------------------------------------------------------------------------
# ErrorOutput
# ---------------------------------------------------------------------------


def test_error_output_schema_valid():
    schema = ErrorOutput.json_schema()
    assert "properties" in schema
    assert "error" in schema["properties"]
    assert "code" in schema["properties"]


def test_error_output_model_dump():
    obj = ErrorOutput(error="Task not found", code="TASK_NOT_FOUND")
    d = obj.model_dump()
    assert d["error"] == "Task not found"
    assert d["code"] == "TASK_NOT_FOUND"
    assert d["details"] is None


# ---------------------------------------------------------------------------
# Server tool output_schema wiring
# ---------------------------------------------------------------------------

pytestmark_server = pytest.mark.anyio


def _make_server():
    from agentique.server import create_server
    return create_server(agents=[AgentInfo(name="test", base_url="http://test.example.com")])


@pytest.mark.anyio
async def test_create_server_agent_tool_has_output_schema():
    mcp = _make_server()
    tool = await mcp._get_tool("agent")
    assert tool is not None
    assert tool.output_schema is not None


@pytest.mark.anyio
async def test_create_server_agents_tool_has_output_schema():
    mcp = _make_server()
    tool = await mcp._get_tool("agents")
    assert tool is not None
    assert tool.output_schema is not None


@pytest.mark.anyio
async def test_create_server_task_tool_has_output_schema():
    mcp = _make_server()
    tool = await mcp._get_tool("task")
    assert tool is not None
    assert tool.output_schema is not None


@pytest.mark.anyio
async def test_create_server_inspect_tool_has_output_schema():
    mcp = _make_server()
    tool = await mcp._get_tool("inspect")
    assert tool is not None
    assert tool.output_schema is not None


@pytest.mark.anyio
async def test_agent_tool_output_schema_matches_agent_message_output():
    mcp = _make_server()
    tool = await mcp._get_tool("agent")
    expected = AgentMessageOutput.json_schema()
    assert tool.output_schema == expected


@pytest.mark.anyio
async def test_agents_tool_output_schema_matches_agent_list_output():
    mcp = _make_server()
    tool = await mcp._get_tool("agents")
    expected = AgentListOutput.json_schema()
    assert tool.output_schema == expected


@pytest.mark.anyio
async def test_task_tool_output_schema_matches_task_status_output():
    mcp = _make_server()
    tool = await mcp._get_tool("task")
    expected = TaskStatusOutput.json_schema()
    assert tool.output_schema == expected


@pytest.mark.anyio
async def test_inspect_tool_output_schema_matches_agent_inspect_output():
    mcp = _make_server()
    tool = await mcp._get_tool("inspect")
    expected = AgentInspectOutput.json_schema()
    assert tool.output_schema == expected


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_task_list_output_exported_from_agentique():
    import agentique
    assert hasattr(agentique, "TaskListOutput")
    assert hasattr(agentique, "TaskSummary")
    assert hasattr(agentique, "SubAgentSummary")


def test_agent_message_output_exported_from_agentique():
    import agentique
    assert hasattr(agentique, "AgentMessageOutput")
    # Verify updated schema is reflected in export
    schema = agentique.AgentMessageOutput.json_schema()
    assert "response" in schema.get("properties", {})
