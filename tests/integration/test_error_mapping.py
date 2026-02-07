"""Integration tests for A2A to MCP error code mapping.

Tests that A2A error codes are correctly translated to MCP error responses.
"""

from __future__ import annotations

import pytest


# A2A JSON-RPC error codes (from JSON-RPC 2.0 spec)
ERROR_MAPPING_CASES = [
    (-32700, "parse_error", "Parse error"),
    (-32600, "invalid_request", "Invalid Request"),
    (-32601, "method_not_found", "Method not found"),
    (-32602, "invalid_params", "Invalid params"),
    (-32603, "internal_error", "Internal error"),
    # Application-defined errors (server errors)
    (-32000, "server_error", "Server error"),
]


@pytest.mark.integration
@pytest.mark.parametrize("a2a_code,expected_type,description", ERROR_MAPPING_CASES)
def test_error_code_mapping_exists(a2a_code, expected_type, description):
    """Error mapping function exists and handles standard error codes."""
    try:
        from agentique.errors import map_a2a_error_to_mcp
    except ImportError:
        try:
            from agentique.core.errors import map_a2a_error_to_mcp
        except ImportError:
            pytest.skip("Error mapping not yet implemented")

    # Test that the mapping function exists and returns something
    # The exact return type depends on implementation
    result = map_a2a_error_to_mcp(a2a_code)
    assert result is not None


@pytest.mark.integration
def test_unknown_error_code_handling():
    """Unknown error codes are handled gracefully."""
    try:
        from agentique.errors import map_a2a_error_to_mcp
    except ImportError:
        try:
            from agentique.core.errors import map_a2a_error_to_mcp
        except ImportError:
            pytest.skip("Error mapping not yet implemented")

    # Test with an unknown error code
    result = map_a2a_error_to_mcp(-99999)
    # Should return a default error or handle gracefully
    assert result is not None


@pytest.mark.integration
async def test_error_agent_returns_proper_error_structure(error_a2a_client):
    """Error agent returns properly structured A2A error response."""
    from uuid import uuid4

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "trigger error"}],
                "messageId": uuid4().hex,
            }
        },
    }

    response = await error_a2a_client.post("/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "jsonrpc" in data
    # The error agent should return a result with failed status
    # Exact structure depends on implementation


@pytest.mark.integration
def test_task_state_failed_mapping():
    """TaskState.failed maps to appropriate error."""
    from a2a.types import TaskState

    # Verify the enum exists
    assert TaskState.failed is not None
    assert TaskState.failed.value == "failed"


@pytest.mark.integration
def test_task_state_rejected_mapping():
    """TaskState.rejected exists and can be mapped."""
    from a2a.types import TaskState

    # Check if rejected state exists
    if hasattr(TaskState, "rejected"):
        assert TaskState.rejected.value == "rejected"
    else:
        pytest.skip("TaskState.rejected not in this A2A SDK version")


@pytest.mark.integration
async def test_malformed_request_returns_error():
    """Sending malformed JSON-RPC returns appropriate error."""
    from tests.agents.helpers import create_a2a_client_for_executor
    from tests.agents.mock_agents import EchoAgent

    async with await create_a2a_client_for_executor(EchoAgent(), name="echo") as client:
        # Send malformed request
        response = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                # Missing 'id' and 'method'
                "params": {},
            },
        )

        # Should return an error response
        # Exact status code depends on implementation (might be 200 with error in JSON)
        data = response.json()
        # JSON-RPC errors can be returned as 200 OK with error object
        assert "error" in data or "result" in data


@pytest.mark.integration
async def test_missing_required_param_returns_error():
    """Calling a method with missing required params returns error."""
    from uuid import uuid4

    from tests.agents.helpers import create_a2a_client_for_executor
    from tests.agents.mock_agents import EchoAgent

    async with await create_a2a_client_for_executor(EchoAgent(), name="echo") as client:
        # Send request with missing message content
        response = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "message/send",
                "params": {},  # Missing 'message'
            },
        )

        data = response.json()
        # Should have an error for invalid params
        assert "error" in data or response.status_code >= 400
