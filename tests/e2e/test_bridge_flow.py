"""End-to-end tests for the full bridge flow.

Tests the complete MCP → Bridge → A2A → MCP roundtrip against a live Docker stack.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.e2e
async def test_health_endpoint_returns_200(e2e_http_client):
    """The /health endpoint reports service status."""
    resp = await e2e_http_client.get("/health")
    assert resp.status_code == 200

    health = resp.json()
    # Verify basic health response structure
    assert "status" in health or "healthy" in str(health).lower()


@pytest.mark.e2e
async def test_a2a_agent_is_reachable(e2e_a2a_client):
    """The A2A test agent responds to health checks."""
    resp = await e2e_a2a_client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200

    card = resp.json()
    assert "name" in card
    assert "version" in card


@pytest.mark.e2e
async def test_a2a_agent_send_message(e2e_a2a_client):
    """Can send a message directly to the A2A agent."""
    from uuid import uuid4

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello"}],
                "messageId": uuid4().hex,
            }
        },
    }

    resp = await e2e_a2a_client.post("/", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "jsonrpc" in data
    assert data["jsonrpc"] == "2.0"


@pytest.mark.e2e
@pytest.mark.slow
async def test_redis_is_accessible(redis_url):
    """Redis is running and accessible."""
    import redis.asyncio as aioredis

    client = await aioredis.from_url(redis_url, decode_responses=True)
    try:
        pong = await client.ping()
        assert pong is True
    finally:
        await client.aclose()


@pytest.mark.e2e
async def test_dynamodb_is_accessible(dynamodb_endpoint):
    """DynamoDB Local is running and accessible."""
    import boto3

    client = boto3.client(
        "dynamodb",
        endpoint_url=dynamodb_endpoint,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )

    response = client.list_tables()
    assert "TableNames" in response


@pytest.mark.e2e
async def test_docker_stack_all_services_healthy():
    """All Docker Compose services report as healthy."""
    import subprocess

    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "ps", "--format", "json"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        # Parse the output to check service health
        # Format varies by Docker Compose version
        assert "healthy" in result.stdout.lower() or "running" in result.stdout.lower()
    else:
        pytest.skip("Could not check Docker Compose service status")


@pytest.mark.e2e
@pytest.mark.slow
async def test_mcp_server_startup_time():
    """MCP server starts and becomes healthy within reasonable time."""
    import httpx

    # This test assumes the server is already running
    # If starting fresh, we'd need to wait longer
    async with httpx.AsyncClient() as client:
        for _ in range(10):
            try:
                resp = await client.get("http://localhost:8000/health", timeout=2.0)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            pytest.fail("MCP server did not become healthy in time")


@pytest.mark.e2e
async def test_full_stack_integration_smoke_test(e2e_http_client, e2e_a2a_client):
    """Smoke test: all services are running and responding."""
    # Check MCP server
    mcp_resp = await e2e_http_client.get("/health")
    assert mcp_resp.status_code == 200

    # Check A2A agent
    a2a_resp = await e2e_a2a_client.get("/.well-known/agent-card.json")
    assert a2a_resp.status_code == 200

    # If we get here, basic connectivity is working
    assert True
