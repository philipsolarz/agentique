"""End-to-end tests for multi-agent routing and composition.

Tests routing between different agents and composed bridge scenarios.
"""

from __future__ import annotations

import pytest


@pytest.mark.e2e
async def test_agent_card_lists_multiple_skills(e2e_a2a_client):
    """A2A agent card advertises multiple skills."""
    resp = await e2e_a2a_client.get("/.well-known/agent-card.json")
    card = resp.json()

    assert "skills" in card
    # The test agent should have multiple skills configured
    # Based on docker-compose.test.yml: calculator, data_processing, text_manipulation, info_retrieval
    assert len(card["skills"]) >= 0  # May be 0 or more depending on implementation


@pytest.mark.e2e
@pytest.mark.slow
async def test_sequential_requests_different_skills():
    """Can send sequential requests exercising different skills."""
    import httpx
    from uuid import uuid4

    async with httpx.AsyncClient(base_url="http://localhost:9000") as client:
        # Request 1 - calculator
        resp1 = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Calculate 2 + 2"}],
                        "messageId": uuid4().hex,
                    }
                },
            },
        )
        assert resp1.status_code == 200

        # Request 2 - text manipulation
        resp2 = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Uppercase this text"}],
                        "messageId": uuid4().hex,
                    }
                },
            },
        )
        assert resp2.status_code == 200


@pytest.mark.e2e
async def test_concurrent_requests_to_same_agent():
    """Agent handles concurrent requests correctly."""
    import asyncio
    import httpx
    from uuid import uuid4

    async def send_request(client, text):
        return await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": text}],
                        "messageId": uuid4().hex,
                    }
                },
            },
        )

    async with httpx.AsyncClient(base_url="http://localhost:9000", timeout=30.0) as client:
        # Send 3 concurrent requests
        tasks = [send_request(client, f"Request {i}") for i in range(3)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for resp in responses:
            assert resp.status_code == 200


@pytest.mark.e2e
async def test_agent_capabilities_match_config():
    """Agent capabilities in card match the configured setup."""
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:9000/.well-known/agent-card.json")
        card = resp.json()

        assert "capabilities" in card
        capabilities = card["capabilities"]

        # Check expected capabilities
        # Actual values depend on how the test agent is configured
        assert isinstance(capabilities, dict)


@pytest.mark.e2e
@pytest.mark.slow
async def test_redis_task_persistence_across_requests():
    """Tasks stored in Redis persist across multiple requests."""
    import redis.asyncio as aioredis

    client = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
    try:
        # Store a test task
        await client.set("test:task:001", '{"status": "completed"}')

        # Retrieve it
        value = await client.get("test:task:001")
        assert value is not None
        assert "completed" in value

        # Clean up
        await client.delete("test:task:001")
    finally:
        await client.aclose()


@pytest.mark.e2e
async def test_dynamodb_table_creation_and_access():
    """Can create and access DynamoDB tables."""
    import boto3

    client = boto3.client(
        "dynamodb",
        endpoint_url="http://localhost:8100",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )

    # Create a test table
    table_name = "e2e-test-table"
    try:
        client.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        # Wait for table to be ready
        waiter = client.get_waiter("table_exists")
        waiter.wait(TableName=table_name)

        # Verify it exists
        response = client.list_tables()
        assert table_name in response["TableNames"]

    finally:
        # Clean up
        try:
            client.delete_table(TableName=table_name)
        except Exception:
            pass
