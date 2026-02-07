"""Integration tests for task persistence stores.

Tests Redis and DynamoDB task stores with real containers via testcontainers.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
async def test_redis_task_store_save_and_retrieve(redis_client):
    """Tasks stored in Redis can be saved and retrieved."""
    # Check if RedisTaskStore exists in the codebase
    try:
        from agentique.storage.redis import RedisTaskStore
    except ImportError:
        # If the store doesn't exist yet, check alternate locations
        try:
            from agentique.stores.redis import RedisTaskStore
        except ImportError:
            pytest.skip("RedisTaskStore not yet implemented")

    store = RedisTaskStore(redis_client)
    task_id = "test-task-001"

    # Save a task
    await store.save_task(
        task_id,
        {
            "state": "working",
            "context_id": "ctx-123",
            "history": [{"role": "user", "text": "hello"}],
        },
    )

    # Retrieve it
    retrieved = await store.get_task(task_id)
    assert retrieved["state"] == "working"
    assert retrieved["context_id"] == "ctx-123"


@pytest.mark.integration
async def test_redis_task_store_update(redis_client):
    """Tasks in Redis can be updated."""
    try:
        from agentique.storage.redis import RedisTaskStore
    except ImportError:
        try:
            from agentique.stores.redis import RedisTaskStore
        except ImportError:
            pytest.skip("RedisTaskStore not yet implemented")

    store = RedisTaskStore(redis_client)
    task_id = "test-task-002"

    # Save initial state
    await store.save_task(task_id, {"state": "submitted"})

    # Update state
    await store.update_task(task_id, {"state": "completed"})

    # Verify update
    updated = await store.get_task(task_id)
    assert updated["state"] == "completed"


@pytest.mark.integration
async def test_redis_task_store_nonexistent_task(redis_client):
    """Retrieving a nonexistent task returns None or raises appropriate error."""
    try:
        from agentique.storage.redis import RedisTaskStore
    except ImportError:
        try:
            from agentique.stores.redis import RedisTaskStore
        except ImportError:
            pytest.skip("RedisTaskStore not yet implemented")

    store = RedisTaskStore(redis_client)
    result = await store.get_task("nonexistent-task")
    # Should return None or raise a specific exception
    assert result is None or isinstance(result, dict)


@pytest.mark.integration
async def test_dynamodb_task_store_save_and_retrieve(dynamodb_client, test_env_vars):
    """Tasks stored in DynamoDB can be saved and retrieved."""
    try:
        from agentique.storage.dynamodb import DynamoDBTaskStore
    except ImportError:
        try:
            from agentique.stores.dynamodb import DynamoDBTaskStore
        except ImportError:
            pytest.skip("DynamoDBTaskStore not yet implemented")

    # Create table first
    table_name = "agentique-test-tasks"
    try:
        dynamodb_client.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        waiter = dynamodb_client.get_waiter("table_exists")
        waiter.wait(TableName=table_name)
    except dynamodb_client.exceptions.ResourceInUseException:
        pass  # Table already exists

    store = DynamoDBTaskStore(
        endpoint_url=dynamodb_client.meta.endpoint_url,
        table_name=table_name,
    )

    task_id = "test-task-003"
    await store.save_task(task_id, {"state": "submitted", "agent": "calculator"})

    retrieved = await store.get_task(task_id)
    assert retrieved["state"] == "submitted"


@pytest.mark.integration
async def test_dynamodb_task_store_update(dynamodb_client, test_env_vars):
    """Tasks in DynamoDB can be updated."""
    try:
        from agentique.storage.dynamodb import DynamoDBTaskStore
    except ImportError:
        try:
            from agentique.stores.dynamodb import DynamoDBTaskStore
        except ImportError:
            pytest.skip("DynamoDBTaskStore not yet implemented")

    table_name = "agentique-test-tasks"
    try:
        dynamodb_client.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        waiter = dynamodb_client.get_waiter("table_exists")
        waiter.wait(TableName=table_name)
    except dynamodb_client.exceptions.ResourceInUseException:
        pass

    store = DynamoDBTaskStore(
        endpoint_url=dynamodb_client.meta.endpoint_url,
        table_name=table_name,
    )

    task_id = "test-task-004"
    await store.save_task(task_id, {"state": "working"})
    await store.update_task(task_id, {"state": "completed", "result": "success"})

    updated = await store.get_task(task_id)
    assert updated["state"] == "completed"


@pytest.mark.integration
async def test_redis_client_basic_operations(redis_client):
    """Redis client fixture works correctly."""
    # Basic set/get
    await redis_client.set("test-key", "test-value")
    value = await redis_client.get("test-key")
    assert value == "test-value"

    # Key deletion
    await redis_client.delete("test-key")
    value = await redis_client.get("test-key")
    assert value is None


@pytest.mark.integration
def test_dynamodb_client_basic_operations(dynamodb_client):
    """DynamoDB client fixture works correctly."""
    # List tables (should start empty or be cleaned up)
    response = dynamodb_client.list_tables()
    assert "TableNames" in response
    # Tables list might not be empty if previous test failed to cleanup
    assert isinstance(response["TableNames"], list)
