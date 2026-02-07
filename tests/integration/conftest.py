"""Conftest for integration tests.

Provides fixtures for:
- Testcontainers (Redis, DynamoDB Local) - requires Docker
- FastMCP Client with in-memory transport
- Mock A2A agents
- Webhook collectors for push notification testing
"""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ============================================================================
# Docker Detection
# ============================================================================

def _docker_available() -> bool:
    """Check if Docker is available for testcontainers."""
    if shutil.which("docker") is None:
        return False
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


_HAS_DOCKER = _docker_available()


# ============================================================================
# Testcontainers Fixtures (require Docker)
# ============================================================================


@pytest.fixture(scope="session")
def redis_container():
    """Session-scoped real Redis for integration tests."""
    if not _HAS_DOCKER:
        pytest.skip("Docker not available for testcontainers")
    from testcontainers.redis import RedisContainer
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest_asyncio.fixture
async def redis_client(redis_container) -> AsyncIterator:
    """Function-scoped async Redis client with cleanup."""
    from redis.asyncio import Redis as AsyncRedis
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    client = AsyncRedis(host=host, port=int(port), decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest.fixture(scope="session")
def dynamodb_container():
    """Session-scoped DynamoDB Local for integration tests."""
    if not _HAS_DOCKER:
        pytest.skip("Docker not available for testcontainers")
    import boto3
    from testcontainers.core.container import DockerContainer
    container = (
        DockerContainer("amazon/dynamodb-local:latest")
        .with_exposed_ports(8000)
        .with_command("-jar DynamoDBLocal.jar -sharedDb")
    )

    with container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(8000)
        endpoint = f"http://{host}:{port}"

        def dynamodb_ready():
            try:
                c = boto3.client(
                    "dynamodb",
                    endpoint_url=endpoint,
                    region_name="us-east-1",
                    aws_access_key_id="testing",
                    aws_secret_access_key="testing",
                )
                c.list_tables()
                return True
            except Exception:
                return False

        import time
        for _ in range(30):
            if dynamodb_ready():
                break
            time.sleep(1)
        else:
            raise RuntimeError("DynamoDB container did not become ready")
        yield endpoint


@pytest.fixture
def dynamodb_client(dynamodb_container):
    """Function-scoped DynamoDB client with table cleanup."""
    import boto3
    client = boto3.client(
        "dynamodb",
        endpoint_url=dynamodb_container,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    yield client
    for table in client.list_tables()["TableNames"]:
        client.delete_table(TableName=table)


# ============================================================================
# Mock A2A Agent Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def echo_a2a_client() -> AsyncIterator[httpx.AsyncClient]:
    """httpx client connected to an in-process echo A2A agent."""
    from tests.agents.helpers import create_a2a_client_for_executor
    from tests.agents.mock_agents import EchoAgent

    async with await create_a2a_client_for_executor(EchoAgent(), name="echo") as client:
        yield client


@pytest_asyncio.fixture
async def streaming_a2a_client() -> AsyncIterator[httpx.AsyncClient]:
    """httpx client connected to an in-process streaming A2A agent."""
    from tests.agents.helpers import create_a2a_client_for_executor
    from tests.agents.mock_agents import StreamingAgent

    async with await create_a2a_client_for_executor(StreamingAgent(), name="streaming") as client:
        yield client


@pytest_asyncio.fixture
async def error_a2a_client() -> AsyncIterator[httpx.AsyncClient]:
    """httpx client connected to an in-process error A2A agent."""
    from tests.agents.helpers import create_a2a_client_for_executor
    from tests.agents.mock_agents import ErrorAgent

    async with await create_a2a_client_for_executor(ErrorAgent(), name="error") as client:
        yield client


@pytest_asyncio.fixture
async def calculator_a2a_client() -> AsyncIterator[httpx.AsyncClient]:
    """httpx client connected to an in-process calculator A2A agent."""
    from tests.agents.helpers import create_a2a_client_for_executor
    from tests.agents.mock_agents import CalculatorAgent

    async with await create_a2a_client_for_executor(CalculatorAgent(), name="calculator") as client:
        yield client


# ============================================================================
# Webhook/Push Notification Fixtures
# ============================================================================


class WebhookCollector:
    """Temporary webhook receiver for push notification testing."""

    def __init__(self):
        self.notifications: list[dict[str, Any]] = []
        self._received = asyncio.Event()

    async def handle(self, request: web.Request) -> web.Response:
        """Handle incoming webhook POST."""
        body = await request.json()
        self.notifications.append(body)
        self._received.set()
        return web.json_response({"status": "ok"})

    async def wait_for_notification(self, timeout: float = 10.0) -> dict[str, Any]:
        """Wait for a notification to arrive."""
        try:
            await asyncio.wait_for(self._received.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(
                f"No webhook received within {timeout}s. "
                f"Got {len(self.notifications)} notifications total."
            )
        return self.notifications[-1]

    def clear(self) -> None:
        """Clear all collected notifications."""
        self.notifications.clear()
        self._received.clear()


@pytest_asyncio.fixture
async def webhook_server():
    """Starts a temporary webhook server, returns collector with URL."""
    collector = WebhookCollector()
    app = web.Application()
    app.router.add_post("/webhook", collector.handle)
    server = TestServer(app)
    await server.start_server()

    yield {
        "url": f"http://{server.host}:{server.port}/webhook",
        "collector": collector,
        "server": server,
    }

    await server.close()


# ============================================================================
# Helper Fixtures
# ============================================================================


@pytest.fixture
def test_env_vars(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
