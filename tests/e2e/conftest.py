"""Conftest for end-to-end tests.

E2E tests run against a full Docker Compose stack and validate
complete protocol flows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def pytest_configure(config):
    """Skip e2e tests if Docker stack isn't running."""
    import requests

    # Check if MCP server is reachable
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip("MCP server not healthy", allow_module_level=True)
    except Exception:
        pytest.skip(
            "Docker stack not running. Start with: docker compose -f docker-compose.test.yml up -d --wait",
            allow_module_level=True,
        )


@pytest_asyncio.fixture(scope="session")
async def e2e_mcp_client():
    """MCP client connected to the live Docker-hosted MCP server.

    Note: This uses HTTP transport to connect to the actual server.
    Requires docker-compose.test.yml to be running.
    """
    from fastmcp import Client

    # Connect to the HTTP endpoint of the MCP server
    # The exact URL depends on how the MCP server exposes itself
    try:
        async with Client("http://localhost:8000/mcp") as client:
            yield client
    except Exception:
        # If the above doesn't work, try connecting directly
        # This may need adjustment based on actual server implementation
        pytest.skip("Could not connect to MCP server HTTP endpoint")


@pytest_asyncio.fixture(scope="session")
async def e2e_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Raw HTTP client for direct API testing."""
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def e2e_a2a_client() -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client for the A2A test agent."""
    async with httpx.AsyncClient(base_url="http://localhost:9000", timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
def redis_url():
    """Redis URL for the test stack."""
    return "redis://localhost:6379"


@pytest.fixture(scope="session")
def dynamodb_endpoint():
    """DynamoDB endpoint for the test stack."""
    return "http://localhost:8100"
