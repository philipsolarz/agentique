# Testing Quick Start Guide

## Test Suite Overview

The agentique test suite uses a 3-tier architecture:

1. **Unit Tests** (`tests/unit/`) - Fast, isolated tests with no external dependencies
2. **Integration Tests** (`tests/integration/`) - Service-level tests using in-memory servers and testcontainers
3. **E2E Tests** (`tests/e2e/`) - Full-stack tests against Docker Compose

## Running Tests

### Install Dependencies

```bash
pip install -e ".[test]"
```

### Run Specific Test Tiers

```bash
# Unit tests only (fast, ~2 seconds)
pytest -m unit

# Integration tests only (medium speed, ~30-60 seconds)
pytest -m integration

# E2E tests only (requires Docker, ~2-3 minutes)
docker compose -f docker-compose.test.yml up -d --wait
pytest -m e2e
docker compose -f docker-compose.test.yml down -v

# All tests except E2E
pytest -m "not e2e"

# Run tests in parallel
pytest -m unit -n auto
```

### Selective Test Execution

```bash
# Run specific test file
pytest tests/integration/test_mcp_tools.py -v

# Run specific test
pytest tests/integration/test_mcp_tools.py::test_client_with_in_memory_transport -v

# Run tests matching a pattern
pytest -k "streaming" -v

# Run slow tests only
pytest -m slow

# Run integration tests but skip slow ones
pytest -m "integration and not slow"
```

### Coverage

```bash
# Run unit tests with coverage
pytest -m unit --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Docker Compose Stack

### Start the Test Stack

```bash
docker compose -f docker-compose.test.yml up -d --wait
```

### Check Service Health

```bash
# MCP Server
curl http://localhost:8000/health

# A2A Test Agent
curl http://localhost:9000/.well-known/agent-card.json

# Redis
redis-cli ping

# DynamoDB Local
curl http://localhost:8100/shell/
```

### View Logs

```bash
# All services
docker compose -f docker-compose.test.yml logs

# Specific service
docker compose -f docker-compose.test.yml logs mcp-server

# Follow logs
docker compose -f docker-compose.test.yml logs -f
```

### Stop the Stack

```bash
docker compose -f docker-compose.test.yml down -v --remove-orphans
```

## Test Markers

Tests are automatically marked based on their directory:

- `@pytest.mark.unit` - Unit tests (in `tests/unit/`)
- `@pytest.mark.integration` - Integration tests (in `tests/integration/`)
- `@pytest.mark.e2e` - E2E tests (in `tests/e2e/`)
- `@pytest.mark.slow` - Tests taking >5 seconds (manually added)

View all markers:

```bash
pytest --markers
```

## Test Organization

```
tests/
├── conftest.py              # Root config, marker registration
├── unit/                    # 25 existing unit tests
│   ├── conftest.py
│   └── test_*.py
├── integration/             # 9 new integration test files
│   ├── conftest.py          # Fixtures for Redis, DynamoDB, FastMCP
│   ├── test_mcp_tools.py
│   ├── test_a2a_adapter.py
│   ├── test_streaming.py
│   ├── test_transforms.py
│   ├── test_middleware.py
│   ├── test_persistence.py
│   ├── test_push_notifications.py
│   ├── test_observability.py
│   └── test_error_mapping.py
├── e2e/                     # 2 new e2e test files
│   ├── conftest.py          # Docker stack fixtures
│   ├── test_bridge_flow.py
│   └── test_multi_agent.py
└── agents/                  # Mock A2A agents
    ├── __init__.py
    ├── mock_agents.py       # 6 agent implementations
    └── helpers.py           # Test utilities
```

## CI/CD

GitHub Actions workflow runs automatically on push/PR:

1. **Unit Tests** - Fast feedback (~2 min)
2. **Integration Tests** - Service validation (~5 min)
3. **E2E Tests** - Full stack verification (~10 min)

View workflow: `.github/workflows/test.yml`

## Troubleshooting

### Integration Tests Failing

```bash
# Check if testcontainers can access Docker
docker ps

# Ensure no port conflicts
lsof -i :6379   # Redis
lsof -i :8100   # DynamoDB
```

### E2E Tests Skipped

The e2e tests auto-skip if the Docker stack isn't running:

```bash
# Start the stack first
docker compose -f docker-compose.test.yml up -d --wait

# Then run tests
pytest -m e2e
```

### Import Errors

```bash
# Ensure you're in the repo root
cd /home/cairon/git/AgentMCP

# Install in editable mode
pip install -e ".[test]"

# Check PYTHONPATH if needed
export PYTHONPATH="src:$PYTHONPATH"
```

## Adding New Tests

### Unit Test

```python
# tests/unit/test_myfeature.py
def test_my_feature():
    """Unit tests are auto-marked @pytest.mark.unit"""
    assert True
```

### Integration Test

```python
# tests/integration/test_myfeature.py
import pytest

@pytest.mark.integration
async def test_with_redis(redis_client):
    """Use fixtures from conftest.py"""
    await redis_client.set("key", "value")
    assert await redis_client.get("key") == "value"
```

### E2E Test

```python
# tests/e2e/test_myfeature.py
import pytest

@pytest.mark.e2e
async def test_full_flow(e2e_http_client):
    """Requires Docker stack running"""
    resp = await e2e_http_client.get("/health")
    assert resp.status_code == 200
```

## Performance Tips

1. Use `-n auto` for parallel execution (unit tests only)
2. Run integration tests without DynamoDB if only testing Redis
3. Use `pytest -x` to stop on first failure
4. Use `pytest --lf` to run only last failed tests
5. Use `pytest --sw` to run tests and pause on failures

## Resources

- Full testing strategy: `TESTING.md`
- Progress tracking: `TESTING_PROGRESS.md`
- CI workflow: `.github/workflows/test.yml`
- Docker stack: `docker-compose.test.yml`
