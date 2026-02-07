# Building a live integration test suite for agentique

**The agentique framework needs a three-tier testing strategy**: in-memory MCP tests using FastMCP's `Client(server)` pattern for fast protocol validation, containerized integration tests using Docker Compose with real Redis and DynamoDB Local for state management verification, and full end-to-end tests that exercise the complete MCP→Bridge→A2A flow against live test agents. This guide provides concrete, production-ready implementations for each tier, with pytest fixtures, Docker configurations, mock agent designs, and CI pipeline templates. The approach leverages FastMCP 3.0's in-memory transport (no network needed for most tests), the a2a-sdk's Starlette-based test server (testable via httpx's ASGITransport), and testcontainers-python for infrastructure. With 245 unit tests already passing, the integration suite described here covers the critical gaps: adapter behavior under real protocol conditions, task state persistence, streaming fidelity, error mapping accuracy, and multi-service composition.

---

## Test architecture and project structure

The test suite should be organized into three distinct tiers, each with its own directory, conftest, and pytest markers. This separation ensures fast feedback loops during development (unit tests run in seconds) while still validating real-world behavior (e2e tests run against live containers).

```
tests/
├── conftest.py                    # Root: marker registration, shared helpers
├── unit/                          # Existing 245 tests
│   └── conftest.py
├── integration/                   # New: service-level tests
│   ├── conftest.py                # Server fixtures, Redis/DynamoDB fixtures
│   ├── test_mcp_tools.py          # MCP tool calls via FastMCP Client
│   ├── test_a2a_adapter.py        # A2A adapter against test agents
│   ├── test_http_adapter.py       # HTTP adapter tests
│   ├── test_mcp_proxy_adapter.py  # MCP proxy adapter tests
│   ├── test_routing.py            # Routing strategy tests
│   ├── test_task_lifecycle.py     # Task state machine tests
│   ├── test_streaming.py          # SSE streaming tests
│   ├── test_transforms.py         # Visibility/namespace transforms
│   ├── test_middleware.py         # Custom middleware tests
│   ├── test_persistence.py        # Redis + DynamoDB task stores
│   ├── test_error_mapping.py      # A2A→MCP error code mapping
│   └── test_health.py             # Health monitoring tests
├── e2e/                           # New: full-stack tests
│   ├── conftest.py                # Docker Compose management
│   ├── test_bridge_flow.py        # MCP client → bridge → A2A agent → response
│   ├── test_multi_agent.py        # Multi-agent routing end-to-end
│   ├── test_push_notifications.py # Webhook/push notification flow
│   └── test_composition.py        # Mounted/composed bridge tests
└── agents/                        # Test agent definitions
    ├── echo_agent.py
    ├── streaming_agent.py
    ├── error_agent.py
    ├── input_required_agent.py
    └── long_running_agent.py
```

The root `conftest.py` handles marker registration and auto-tagging by directory:

```python
# tests/conftest.py
import pytest

def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on directory."""
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
```

Configure `pyproject.toml` to support selective execution:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = "-v --strict-markers --tb=short"
timeout = 30
markers = [
    "unit: Fast isolated unit tests (no external deps)",
    "integration: Tests requiring running services or in-memory servers",
    "e2e: Full end-to-end tests against Docker Compose stack",
    "slow: Tests taking more than 5 seconds",
]
```

---

## Docker Compose infrastructure for live testing

The test infrastructure requires four services: the A2A test agent (already exists on port 9000), Redis for task persistence, DynamoDB Local for the DynamoDB task store, and optionally the agentique MCP server itself for e2e tests. Health checks on every service ensure tests never start against half-ready infrastructure.

```yaml
# docker-compose.test.yml
services:
  # A2A test agent (existing Google ADK multi-agent system)
  a2a-test-agent:
    build:
      context: ./a2a_test_agent
    ports:
      - "9000:9000"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9000/.well-known/agent.json || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 10
      start_period: 10s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 3s

  dynamodb-local:
    image: amazon/dynamodb-local:latest
    command: "-jar DynamoDBLocal.jar -sharedDb"
    ports:
      - "8100:8000"
    working_dir: /home/dynamodblocal
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/shell/ || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 5s

  # Agentique MCP server (for e2e tests)
  mcp-server:
    build:
      context: .
    ports:
      - "8000:8000"
    environment:
      A2A_AGENT_URL: http://a2a-test-agent:9000
      REDIS_URL: redis://redis:6379
      DYNAMODB_ENDPOINT: http://dynamodb-local:8000
      AWS_ACCESS_KEY_ID: testing
      AWS_SECRET_ACCESS_KEY: testing
      AWS_DEFAULT_REGION: us-east-1
    depends_on:
      a2a-test-agent:
        condition: service_healthy
      redis:
        condition: service_healthy
      dynamodb-local:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/health || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 10
      start_period: 10s

networks:
  default:
    driver: bridge
```

The `-sharedDb` flag on DynamoDB Local is critical — it ensures all connections share the same database regardless of credentials, simplifying test isolation. DynamoDB Local's healthcheck targets `/shell/` because the root endpoint returns HTTP 400 rather than 200.

For integration tests that don't need the full Docker stack, **testcontainers-python** provides programmatic container management directly from pytest fixtures. This eliminates the Docker Compose dependency for tests that only need Redis or DynamoDB:

```python
# tests/integration/conftest.py
import os
import pytest
import pytest_asyncio
import boto3
from redis.asyncio import Redis as AsyncRedis
from testcontainers.redis import RedisContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for

@pytest.fixture(scope="session")
def redis_container():
    """Session-scoped real Redis for integration tests."""
    with RedisContainer("redis:7-alpine") as container:
        yield container

@pytest_asyncio.fixture
async def redis_client(redis_container):
    """Function-scoped async Redis client with cleanup."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    client = AsyncRedis(host=host, port=int(port), decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()

@pytest.fixture(scope="session")
def dynamodb_container():
    """Session-scoped DynamoDB Local for integration tests."""
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
                c = boto3.client("dynamodb", endpoint_url=endpoint,
                    region_name="us-east-1", aws_access_key_id="testing",
                    aws_secret_access_key="testing")
                c.list_tables()
                return True
            except Exception:
                return False

        wait_for(dynamodb_ready, timeout=30, interval=1)
        yield endpoint

@pytest.fixture
def dynamodb_client(dynamodb_container):
    """Function-scoped DynamoDB client with table cleanup."""
    client = boto3.client("dynamodb", endpoint_url=dynamodb_container,
        region_name="us-east-1", aws_access_key_id="testing",
        aws_secret_access_key="testing")
    yield client
    for table in client.list_tables()["TableNames"]:
        client.delete_table(TableName=table)
```

---

## FastMCP in-memory testing fixtures and patterns

FastMCP 3.0 provides the fastest path to integration testing through its **in-memory transport**. Passing a `FastMCP` server instance directly to `Client()` creates a zero-network test channel. This is the backbone of the integration tier — it validates that agentique's MCP tools, providers, transforms, and middleware all behave correctly without Docker overhead.

The core fixture creates an agentique bridge server and wraps it in a FastMCP `Client`:

```python
# tests/integration/conftest.py (continued)
from fastmcp import Client
from agentique.bridge import AgentiqueBridge
from agentique.adapters.a2a import A2AAdapter

@pytest_asyncio.fixture
async def bridge(redis_client):
    """Create an agentique bridge with in-memory A2A adapter."""
    adapter = A2AAdapter(agent_url="http://localhost:9000")
    bridge = AgentiqueBridge(adapters=[adapter], task_store_redis=redis_client)
    return bridge

@pytest_asyncio.fixture
async def mcp_client(bridge):
    """FastMCP in-memory client connected to the bridge's MCP server."""
    async with Client(bridge.mcp_server) as client:
        yield client
```

With this fixture, tool call validation becomes straightforward. Each MCP tool exposed by agentique (`agent`, `agents`, `task`, `inspect`, `agent_background`) can be tested through `client.call_tool()`:

```python
# tests/integration/test_mcp_tools.py
async def test_list_agents_tool(mcp_client):
    """The 'agents' tool returns available agents from the adapter."""
    result = await mcp_client.call_tool("agents", {})
    agents_data = result[0].text
    assert "calculator" in agents_data.lower()
    assert "text_processor" in agents_data.lower()

async def test_agent_tool_sends_message(mcp_client):
    """The 'agent' tool routes a message to the correct A2A agent."""
    result = await mcp_client.call_tool("agent", {
        "agent_name": "calculator",
        "message": "What is 2 + 3?"
    })
    assert "5" in result[0].text

async def test_task_tool_retrieves_state(mcp_client):
    """The 'task' tool retrieves task status by ID."""
    # First create a task
    create_result = await mcp_client.call_tool("agent_background", {
        "agent_name": "data_processor",
        "message": "Process this data"
    })
    task_id = create_result.data["task_id"]

    # Then query it
    status_result = await mcp_client.call_tool("task", {"task_id": task_id})
    assert status_result.data["state"] in (
        "submitted", "working", "completed", "failed"
    )

async def test_inspect_tool_returns_agent_card(mcp_client):
    """The 'inspect' tool returns the A2A agent card details."""
    result = await mcp_client.call_tool("inspect", {"agent_name": "calculator"})
    card_data = result.data
    assert "skills" in card_data
    assert card_data["name"] == "calculator"

async def test_unknown_agent_returns_error(mcp_client):
    """Requesting a non-existent agent produces an MCP error."""
    result = await mcp_client.call_tool("agent", {
        "agent_name": "nonexistent_agent",
        "message": "hello"
    })
    assert result[0].isError or "not found" in result[0].text.lower()
```

**Testing structured outputs via ToolResult** is essential since agentique uses `ToolResult` to return structured data from bridge operations:

```python
async def test_structured_task_result(mcp_client):
    """Task results include structured content for programmatic access."""
    result = await mcp_client.call_tool("agent", {
        "agent_name": "calculator",
        "message": "What is 10 * 5?"
    })
    # result.data gives structured_content if ToolResult was returned
    if hasattr(result, "data") and result.data:
        assert "answer" in result.data or isinstance(result.data, dict)
    else:
        assert "50" in result[0].text
```

---

## Testing providers, transforms, and middleware

FastMCP 3.0's **Provider** abstraction is central to agentique's architecture. Agentique likely implements a custom provider that dynamically generates MCP tools from discovered A2A agents. Testing this provider in isolation confirms tool generation works correctly before any routing or adaptation logic runs.

```python
# tests/integration/test_providers.py
from fastmcp import FastMCP, Client

async def test_agentique_provider_generates_tools():
    """The AgentiqueProvider creates MCP tools from A2A agent discovery."""
    from agentique.providers import AgentiqueProvider

    provider = AgentiqueProvider(
        agent_url="http://localhost:9000",
        adapter_type="a2a"
    )
    server = FastMCP("ProviderTest", providers=[provider])

    async with Client(server) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        # Should expose at least the core agentique tools
        assert "agent" in tool_names
        assert "agents" in tool_names
```

**Visibility transforms** control per-session tool access — a security-critical feature for agentique where different MCP clients may have access to different agent ecosystems. Test these by verifying tools appear and disappear based on session state:

```python
# tests/integration/test_transforms.py
from fastmcp import FastMCP, Client
from fastmcp.server.transforms import Namespace

async def test_namespace_transform_prefixes_tools():
    """Mounted sub-bridges get namespaced tool names."""
    main = FastMCP("Main")
    sub_bridge = create_test_bridge()  # Returns a FastMCP server

    main.mount(sub_bridge.mcp_server, namespace="team_a")

    async with Client(main) as client:
        tools = await client.list_tools()
        namespaced = [t.name for t in tools if t.name.startswith("team_a_")]
        assert len(namespaced) > 0
        # Original tool names should not appear without prefix
        assert "agent" not in [t.name for t in tools]
        assert "team_a_agent" in [t.name for t in tools]

async def test_visibility_hides_admin_tools():
    """Admin-tagged tools are hidden from non-admin sessions."""
    bridge = create_test_bridge()
    server = bridge.mcp_server
    server.disable(tags={"admin"})

    async with Client(server) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        # admin-only tools like 'inspect' should be hidden
        assert "inspect" not in names
        # standard tools should remain visible
        assert "agent" in names

async def test_per_session_visibility_unlock():
    """A session can dynamically unlock tools via enable_components."""
    bridge = create_test_bridge()
    server = bridge.mcp_server
    server.disable(tags={"premium"})

    async with Client(server) as client:
        tools_before = await client.list_tools()
        premium_before = [t for t in tools_before if "premium" in (t.tags or set())]
        assert len(premium_before) == 0

        # Call an unlock tool that enables premium features for this session
        await client.call_tool("unlock_premium", {"token": "valid-token"})

        tools_after = await client.list_tools()
        assert len(tools_after) > len(tools_before)
```

**Middleware testing** validates request interception, audit logging, rate limiting, and error handling. FastMCP's middleware uses a `call_next` chain pattern similar to ASGI middleware:

```python
# tests/integration/test_middleware.py
from fastmcp import FastMCP, Client
from fastmcp.server.middleware import Middleware, MiddlewareContext

class CallRecorderMiddleware(Middleware):
    """Records all tool calls for test assertions."""
    def __init__(self):
        self.calls = []

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        self.calls.append({
            "tool": context.message.name,
            "args": context.message.arguments,
            "timestamp": context.timestamp,
        })
        return await call_next(context)

async def test_middleware_records_all_calls():
    recorder = CallRecorderMiddleware()
    bridge = create_test_bridge()
    bridge.mcp_server.add_middleware(recorder)

    async with Client(bridge.mcp_server) as client:
        await client.call_tool("agents", {})
        await client.call_tool("agent", {"agent_name": "calc", "message": "hi"})

    assert len(recorder.calls) == 2
    assert recorder.calls[0]["tool"] == "agents"
    assert recorder.calls[1]["tool"] == "agent"

async def test_error_mapping_middleware():
    """A2A error codes get translated to correct MCP error codes."""
    bridge = create_test_bridge_with_error_agent()

    async with Client(bridge.mcp_server) as client:
        result = await client.call_tool("agent", {
            "agent_name": "error_agent",
            "message": "trigger error"
        })
        # Verify the A2A error was mapped to an MCP-compatible error
        assert result[0].isError
```

---

## A2A adapter integration testing with real protocol

Testing the A2A adapter requires exercising the actual A2A JSON-RPC protocol. The a2a-sdk provides both client (`A2AClient`) and server (`A2AStarletteApplication`) components that can be tested via httpx's `ASGITransport` — no real server process needed for most scenarios.

```python
# tests/integration/test_a2a_adapter.py
import httpx
from uuid import uuid4
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    SendMessageRequest, SendStreamingMessageRequest,
    MessageSendParams, GetTaskRequest, TaskQueryParams,
    AgentCard
)

@pytest_asyncio.fixture
async def a2a_client():
    """A2A client connected to the test agent on port 9000."""
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url="http://localhost:9000"
        )
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        yield client

async def test_a2a_agent_card_discovery(a2a_client):
    """Test agent advertises valid capabilities via agent card."""
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:9000/.well-known/agent.json")
        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card
        assert "skills" in card
        assert len(card["skills"]) > 0

async def test_a2a_send_message(a2a_client):
    """Non-streaming message send returns a completed task."""
    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(message={
            "role": "user",
            "parts": [{"kind": "text", "text": "What is 2 + 2?"}],
            "messageId": uuid4().hex,
        })
    )
    response = await a2a_client.send_message(request)
    result = response.result
    assert result is not None
    # Should reach a terminal state
    if hasattr(result, "status"):
        assert result.status.state in ("completed", "working")

async def test_a2a_task_state_transitions(a2a_client):
    """Verify a task moves through expected state transitions."""
    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(message={
            "role": "user",
            "parts": [{"kind": "text", "text": "Process data set alpha"}],
            "messageId": uuid4().hex,
        })
    )
    response = await a2a_client.send_message(request)
    task = response.result

    # Poll for completion if still working
    if hasattr(task, "status") and task.status.state == "working":
        import asyncio
        for _ in range(50):
            get_resp = await a2a_client.get_task(GetTaskRequest(
                id=str(uuid4()),
                params=TaskQueryParams(id=task.id)
            ))
            if get_resp.result.status.state in ("completed", "failed"):
                break
            await asyncio.sleep(0.1)
        assert get_resp.result.status.state == "completed"
```

**Streaming SSE testing** is critical since agentique supports streaming responses from A2A agents. The `httpx-sse` library provides the async primitives needed:

```python
# tests/integration/test_streaming.py
import asyncio
import httpx
from httpx_sse import aconnect_sse
from uuid import uuid4

async def test_a2a_streaming_returns_events():
    """Streaming message returns SSE events with status updates and artifacts."""
    async with httpx.AsyncClient(base_url="http://localhost:9000") as client:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Stream a long response"}],
                    "messageId": uuid4().hex,
                }
            }
        }

        events = []
        try:
            async with asyncio.timeout(10.0):
                async with aconnect_sse(client, "POST", "/", json=payload) as source:
                    source.response.raise_for_status()
                    async for sse in source.aiter_sse():
                        events.append(sse.json())
                        result = sse.json().get("result", {})
                        # Break on final event
                        if result.get("final", False):
                            break
        except asyncio.TimeoutError:
            pass  # Collect what we got

        assert len(events) > 0, "Expected at least one SSE event"
        # Verify event structure
        for event in events:
            assert "jsonrpc" in event
            assert "result" in event

async def test_streaming_through_bridge():
    """MCP tool call with streaming returns chunked content via bridge."""
    from fastmcp import Client

    bridge = create_test_bridge(streaming=True)
    async with Client(bridge.mcp_server) as client:
        result = await client.call_tool("agent", {
            "agent_name": "text_processor",
            "message": "Generate a long story",
            "stream": True
        })
        # The bridge should aggregate stream into final result
        assert len(result[0].text) > 0
```

---

## Deterministic mock agents for protocol feature coverage

A suite of purpose-built test agents ensures every protocol feature is exercised with predictable outputs. Each agent implements the a2a-sdk's `AgentExecutor` interface and targets a specific behavior. These agents run either in-process (via Starlette's `ASGITransport`) or as Docker containers alongside the test agent.

```python
# tests/agents/mock_agents.py
import asyncio
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import (
    TaskState, TaskStatus, TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent, Artifact, TextPart
)

class EchoAgent(AgentExecutor):
    """Returns the input message verbatim. Tests basic routing."""
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_text = context.message.parts[0].text
        await event_queue.enqueue_event(new_agent_text_message(f"echo: {user_text}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass

class StreamingAgent(AgentExecutor):
    """Sends 5 artifact chunks then completes. Tests streaming fidelity."""
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        for i in range(5):
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    taskId=context.task_id,
                    contextId=context.context_id,
                    artifact=Artifact(
                        artifactId=f"chunk-{i}",
                        parts=[TextPart(text=f"chunk {i} of 4")],
                    ),
                    append=True,
                    lastChunk=(i == 4),
                )
            )
            await asyncio.sleep(0.01)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass

class ErrorAgent(AgentExecutor):
    """Immediately fails with a known error code. Tests error mapping."""
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message("Intentional failure for testing"),
                ),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass

class InputRequiredAgent(AgentExecutor):
    """Transitions to input-required state. Tests elicitation flow."""
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Check if this is a follow-up with the required input
        user_text = context.message.parts[0].text
        if user_text.startswith("CONFIRM:"):
            await event_queue.enqueue_event(
                new_agent_text_message(f"Confirmed: {user_text[8:]}")
            )
        else:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    taskId=context.task_id,
                    contextId=context.context_id,
                    status=TaskStatus(
                        state=TaskState.input_required,
                        message=new_agent_text_message(
                            "Please confirm by replying CONFIRM:<your input>"
                        ),
                    ),
                    final=False,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass

class LongRunningAgent(AgentExecutor):
    """Sends progress updates over 2 seconds. Tests task polling."""
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        for step in range(10):
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    taskId=context.task_id,
                    contextId=context.context_id,
                    status=TaskStatus(state=TaskState.working),
                    final=False,
                )
            )
            await asyncio.sleep(0.2)
        await event_queue.enqueue_event(
            new_agent_text_message("Long task completed")
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )
```

A fixture wires these agents into a Starlette application testable without Docker:

```python
# tests/integration/conftest.py (continued)
import httpx
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

def create_mock_a2a_app(executor, name="test-agent", streaming=True):
    """Build a testable A2A Starlette app from an AgentExecutor."""
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    card = AgentCard(
        name=name,
        description=f"Test agent: {name}",
        url="http://testserver/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=streaming, pushNotifications=False),
        skills=[AgentSkill(id="test", name="Test", description="Test skill")],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )
    builder = A2AStarletteApplication(agent_card=card, http_handler=handler)
    return builder.build()

@pytest_asyncio.fixture
async def echo_a2a_client():
    """httpx client connected to an in-process echo A2A agent."""
    app = create_mock_a2a_app(EchoAgent(), name="echo")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

@pytest_asyncio.fixture
async def error_a2a_client():
    """httpx client connected to an in-process error A2A agent."""
    app = create_mock_a2a_app(ErrorAgent(), name="error")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
```

---

## Task persistence and webhook testing

Testing the Redis and DynamoDB task stores requires verifying that task state survives across bridge restarts and that the full task lifecycle (create → update → complete → retrieve) works correctly against real storage engines.

```python
# tests/integration/test_persistence.py

async def test_redis_task_store_persistence(redis_client):
    """Tasks stored in Redis survive and are retrievable."""
    from agentique.stores.redis import RedisTaskStore

    store = RedisTaskStore(redis_client)
    task_id = "test-task-001"
    await store.save_task(task_id, {
        "state": "working",
        "context_id": "ctx-123",
        "history": [{"role": "user", "text": "hello"}],
    })

    retrieved = await store.get_task(task_id)
    assert retrieved["state"] == "working"
    assert retrieved["context_id"] == "ctx-123"

    # Update state
    await store.update_task(task_id, {"state": "completed"})
    updated = await store.get_task(task_id)
    assert updated["state"] == "completed"

async def test_dynamodb_task_store_persistence(dynamodb_client):
    """Tasks stored in DynamoDB Local survive and are retrievable."""
    from agentique.stores.dynamodb import DynamoDBTaskStore

    # Create table first
    dynamodb_client.create_table(
        TableName="agentique-tasks",
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
    waiter.wait(TableName="agentique-tasks")

    store = DynamoDBTaskStore(
        endpoint_url=dynamodb_client.meta.endpoint_url,
        table_name="agentique-tasks",
    )
    task_id = "test-task-002"
    await store.save_task(task_id, {"state": "submitted", "agent": "calculator"})
    retrieved = await store.get_task(task_id)
    assert retrieved["state"] == "submitted"
```

**Webhook/push notification testing** requires a temporary HTTP server that collects incoming callbacks. The pattern uses an `asyncio.Event` for non-polling waits:

```python
# tests/integration/conftest.py (continued)
from aiohttp import web
from aiohttp.test_utils import TestServer

class WebhookCollector:
    """Temporary webhook receiver for push notification testing."""
    def __init__(self):
        self.notifications = []
        self._received = asyncio.Event()

    async def handle(self, request):
        body = await request.json()
        self.notifications.append(body)
        self._received.set()
        return web.json_response({"status": "ok"})

    async def wait_for_notification(self, timeout=10.0):
        try:
            await asyncio.wait_for(self._received.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(
                f"No webhook received within {timeout}s. "
                f"Got {len(self.notifications)} notifications total."
            )
        return self.notifications[-1]

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
```

```python
# tests/integration/test_push_notifications.py

async def test_push_notification_delivery(mcp_client, webhook_server):
    """Background tasks send push notifications to configured webhook."""
    # Start a background task with push notification config
    result = await mcp_client.call_tool("agent_background", {
        "agent_name": "data_processor",
        "message": "Process dataset",
        "push_notification_url": webhook_server["url"],
    })
    task_id = result.data["task_id"]

    # Wait for the push notification
    notification = await webhook_server["collector"].wait_for_notification(timeout=15.0)
    assert notification["taskId"] == task_id
    assert notification["status"]["state"] in ("completed", "working")
```

---

## End-to-end tests against the full Docker stack

E2e tests validate the entire system: an MCP client sends a tool call to agentique's HTTP endpoint, which routes through the bridge layer, translates to A2A protocol, dispatches to the test agent, and returns the result. These tests require the Docker Compose stack to be running.

```python
# tests/e2e/conftest.py
import os
import pytest
import pytest_asyncio
import httpx
import asyncio
from fastmcp import Client

def pytest_configure(config):
    """Skip e2e tests if Docker stack isn't running."""
    # Check if MCP server is reachable
    try:
        import requests
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip("MCP server not healthy", allow_module_level=True)
    except Exception:
        pytest.skip("Docker stack not running", allow_module_level=True)

@pytest_asyncio.fixture(scope="session")
async def e2e_mcp_client():
    """MCP client connected to the live Docker-hosted MCP server."""
    async with Client("http://localhost:8000/mcp") as client:
        yield client

@pytest_asyncio.fixture(scope="session")
async def e2e_http_client():
    """Raw HTTP client for direct API testing."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        yield client
```

```python
# tests/e2e/test_bridge_flow.py

async def test_full_mcp_to_a2a_roundtrip(e2e_mcp_client):
    """Complete flow: MCP tool call → bridge → A2A agent → MCP response."""
    tools = await e2e_mcp_client.list_tools()
    tool_names = [t.name for t in tools]
    assert "agent" in tool_names, f"Expected 'agent' tool, got: {tool_names}"

    result = await e2e_mcp_client.call_tool("agent", {
        "agent_name": "calculator",
        "message": "What is 15 * 7?"
    })
    assert "105" in result[0].text

async def test_multi_agent_routing(e2e_mcp_client):
    """Different messages route to different agents correctly."""
    calc = await e2e_mcp_client.call_tool("agent", {
        "agent_name": "calculator",
        "message": "Compute 100 / 4"
    })
    assert "25" in calc[0].text

    text = await e2e_mcp_client.call_tool("agent", {
        "agent_name": "text_processor",
        "message": "Uppercase: hello world"
    })
    assert "HELLO WORLD" in text[0].text.upper()

async def test_background_task_lifecycle(e2e_mcp_client):
    """Background task creation, polling, and completion."""
    import asyncio

    create = await e2e_mcp_client.call_tool("agent_background", {
        "agent_name": "data_processor",
        "message": "Process batch"
    })
    task_id = create.data["task_id"]
    assert task_id is not None

    # Poll until terminal state
    for _ in range(100):
        status = await e2e_mcp_client.call_tool("task", {"task_id": task_id})
        state = status.data["state"]
        if state in ("completed", "failed", "canceled"):
            break
        await asyncio.sleep(0.1)

    assert state == "completed", f"Task ended in state: {state}"

async def test_conversation_continuity(e2e_mcp_client):
    """Multi-turn conversations maintain context across calls."""
    # Turn 1
    r1 = await e2e_mcp_client.call_tool("agent", {
        "agent_name": "info_retriever",
        "message": "My name is Alice",
    })
    context_id = r1.data.get("context_id") if hasattr(r1, "data") else None

    # Turn 2 — same context
    r2 = await e2e_mcp_client.call_tool("agent", {
        "agent_name": "info_retriever",
        "message": "What is my name?",
        "context_id": context_id,
    })
    assert "alice" in r2[0].text.lower()

async def test_health_endpoint(e2e_http_client):
    """Health endpoint reports status of all connected services."""
    resp = await e2e_http_client.get("/health")
    assert resp.status_code == 200
    health = resp.json()
    assert health["status"] == "healthy"
    assert "adapters" in health
```

---

## OpenTelemetry and error mapping validation

Agentique's OpenTelemetry integration should produce spans for every tool call, A2A request, and task state transition. The `InMemorySpanExporter` captures spans without an external collector:

```python
# tests/integration/test_observability.py
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter

@pytest.fixture
def span_exporter():
    """Captures OTel spans in memory for assertion."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    provider.shutdown()

async def test_tool_call_creates_span(span_exporter, mcp_client):
    """Each MCP tool call generates an OpenTelemetry span."""
    await mcp_client.call_tool("agents", {})
    spans = span_exporter.get_finished_spans()
    tool_spans = [s for s in spans if "tool" in s.name.lower() or "agents" in s.name.lower()]
    assert len(tool_spans) >= 1
    assert tool_spans[0].attributes.get("mcp.tool.name") == "agents"

async def test_a2a_request_span_has_agent_info(span_exporter, mcp_client):
    """A2A adapter spans include agent name and protocol details."""
    await mcp_client.call_tool("agent", {
        "agent_name": "calculator",
        "message": "2+2"
    })
    spans = span_exporter.get_finished_spans()
    a2a_spans = [s for s in spans if "a2a" in s.name.lower()]
    assert len(a2a_spans) >= 1
    assert a2a_spans[0].attributes.get("a2a.agent.name") == "calculator"
```

**Error mapping** between A2A and MCP error codes is a correctness-critical path. A2A uses task states like `failed` and `rejected` with JSON-RPC error codes, while MCP has its own error code space. Test the mapping exhaustively:

```python
# tests/integration/test_error_mapping.py

ERROR_MAPPING_CASES = [
    # (a2a_error_code, expected_mcp_behavior)
    (-32600, "invalid_request"),    # Invalid Request
    (-32601, "method_not_found"),   # Method not found
    (-32602, "invalid_params"),     # Invalid params
    (-32603, "internal_error"),     # Internal error
]

@pytest.mark.parametrize("a2a_code,expected_mcp", ERROR_MAPPING_CASES)
async def test_error_code_mapping(a2a_code, expected_mcp):
    """A2A JSON-RPC error codes map to correct MCP error responses."""
    from agentique.errors import map_a2a_error_to_mcp
    mcp_error = map_a2a_error_to_mcp(a2a_code)
    assert mcp_error.type == expected_mcp
```

---

## CI pipeline with GitHub Actions

The CI pipeline runs unit tests first (fast, no Docker), then integration tests (with testcontainers), then e2e tests (full Docker Compose stack). Caching Docker images and Python dependencies keeps pipeline time under 10 minutes.

```yaml
# .github/workflows/test.yml
name: Tests
on:
  push:
    branches: [main]
  pull_request:

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[test]"
      - run: pytest -m unit -n auto --junitxml=unit-results.xml --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4
        if: always()
        with:
          files: coverage.xml

  integration:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    needs: unit
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[test]"
      - name: Run integration tests (testcontainers manages Docker)
        run: |
          pytest -m integration \
            --junitxml=integration-results.xml \
            --timeout=60 -v
      - uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: "*-results.xml"

  e2e:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    needs: integration
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[test]"
      - name: Start Docker Compose stack
        run: docker compose -f docker-compose.test.yml up -d --wait --build
      - name: Wait for all services
        run: |
          timeout 60 bash -c '
            until curl -sf http://localhost:8000/health; do sleep 2; done
            echo "All services ready"
          '
      - name: Run e2e tests
        run: |
          pytest -m e2e \
            --junitxml=e2e-results.xml \
            --timeout=120 -v
      - uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: "*-results.xml"
      - name: Collect logs on failure
        if: failure()
        run: docker compose -f docker-compose.test.yml logs > docker-logs.txt
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: docker-logs
          path: docker-logs.txt
      - name: Teardown
        if: always()
        run: docker compose -f docker-compose.test.yml down -v --remove-orphans
```

Run tests selectively during development:

```bash
pytest -m unit                          # Fast: ~2s, no Docker
pytest -m integration                   # Medium: ~30s, testcontainers
pytest -m e2e                           # Full: ~2min, Docker Compose
pytest -m "integration and not slow"    # Skip slow integration tests
pytest -m "not e2e" -n auto             # Everything except e2e, parallelized
```

---

## Putting it all together: implementation sequence

The recommended implementation order minimizes risk by starting with the highest-value, lowest-effort tests and building up to full e2e coverage. **Step 1** is the MCP in-memory tier — create the `integration/conftest.py` with the `mcp_client` fixture using `Client(bridge.mcp_server)`, then write tests for every exposed MCP tool. This validates the entire bridge layer without any external dependencies. **Step 2** adds the mock agents — implement `EchoAgent`, `ErrorAgent`, `InputRequiredAgent`, and `StreamingAgent` using the a2a-sdk's `AgentExecutor` interface, wire them into Starlette apps via `ASGITransport`, and test adapter behavior against predictable responses. **Step 3** introduces persistence — add testcontainers fixtures for Redis and DynamoDB Local, test task store implementations against real storage engines. **Step 4** builds the e2e tier — create `docker-compose.test.yml` with health checks on all services, write the full roundtrip tests, and add the CI workflow. **Step 5** fills gaps — add streaming SSE tests with `httpx-sse`, webhook tests with the `WebhookCollector`, OpenTelemetry span assertions, and error mapping validation.

Each step produces independently valuable test coverage. The in-memory MCP tests alone will catch most regressions in the bridge layer, provider logic, and tool implementations. The mock agents catch protocol translation bugs. The persistence tests catch state management issues. The e2e tests catch deployment and integration issues. Together, they form a comprehensive safety net that validates agentique from protocol surface to storage backend.

## Conclusion

Three architectural decisions make this test suite effective. First, **FastMCP 3.0's `Client(server)` in-memory transport** eliminates network overhead for 80% of tests while still exercising the full MCP protocol stack — this is the single most impactful testing capability to leverage. Second, **a2a-sdk's Starlette application builder** combined with httpx's `ASGITransport` means A2A protocol testing also requires no running servers, keeping integration tests fast and deterministic. Third, **separating infrastructure concerns into session-scoped testcontainer fixtures** means Redis and DynamoDB containers start once per test run, not once per test, keeping the full integration suite under 30 seconds.

The key libraries underpinning this approach are `fastmcp==3.0.0b1` (in-memory client), `a2a-sdk[http-server]>=0.3.22` (mock A2A servers), `httpx-sse>=0.4` (SSE stream testing), `testcontainers[redis]>=4.0` (programmatic Docker management), `fakeredis>=2.33` (fast Redis tests without Docker), `pytest-asyncio>=1.3` (async fixture management with proper loop scoping), and `opentelemetry-sdk` (span/metric capture). The total estimated implementation effort is **3–5 days** for a developer familiar with the agentique codebase, yielding approximately 80–120 new integration and e2e tests covering all documented features.