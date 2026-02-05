# Agentique: bridging MCP and agent ecosystems

**Agentique should become the protocol-agnostic bridge between MCP clients and any agent ecosystem.** The library currently bridges MCP to A2A agents using FastMCP 3.0, but research reveals significant untapped features in FastMCP 3.0's provider/transform/middleware architecture, missing A2A protocol capabilities (push notifications, gRPC, extensions), and clear design patterns from top Python libraries that would make agentique genuinely unopinionated and extensible. This report provides a concrete roadmap for evolving agentique from an A2A-specific bridge into a generic, pluggable agent protocol gateway — one that treats A2A as merely its first backend adapter.

---

## Proposed mission statement and architectural vision

**Mission statement**: *Agentique is a Python framework that exposes any agent ecosystem as a first-class MCP server. It translates MCP's tool/resource/prompt primitives into agent protocol operations, letting any MCP client — Claude, Cursor, ChatGPT, or custom hosts — seamlessly interact with remote agents regardless of their underlying protocol.*

The core architectural insight is that agentique sits at the intersection of three rapidly maturing specifications: **MCP** (the client-facing protocol, now at its 2025-11-25 spec with tasks, structured output, and extensions), **A2A** (the first agent-to-agent protocol, now at v0.3 with gRPC support and heading toward v1.0), and **FastMCP 3.0** (the server framework, now with providers, transforms, and middleware). Each of these has evolved substantially, and agentique must leverage all three fully while remaining open to future protocols.

The recommended architecture follows three layers:

- **Protocol Layer** (top): FastMCP 3.0 server exposing MCP primitives to clients
- **Bridge Layer** (middle): Protocol-agnostic routing, translation, and state management  
- **Adapter Layer** (bottom): Pluggable backends — A2A first, then OpenAI Agents API, LangChain Runnable endpoints, custom HTTP agents, etc.

Each layer communicates through **Protocol classes** (structural subtyping), uses **middleware chains** for cross-cutting concerns, and emits **events** for observability. This mirrors how FastAPI achieves unopinionated design while remaining highly capable.

---

## FastMCP 3.0 features agentique should leverage

FastMCP 3.0 (currently v3.0.0b1, released January 2026) introduced a fundamentally new architecture built on three primitives: **Components** (tools/resources/prompts), **Providers** (where components come from), and **Transforms** (middleware that modifies components as they flow to clients). Agentique's `A2AAgentProvider` correctly uses the Provider pattern, but several powerful features remain untapped.

**Transforms should replace custom filtering logic.** FastMCP 3.0's `Transform` classes — `Namespace`, `ToolTransform`, `Visibility`, `ResourcesAsTools`, `PromptsAsTools` — provide a composable middleware pipeline for component modification. Instead of agentique implementing its own tool name prefixing or visibility logic, it should expose transform hooks. For example, users could apply a `Namespace` transform per agent to prevent tool name collisions, or a custom `ToolTransform` to rename verbose agent skills into concise tool names. The `Visibility` transform with **session-level control** via `ctx.enable_components()` / `ctx.disable_components()` enables dynamic agent availability — a user could "unlock" premium agents mid-session.

**Middleware should handle cross-cutting concerns.** FastMCP 3.0's `Middleware` class provides `on_call_tool`, `on_list_tools`, `on_read_resource`, and other hooks in a chain-of-responsibility pattern. Agentique should use this for logging, rate limiting, authentication forwarding, and metrics collection rather than embedding these in the bridge layer. An `AuthMiddleware` with `AuthContext` already exists for server-wide enforcement.

**Composition via mounting enables multi-bridge architectures.** FastMCP's `mount()` creates live sub-servers with namespacing. Agentique could mount separate bridges — one for A2A agents, one for OpenAI-compatible agents, one for local agents — under a single MCP server with automatic namespace isolation. The `create_proxy()` function could proxy to remote MCP servers that themselves wrap agents.

**Background tasks need the full TaskConfig API.** Agentique uses `task=True` but should leverage `TaskConfig(mode="optional", poll_interval=timedelta(seconds=2))` for fine-grained control. The `Progress` dependency injection (`from fastmcp.dependencies import Progress`) provides clean progress reporting: `await progress.set_total(n)`, `await progress.increment()`, `await progress.set_message("Processing agent X")`.

**Additional FastMCP 3.0 features to adopt:**

- **Dependency injection** via `Depends()` — inject agent clients, configuration, and services into tools cleanly
- **OpenTelemetry integration** — zero-config tracing with `fastmcp.*` span attributes; agentique should add `agentique.agent_name`, `agentique.protocol`, `agentique.task_state` attributes
- **Storage backends** — pluggable persistent state (Redis, DynamoDB, filesystem) for task state instead of in-memory dicts
- **Elicitation with response types** — `ctx.elicit()` supports Pydantic models, enabling structured confirmation dialogs for agent actions
- **Sampling with tool loop** — `ctx.sample()` now supports tools and `tool_choice`, enabling agentique to use LLM-driven routing for agent selection
- **Lifespan composition** — pipe operator (`lifespan_a | lifespan_b`) for composing startup/shutdown logic across multiple agent connections
- **Structured content** — tools returning dicts/Pydantic models get automatic structured JSON alongside traditional content, aligning with MCP's new `outputSchema`/`structuredContent`

---

## Missing A2A protocol features to implement

The A2A protocol has matured significantly to v0.3.0 (July 2025) with an RC v1.0 on the horizon. It now supports **three protocol bindings** (JSON-RPC, gRPC, REST), an **extensions mechanism**, and a richer task lifecycle. Several features are absent from the current agentique implementation.

**Push notifications are critical for production deployments.** A2A supports webhook-based push notifications for long-running tasks where clients can't maintain persistent connections. The server POSTs task updates to a client-specified URL, secured via JWT signing. Agentique should implement `PushNotificationConfig` support — when an MCP client starts a background task, agentique should configure push notifications on the A2A server and translate incoming webhooks into MCP `notifications/tasks/status` events. The SDK provides `InMemoryPushNotifier` and `PushNotificationConfigStore` interfaces.

**Task resubscription enables resilient streaming.** A2A's `tasks/resubscribe` method allows reconnecting to an active stream after disconnection. Agentique's `RouterBridge` should implement reconnection logic — if an SSE connection drops, it should resubscribe using the task ID rather than failing the MCP request.

**gRPC transport should be a backend option.** The A2A SDK now includes `GrpcTransport` alongside `JsonRpcTransport` and `RestTransport`. For high-throughput deployments, agentique should support gRPC as a backend transport via `ClientConfig(ordered_transports=["gRPC", "JSONRPC"])`.

**Context ID management needs proper implementation.** A2A uses `contextId` to group related tasks into conversations and `taskId` for individual operations. Agentique should map MCP session IDs to A2A context IDs and maintain this mapping in its state store. The rules are: agents MUST infer contextId from task if only taskId is provided, and agents MUST reject messages with mismatching contextId and taskId.

**The extensions mechanism should be exposed.** A2A extensions add custom data to tasks, messages, parts, and agent card capabilities. Agentique should allow users to define extensions that are propagated to A2A agents and received back in responses — for example, a tracing extension that carries OpenTelemetry span context.

**Proper error code mapping is needed.** A2A defines specific error codes: `-32001` (TaskNotFound), `-32002` (ContentTypeNotSupported), `-32003` (UnsupportedOperation). Agentique should map these to appropriate MCP error responses rather than generic internal errors. Additionally, A2A's `auth-required` and `rejected` task states should be translated into MCP elicitation or error flows.

**Extended Agent Cards should be supported.** A2A distinguishes between public agent cards (unauthenticated, at `/.well-known/agent-card.json`) and extended agent cards (authenticated, revealing additional skills). Agentique should fetch extended cards when credentials are available, exposing more tools to authenticated MCP clients.

---

## Design patterns for an unopinionated architecture

Research into top Python libraries reveals consistent patterns for achieving extensibility without imposing opinions. The following patterns should form agentique's architectural backbone.

**Protocol classes over abstract base classes.** Python's `typing.Protocol` enables structural subtyping — any class with matching methods satisfies the interface without inheriting from a base class. This is how agentique should define its adapter interface:

```python
from typing import Protocol, AsyncIterator, runtime_checkable

@runtime_checkable
class AgentAdapter(Protocol):
    async def discover_agents(self) -> list[AgentInfo]: ...
    async def send_message(self, agent_id: str, message: str, 
                          context: BridgeContext) -> AsyncIterator[AgentEvent]: ...
    async def get_task_state(self, task_id: str) -> TaskState: ...
    async def cancel_task(self, task_id: str) -> None: ...
```

Users implementing an adapter for OpenAI's Agents API or a custom protocol simply write a class with these methods — no inheritance required. This mirrors PydanticAI's `AbstractToolset` approach, which research identified as the most "unopinionated" agent framework design.

**Middleware chains for processing pipelines.** Following FastMCP 3.0's own middleware pattern and ASGI conventions, agentique should implement a middleware chain at the bridge layer:

```python
class BridgeMiddleware(Protocol):
    async def process(self, request: BridgeRequest, 
                     call_next: Callable) -> BridgeResponse: ...
```

This enables users to inject logging, rate limiting, authentication forwarding, message transformation, and caching without modifying core code. Each middleware wraps the next, creating a composable stack identical to Starlette's middleware model.

**Event hooks via an async event emitter.** Research shows that blinker (Flask/Celery), SQLAlchemy's event system, and LangChain's callback handlers all follow the observer pattern. Agentique should emit events at key lifecycle points:

- `agent.discovered` / `agent.lost` — agent availability changes
- `tool.called` / `tool.completed` / `tool.failed` — tool invocation lifecycle  
- `task.created` / `task.state_changed` / `task.completed` — task lifecycle
- `message.sent` / `message.received` — message flow
- `stream.chunk` — streaming data

Users subscribe to events for observability, custom metrics, or side effects. Supporting both sync and async handlers with `asyncio.gather()` ensures flexibility.

**Registry pattern with entry points for adapters.** Built-in adapters (A2A, future OpenAI) use a decorator-based registry. Third-party adapters use setuptools entry points:

```python
# Built-in
@register_adapter("a2a")
class A2AAdapter: ...

# Third-party (pyproject.toml)
[project.entry-points."agentique.adapters"]
openai = "agentique_openai:OpenAIAdapter"
```

This mirrors pytest's pluggy-based discovery and Celery's broker/backend URL pattern. Discovery is automatic: `pip install agentique-openai` makes the adapter available.

**Pydantic Settings for layered configuration.** Configuration should use `pydantic_settings.BaseSettings` with environment variable support, `.env` files, and type validation:

```python
class AgentiqueConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTIQUE_")
    
    transport: Literal["stdio", "streamable-http", "sse"] = "streamable-http"
    host: str = "127.0.0.1"
    port: int = 8000
    task_backend: str = "memory://"  # or "redis://..."
    default_timeout: float = 30.0
    adapters: dict[str, AdapterConfig] = {}
```

---

## Concrete code improvements and implementation recommendations

### Separation of concerns through a layered package structure

The library should adopt a modular package architecture inspired by LangChain's `core` / `implementations` / `integrations` split:

```
agentique/
├── core/                    # Protocols, types, interfaces — zero dependencies
│   ├── protocols.py         # AgentAdapter, ToolMapper, EventHandler protocols
│   ├── types.py             # AgentInfo, BridgeContext, TaskState, AgentEvent
│   ├── events.py            # AsyncEventEmitter
│   └── config.py            # BaseSettings subclasses
├── bridge/                  # Core bridge logic
│   ├── provider.py          # AgentProvider (FastMCP Provider)
│   ├── router.py            # AgentRouter — strategy-based agent selection
│   ├── translator.py        # MCP ↔ bridge type translation
│   ├── task_manager.py      # Task lifecycle, state machine, persistence
│   ├── middleware.py         # BridgeMiddleware chain
│   └── stream.py            # Streaming adapter (SSE → MCP streaming)
├── adapters/                # Protocol-specific adapters
│   ├── a2a/                 # A2A adapter (current A2ABridge, A2AClientFactory)
│   │   ├── adapter.py       # A2AAgentAdapter implementing AgentAdapter protocol
│   │   ├── card_parser.py   # AgentCard → AgentInfo + tool/resource/prompt mapping
│   │   ├── client.py        # A2A SDK client wrapper
│   │   └── push.py          # Push notification handler
│   └── base.py              # Reference adapter implementation
├── server.py                # Main entry point — FastMCP server factory
├── tools.py                 # Core MCP tools (agent, agents, task, inspect)
└── contrib/                 # Community adapters, transforms, middleware
```

### Type safety improvements

Every public interface should use generic types. The adapter protocol should be parameterized:

```python
from typing import TypeVar, Generic

TConfig = TypeVar("TConfig", bound=BaseModel)
TMessage = TypeVar("TMessage")

class AgentAdapter(Protocol[TConfig]):
    config: TConfig
    async def discover_agents(self) -> list[AgentInfo]: ...
```

Use `Annotated` types for dependency injection alignment with FastMCP 3.0's `Depends()`:

```python
from fastmcp.dependencies import Depends
from typing import Annotated

AgentRouter = Annotated[BaseRouter, Depends(get_router)]
```

### Better error handling with typed exceptions

Define a hierarchy of bridge-specific exceptions that map cleanly to both MCP and A2A error codes:

```python
class AgentiqueError(Exception):
    mcp_code: int = -32603  # Internal error
    a2a_code: int | None = None

class AgentNotFoundError(AgentiqueError):
    mcp_code = -32602  # Invalid params
    a2a_code = -32001  # TaskNotFound

class AgentUnavailableError(AgentiqueError):
    mcp_code = -32603
    
class InputRequiredError(AgentiqueError):
    """Maps to A2A input-required state → MCP elicitation"""
    pass
```

### Customizable tool creation from agent cards

The current `A2ATranslator` maps agent cards to MCP tools. This should be a pluggable `ToolMapper` protocol:

```python
class ToolMapper(Protocol):
    def map_agent_to_tools(self, agent: AgentInfo) -> list[ToolDefinition]: ...
    def map_agent_to_resources(self, agent: AgentInfo) -> list[ResourceDefinition]: ...
    def map_agent_to_prompts(self, agent: AgentInfo) -> list[PromptDefinition]: ...
```

A default implementation creates one tool per agent (current behavior). Users can provide mappers that create one tool per skill, flatten sub-agent hierarchies, or apply custom naming conventions. This is registered via the configuration or dependency injection.

### Pluggable routing strategies

The router should be a protocol with swappable implementations:

```python
class AgentRouter(Protocol):
    async def select_agent(self, message: str, 
                          available_agents: list[AgentInfo],
                          context: BridgeContext) -> AgentInfo: ...

class KeywordRouter:
    """Routes based on keyword matching against agent skills."""
    
class LLMRouter:
    """Uses ctx.sample() to let the MCP client's LLM choose an agent."""
    
class DirectRouter:
    """Routes to a specifically named agent (for single-agent bridges)."""
```

The `LLMRouter` is particularly powerful — it leverages FastMCP 3.0's `ctx.sample()` to ask the MCP client's own LLM which agent is best suited for a request, returning structured output via `result_type=AgentSelection`.

### Testing patterns

Agentique should provide test utilities:

```python
# Test fixtures
from agentique.testing import MockAdapter, InMemoryBridge, mock_agent_card

# Property: adapter protocol compliance
def test_my_adapter_satisfies_protocol():
    adapter = MyCustomAdapter(config)
    assert isinstance(adapter, AgentAdapter)  # runtime_checkable Protocol

# Integration test with FastMCP's in-process client
async def test_bridge_end_to_end():
    server = create_agentique_server(adapters=[MockAdapter()])
    async with server.test_client() as client:
        tools = await client.list_tools()
        assert len(tools) > 0
```

---

## Lessons from competing frameworks

Analysis of LangChain, CrewAI, AutoGen, Semantic Kernel, and PydanticAI reveals consistent patterns that agentique should adopt and specific anti-patterns to avoid.

**LangChain's Runnable interface** demonstrates the power of a universal composable unit — every component implements `.invoke()`, `.ainvoke()`, `.stream()`, enabling LCEL pipe composition (`chain = prompt | llm | parser`). Agentique should ensure its `AgentAdapter` protocol supports both synchronous and streaming variants, and that adapters compose naturally.

**PydanticAI's toolset abstraction is the closest model.** Its `AbstractToolset` with `get_tools()` and `call_tool()` mirrors exactly what agentique's adapter layer needs. PydanticAI's `Agent[DepsT, OutputT]` generic typing ensures full type safety through the entire pipeline. Its `MCPServer` and `FastMCPToolset` classes demonstrate clean MCP integration. Agentique should study PydanticAI's `FastA2A` for design inspiration on A2A server exposure.

**Semantic Kernel's Kernel-as-DI-container** pattern is relevant. The Kernel manages AI services, plugins, and configuration centrally while remaining model-agnostic. Agentique's server factory should similarly serve as the composition root where adapters, middleware, transforms, and configuration converge.

**What to avoid:** CrewAI's role-based abstractions are too opinionated for a bridge library — agentique should not impose workflow patterns. AutoGen's conversation-as-workflow model, while powerful for multi-agent collaboration, is too specific. LangChain's frequent API changes and heavy abstraction layers have caused developer friction — agentique should keep its core interface surface small and stable.

**The callback system pattern** appears in every framework: LangChain's `BaseCallbackHandler`, ADK's six callback types, FastMCP's middleware hooks. Agentique should implement **at most 8-10 well-defined lifecycle hooks** rather than proliferating callbacks:

- `on_server_start` / `on_server_stop`
- `on_agent_discovered` / `on_agent_lost`
- `before_tool_call` / `after_tool_call`
- `before_message_send` / `after_message_receive`
- `on_task_state_change`
- `on_error`

---

## Aligning with MCP and A2A protocol evolution

Both MCP and A2A are evolving rapidly, and agentique must track their trajectories. **MCP's November 2025 spec** added experimental Tasks support — asynchronous, long-running operations with states (`working`, `input_required`, `completed`, `failed`, `cancelled`) and `notifications/tasks/status`. This maps almost perfectly to A2A's task lifecycle, meaning agentique's task bridge becomes simpler as MCP natively supports the concept.

**MCP's structured tool output** (`outputSchema` + `structuredContent`) should be used when translating A2A agent responses. Instead of returning plain text, agentique should define output schemas for its tools — the `agent` tool could return structured JSON with `task_id`, `state`, `response_text`, and `artifacts`.

**MCP's extensions framework** (2025-11-25) enables optional, independently versioned protocol extensions. Agentique could define an `agentique://` extension that carries metadata like `agent_protocol`, `original_task_id`, and `agent_card_url` through MCP interactions.

**A2A's donation to the Linux Foundation** signals long-term stability. The protocol's three-binding approach (JSON-RPC, gRPC, REST) means agentique's adapter should support transport selection. The `ClientConfig(ordered_transports=["JSONRPC", "gRPC"])` pattern from the SDK enables automatic transport negotiation.

**Google ADK's dual-direction integration** is notable: `McpToolset` consumes MCP servers as ADK tools, while `to_a2a()` exposes ADK agents as A2A servers. Agentique occupies the complementary position — consuming A2A agents as MCP tools. Together, these create a full bidirectional bridge: `MCP Client → agentique → A2A → ADK Agent → McpToolset → other MCP servers`.

---

## Priority implementation roadmap

Based on impact and effort analysis, the following sequence maximizes value:

**Phase 1 — Foundation refactor (high impact, moderate effort):**
Restructure into the layered package architecture. Extract the `AgentAdapter` protocol from `A2ABridge`. Implement `AgentiqueConfig` with Pydantic Settings. Add the `AsyncEventEmitter` for lifecycle hooks. Switch to FastMCP 3.0's `Middleware` for logging and auth forwarding. Add `ToolMapper` protocol with default implementation.

**Phase 2 — FastMCP 3.0 deep integration (high impact, low effort):**
Use `Transform` classes for namespace isolation and tool renaming. Implement `Visibility` for session-level agent control. Use `Depends()` for dependency injection. Add `Progress` reporting to background tasks. Configure storage backends for task persistence. Add OpenTelemetry span attributes.

**Phase 3 — A2A protocol completeness (medium impact, moderate effort):**
Implement push notification support. Add task resubscription for resilient streaming. Map A2A error codes to MCP errors. Support extended agent cards. Add context ID ↔ MCP session ID mapping. Expose A2A extensions mechanism.

**Phase 4 — Extensibility and ecosystem (high long-term impact, higher effort):**
Implement pluggable routing strategies (keyword, LLM-based, direct). Add entry point discovery for third-party adapters. Create a second adapter (OpenAI Agents API or generic HTTP) to validate the protocol abstraction. Publish `agentique-core` as a separate package. Add comprehensive test utilities.

## Conclusion

Agentique sits at a uniquely valuable intersection point. MCP has become the dominant client-to-tool protocol with **97 million+ monthly SDK downloads**, A2A is the emerging agent-to-agent standard backed by **150+ organizations**, and FastMCP 3.0 provides a sophisticated server framework with providers, transforms, and middleware. The key architectural insight is that agentique should not be an A2A-specific bridge but a **generic agent protocol gateway** — one where A2A is the first adapter in a pluggable ecosystem.

The most important technical decisions are: adopt Protocol classes over ABCs for all interfaces (enabling structural subtyping without forced inheritance), use FastMCP 3.0's Transform and Middleware systems instead of reimplementing cross-cutting concerns, implement the full A2A task lifecycle including push notifications and resubscription, and provide a `ToolMapper` protocol so users control how agent capabilities become MCP primitives. The combination of Pydantic Settings for configuration, entry points for adapter discovery, and an async event emitter for lifecycle hooks creates the unopinionated foundation that lets agentique serve diverse use cases — from single-agent wrappers to enterprise multi-protocol agent meshes — without imposing architectural opinions on its users.