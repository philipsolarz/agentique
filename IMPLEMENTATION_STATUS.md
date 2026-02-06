# Agentique Implementation Status Report

## 1. Completed

This section covers all features that have been implemented in the codebase, organized by the original research report's phasing.

### Phase 1 — Foundation (Complete)

**Layered package structure** — The library follows the recommended three-layer architecture: `core/` (zero-dependency protocols, types, config), `bridge/` (protocol-agnostic routing and state), `adapters/` (pluggable backends). This mirrors the LangChain `core` / `implementations` / `integrations` split recommended in the report.

**Protocol classes via `typing.Protocol`** — All public interfaces use structural subtyping rather than abstract base classes:
- `AgentAdapter` — the contract every backend must satisfy (`discover_agents`, `send_message`, `stream_message`, `close`)
- `BridgeMiddleware` — chain-of-responsibility request processing
- `ToolMapper` — controls how agent capabilities become MCP primitives
- `AdapterFactory` — entry-point based adapter creation
- `RoutingStrategy` — pluggable agent selection

All are decorated with `@runtime_checkable` for isinstance checks without inheritance.

**Pydantic Settings configuration** — `AgentiqueConfig` uses `pydantic_settings.BaseSettings` with `AGENTIQUE_` env prefix, `.env` file support, and type validation. Covers transport, host/port, feature flags, performance tuning, agent routing, A2A extensions, and transport selection.

**AsyncEventEmitter for lifecycle hooks** — Supports both sync and async handlers with `asyncio.gather()`. Emits events at: `server.start/stop`, `agent.discovered/lost`, `tool.called/completed/failed`, `task.created/state_changed/completed`, `message.sent/received`, `stream.chunk`, `error`.

**Typed exception hierarchy** — Every exception carries `mcp_code` (JSON-RPC) and optional `a2a_code`:
- `AgentNotFoundError` → MCP -32602, A2A -32001
- `AgentUnavailableError` → MCP -32603
- `TaskNotFoundError` → MCP -32602, A2A -32001
- `InputRequiredError`, `TranslationError`, `AdapterError`

**Default ToolMapper implementations** — Three built-in mappers:
- `DefaultToolMapper` — one MCP tool per agent (original behaviour)
- `PerSkillToolMapper` — one MCP tool per agent skill
- `FlatHierarchyToolMapper` — one MCP tool per sub-agent from card metadata

**Middleware chain** — `MiddlewareChain` implements chain-of-responsibility with fluent `.add()`. Built-in middleware:
- `LoggingMiddleware` — request/response timing
- `ErrorMappingMiddleware` — maps A2A error codes (-32001, -32002, -32003) and HTTP status codes to typed exceptions
- `RateLimitMiddleware` — sliding-window per-agent rate limiting
- `MetricsMiddleware` — timing and counter collection

**Pluggable routing with five strategies:**
- `KeywordRouter` — keyword matching against agent skills (default)
- `WeightedKeywordRouter` — configurable skill weights
- `DirectRouter` — always routes to a named agent
- `LLMRouter` — uses `ctx.sample()` for intelligent routing, falls back to keywords
- `AgentRouter` — main registry with `resolve()` (sync) and `aresolve()` (async/LLM-capable)

**Adapter registry with entry-point discovery** — `@register_adapter` decorator for built-in adapters. `discover_adapters()` scans `agentique.adapters` entry points for third-party packages. `create_adapter(protocol, agents)` factory function.

**Context ID ↔ MCP session mapping** — `ContextManager` maintains bidirectional mapping between MCP sessions and A2A context IDs. `ContextMapping` dataclass tracks session→contexts, context→session, and context→task_ids. Validates context/task consistency per A2A spec.

### Phase 1 — A2A Adapter (Complete)

**A2AAgentAdapter** — Full implementation with `send_message`, `stream_message`, `get_agent_card`, `get_agent_extensions`, and `close`. Handles both streaming and non-streaming A2A communication. Translates A2A SDK events into `AgentEvent` types. Supports extension propagation in outgoing messages.

**A2AClientPool** — Creates, caches, and manages A2A SDK clients by base URL with configurable timeout, client config, extensions, transport selection, and gRPC channel factory.

**A2ACardParser** — Extracts MCP tool/resource/prompt definitions from agent card extensions (`urn:mcp:extension:tools`). Builds sub-agent hierarchies from card metadata.

### Phase 1 — FastMCP Provider (Complete)

**AgentProvider** — FastMCP 3.0 Provider that dynamically sources MCP components from agents:
- Lazy agent card fetching with TTL-based caching
- Dynamic tool creation from agent card MCP extensions
- One base tool per agent for direct access
- Proxy tools for card-declared MCP tools
- Agent catalog resource
- Routing prompt generation
- Proper lifecycle management via `lifespan()`

### Phase 1 — Server & Tools (Complete)

**FastMCP server factory** with five core tools:
- `agent` — send message and stream response (with LLM-capable routing via `aresolve`)
- `agents` — list available agents
- `task` — query task state
- `inspect` — view agent hierarchy
- `agent_background` — background task with `Progress` reporting

**Task manager** — In-memory task tracking with state machine, conversation continuity, hierarchy tracking.

**Streaming** — Full streaming support via `stream_message` with `StreamChunk` types.

**Sub-agent visibility** — `AgentHierarchy` and `SubAgentInfo` track nested agent structures with branch path resolution.

**Background task support** — Uses FastMCP's `task=True` and `Progress` dependency injection for progress tracking.

### Phase 1 — Second Adapter (Complete)

**Generic HTTP adapter** (`adapters/http/`) — Validates the protocol abstraction with a second backend:
- `HttpAgentAdapter` connects to agents with simple JSON-over-HTTP APIs
- Supports `POST /message` (non-streaming) and `POST /stream` (SSE streaming)
- Health check via `GET /health`
- Automatic fallback from streaming to non-streaming
- Proper error mapping for HTTP status codes and connection errors

### Phase 1 — Testing Utilities (Complete)

**`agentique.testing` module** with:
- `MockAdapter` — in-memory adapter with pre-configured responses and call recording
- `MockStreamingAdapter` — streams word-by-word with progress
- `RecordingMiddleware` — records all requests/responses passing through
- `mock_agent_info()` — factory for test `AgentInfo` instances
- `mock_agent_card()` — factory for test agent card dictionaries

### Phase 2 — FastMCP 3.0 Deep Integration (Complete)

**Dependency injection via `Depends()`** — New `agentique.bridge.dependencies` module provides a registry-based DI system compatible with FastMCP 3.0's `Depends()`:
- `configure()` populates the registry during `create_server()` construction
- `clear()` resets the registry for testing isolation
- Factory functions: `get_router()`, `get_adapter()`, `get_task_manager()`, `get_config()`, `get_emitter()`, `get_context_manager()`
- All server tools use `Depends()` for clean parameter injection instead of closure-captured variables
- Each factory raises `RuntimeError` with a descriptive message when not configured

**Composition via `mount()`** — New `mount_bridge()` helper enables multi-bridge architectures:
- Creates a child `FastMCP` server for each adapter bridge
- Mounts it under the parent server with namespace isolation via `FastMCP.mount()`
- Separate A2A, HTTP, and local bridges compose under a single MCP endpoint
- Each mounted bridge gets independent routing, task management, and middleware
- Example: `mount_bridge(main, agents=a2a_agents, adapter=a2a, namespace="a2a")`

**FastMCP Middleware integration** — `AgentiqueMiddleware` extends FastMCP 3.0's native `Middleware` class with `on_call_tool` and `on_list_tools` hooks. Automatically added by `create_server()`. Provides:
- Event emission on tool calls (`tool.called` / `tool.completed` / `tool.failed`) via `AsyncEventEmitter`
- Request timing and logging at the FastMCP server level
- OpenTelemetry span attribute injection when tracing is active

**Transform support** — `create_server()` accepts `namespace` and `transforms` parameters:
- `namespace="prefix"` applies FastMCP's `Namespace` transform for tool name isolation
- `transforms=[...]` applies arbitrary FastMCP Transform instances (e.g., `Visibility` for session-level agent control, `ToolTransform` for renaming)
- Works with `Visibility`, `Namespace`, `ResourcesAsTools`, `PromptsAsTools`, and custom transforms

**Elicitation for input-required states** — When an A2A agent returns `input-required` task state during streaming, the `agent` tool uses `ctx.elicit()` to prompt the MCP client for additional input. The user's response is sent back to the agent, continuing the conversation loop. Handles `AcceptedElicitation`, `DeclinedElicitation`, and `CancelledElicitation` gracefully.

**Structured sampling in LLMRouter** — `LLMRouter.aselect()` now uses `ctx.sample()` with `result_type` (list of agent names) for structured agent selection. Falls back gracefully:
1. First attempts structured sampling with `result_type=agent_names`
2. If `TypeError` (structured not supported), falls back to plain-text sampling
3. If all sampling fails, falls back to `KeywordRouter`

**Structured content via ToolResult** — The `agents`, `task`, and `inspect` tools now return `ToolResult` with both `content` (JSON string for backward compatibility) and `structured_content` (dict for MCP's `outputSchema`/`structuredContent` spec). The `agent_background` tool also returns structured results with `task_id`, `agent`, `state`, and `event_count`.

**TaskConfig API** — Background task tool (`agent_background`) uses `TaskConfig(mode="optional")` from `fastmcp.server.tasks.config` for fine-grained control. Falls back to `task=True` with `Progress` dependency if `TaskConfig` is unavailable.

**OpenTelemetry integration** — New `agentique.core.telemetry` module:
- `get_tracer()` returns an agentique OTel tracer (no-op if SDK not installed)
- `trace_agent_call()` context manager creates spans with `agentique.agent_name`, `agentique.protocol`, `agentique.task_id`, and `agentique.task_state` attributes
- `set_span_attribute()` adds attributes to the current active span
- `AgentiqueMiddleware` injects `agentique.tool_name` and `agentique.protocol` on tool calls
- Zero overhead when OpenTelemetry SDK is not installed (uses no-op fallbacks)

**Storage backend abstraction** — New `TaskStore` protocol in `agentique.bridge.storage`:
- `InMemoryTaskStore` — default in-memory implementation
- `TaskManager` accepts a pluggable `store` parameter
- `TaskStore` protocol defines `save()`, `load()`, `delete()`, `list_ids()` async methods
- Production deployments can implement Redis, DynamoDB, filesystem, or other backends
- `create_server()` accepts `task_store` parameter

**Lifespan composition support** — FastMCP's lifespan pipe operator (`lifespan_a | lifespan_b`) is available for composing startup/shutdown logic across multiple adapter connections. The `AgentProvider` uses FastMCP's `Provider.lifespan()` for adapter lifecycle management. The server factory is structured to support composition via the `providers` parameter.

### Phase 3 — A2A Protocol Completeness (Complete)

**Push notification support** — `A2AAgentAdapter.configure_push_notifications()` creates `PushNotificationConfig` instances and configures them on A2A agents via the SDK client:
- Accepts per-task callback URLs or falls back to adapter-level `push_notification_url`
- Generates unique notification IDs and tokens
- Gracefully handles agents that don't support push notifications
- `create_server()` can be configured with `push_notification_url` via the adapter

**Task resubscription** — `A2AAgentAdapter.resubscribe()` reconnects to active task streams after SSE disconnection:
- Uses A2A's `tasks/resubscribe` method via `TaskIdParams`
- Streaming automatically retries on `ConnectionError`/`OSError` up to `max_resubscribe_attempts` (default: 3)
- Configurable via `retry_on_disconnect` flag on the adapter
- Translates resubscribed events through the same `_translate_event()` pipeline

**Extended agent card support** — `A2AAgentAdapter.get_agent_card()` accepts an `authenticated` parameter:
- Checks the `supports_authenticated_extended_card` flag on the public card
- When supported, fetches the authenticated extended card via `get_authenticated_extended_card()`
- Falls back to the public card when extended card is not available or not supported
- Exposes additional skills/tools from authenticated cards

**Auth-required and rejected state handling** — The `agent` tool in `server.py` handles A2A task states:
- `input-required` → Uses `ctx.elicit()` to prompt MCP client for input, sends response back to agent
- `auth-required` → Sends a warning via `ctx.warning()` informing the client to provide credentials
- `rejected` → Tracked in `TaskState.rejected` (terminal state), properly handled by `TaskTracker`

**A2A extensions mechanism** — Full extension propagation support:
- `AgentiqueConfig.extensions` — list of A2A extension URIs the client advertises support for (e.g., `["urn:a2a:ext:tracing"]`)
- `A2AClientPool` accepts `extensions` parameter, passes it to `ClientConfig.extensions` for SDK-level header propagation via `X-A2A-Extensions`
- `A2AAgentAdapter` accepts `extensions` parameter, attaches extension URIs to outgoing `Message` objects
- `A2AAgentAdapter.get_agent_extensions()` discovers extensions supported by an agent by reading the agent card's `capabilities.extensions` field
- Response metadata extraction captures `extensions` from incoming A2A messages/events
- Supports the full a2a-sdk extension flow: `AgentExtension` type, `find_extension_by_uri()`, `update_extension_header()`, and `HTTP_EXTENSION_HEADER`

**gRPC transport support** — Transport selection is fully configurable:
- `AgentiqueConfig.supported_transports` — ordered list of preferred transports (e.g., `["JSONRPC", "GRPC"]`)
- `A2AClientPool` accepts `supported_transports` and `grpc_channel_factory` parameters
- Passes `supported_transports` to `ClientConfig.supported_transports` for SDK transport negotiation
- Passes `grpc_channel_factory` to `ClientConfig.grpc_channel_factory` for gRPC channel creation
- Compatible with the a2a-sdk's `TransportProtocol` enum (`JSONRPC`, `GRPC`, `HTTP+JSON`)
- Transport negotiation follows the SDK's preference logic: server-preferred by default, client-preferred with `use_client_preference=True`

### Test Coverage

- `test_core.py` — TaskState, TaskTracker, AgentInfo, BridgeContext (including `.replace()`), AgentEvent, StreamChunk, AgentHierarchy, ContextMapping, Router, Events
- `test_middleware.py` — MiddlewareChain ordering, LoggingMiddleware, ErrorMappingMiddleware, RateLimitMiddleware, MetricsMiddleware
- `test_tool_mapper.py` — DefaultToolMapper, PerSkillToolMapper, FlatHierarchyToolMapper
- `test_context_manager.py` — ContextMapping, ContextManager async operations
- `test_router_extended.py` — WeightedKeywordRouter, LLMRouter (sync/async/fuzzy/error), aresolve, unregister
- `test_registry.py` — adapter registration, discovery, creation
- `test_testing_utils.py` — MockAdapter, MockStreamingAdapter, RecordingMiddleware, factory helpers
- `test_bridge.py` — end-to-end MCP→A2A integration tests
- `test_fastmcp_middleware.py` — AgentiqueMiddleware event emission, tool name extraction, error handling, list_tools passthrough
- `test_telemetry.py` — NoOpSpan, NoOpTracer, get_tracer caching, trace_agent_call context manager, set_span_attribute no-op safety
- `test_storage.py` — InMemoryTaskStore CRUD, TaskStore protocol compliance, TaskManager with custom store, TaskManager default store
- `test_adapter_extended.py` — Push notification config (custom URL, no URL), task resubscription, extended agent cards (basic, authenticated, no support), retry-on-disconnect
- `test_server_phase2.py` — create_server new params (task_store, namespace, transforms, events), FastMCP middleware auto-registration, ToolResult structured content, TaskConfig API
- `test_router_structured.py` — LLMRouter structured sampling, fallback to text sampling, complete failure fallback, aresolve with structured sampling
- `test_dependencies.py` — DI registry configure/clear, all six factory functions, error messages when not configured, partial registration, overwrite semantics
- `test_composition.py` — mount_bridge child creation, multiple namespace mounting, custom config, child server usability, standalone create_server
- `test_extensions.py` — Extension attachment on messages, no-extensions default, extension extraction from responses, get_agent_extensions (with card, no card, no capabilities)
- `test_transport.py` — Config defaults/custom for extensions and transports, A2AClientPool with extensions/transports/gRPC factory/user config, empty pool close

**Total: 166 tests passing (1 skipped)**

---

## 2. Pending

This section covers features from the research report that are not yet implemented, organized by priority and effort. Items marked with ✦ are new ideas that emerged during implementation.

### Phase 4 — Ecosystem & Community (High Long-Term Impact, Higher Effort)

**Published `agentique-core` package** — Extract the `core/` module as a standalone zero-dependency package so adapter authors don't need to depend on FastMCP or a2a-sdk.

**OpenAI Agents API adapter** — A third adapter targeting OpenAI's Agents API would further validate the protocol abstraction and expand the ecosystem.

**LangChain Runnable adapter** — An adapter wrapping LangChain Runnable endpoints would connect to the largest agent framework ecosystem.

**Comprehensive documentation site** — API reference, tutorials, adapter development guide, and deployment patterns.

**CI/CD pipeline** — Automated testing, type checking (mypy), linting (ruff), and publishing to PyPI.

**MCP Tasks spec alignment** — MCP's November 2025 spec added experimental Tasks support with states matching A2A's lifecycle. As this stabilizes, agentique's task bridge should align with native MCP task primitives.

**MCP extensions framework** — Define an `agentique://` extension carrying metadata like `agent_protocol`, `original_task_id`, and `agent_card_url` through MCP interactions.

**Property-based testing** — Use Hypothesis for property-based tests validating protocol compliance across all adapters.

### ✦ New Ideas and Follow-Up Tasks

**✦ Proxy-based adapter via `create_proxy()`** — FastMCP's `create_proxy()` can proxy to remote MCP servers. An adapter could wrap remote MCP servers as "agents" — useful for MCP-to-MCP bridging where a remote MCP server acts like an agent from agentique's perspective.

**✦ Session-scoped dependency injection** — The current DI module uses module-level state. A future improvement could use FastMCP's `CurrentContext()` to scope dependencies per-session, enabling multi-tenant configurations where different sessions get different adapters or routing strategies.

**✦ Webhook receiver for push notifications** — The current push notification support configures agents to POST to a callback URL, but agentique doesn't yet implement the webhook receiver endpoint. A FastMCP resource or dedicated HTTP endpoint should receive push notifications and translate them into MCP `notifications/tasks/status` events.

**✦ Redis/DynamoDB `TaskStore` implementations** — The `TaskStore` protocol is defined with `InMemoryTaskStore` as the default. Production deployments need persistent backends. Reference implementations for Redis (using `redis.asyncio`) and DynamoDB (using `aiobotocore`) would demonstrate the pattern.

**✦ Extension-to-OTel bridge** — Since both A2A extensions and OpenTelemetry are now supported, a built-in "tracing extension" could automatically propagate OTel trace context as an A2A extension (`urn:agentique:ext:otel`), creating end-to-end distributed traces across MCP → agentique → A2A agents.

**✦ `Visibility` transform integration** — Use FastMCP's `Visibility` transform with `ctx.enable_components()`/`ctx.disable_components()` for dynamic agent availability. Users could "unlock" premium agents mid-session or hide agents based on authentication state.

**✦ Client interceptor middleware** — The a2a-sdk supports `ClientCallInterceptor` middleware on the client side. Agentique could expose this to users for request/response interception at the transport level (e.g., adding auth headers, logging raw A2A payloads).

**✦ Adapter health monitoring** — Implement periodic health checks for registered adapters. Unhealthy adapters could be automatically removed from routing and re-added when they recover. The `AsyncEventEmitter` could emit `adapter.healthy`/`adapter.unhealthy` events.

**✦ Tool output schemas** — MCP's `outputSchema` enables structured validation of tool outputs. Agentique's tools could define Pydantic output models that get automatically mapped to `outputSchema` + `structuredContent` validation.

---

## File Inventory

### New Files (This Iteration)
| File | Description |
|------|-------------|
| `src/agentique/bridge/dependencies.py` | Dependency injection registry — configure/clear/get_* factories for Depends() integration |
| `tests/test_dependencies.py` | Tests for DI registry configure, clear, all factory functions, error handling |
| `tests/test_composition.py` | Tests for mount_bridge multi-bridge composition and namespace isolation |
| `tests/test_extensions.py` | Tests for A2A extension propagation, discovery, and response extraction |
| `tests/test_transport.py` | Tests for gRPC transport config, extensions config, A2AClientPool parameters |

### New Files (Phase 2/3 — Previous Iteration)
| File | Description |
|------|-------------|
| `src/agentique/bridge/fastmcp_middleware.py` | AgentiqueMiddleware — FastMCP 3.0 native middleware with event emission, logging, and OTel support |
| `src/agentique/bridge/storage.py` | TaskStore protocol and InMemoryTaskStore for pluggable task persistence |
| `src/agentique/core/telemetry.py` | OpenTelemetry integration — tracer, span attributes, trace_agent_call context manager |
| `tests/test_fastmcp_middleware.py` | Tests for AgentiqueMiddleware event emission and tool name extraction |
| `tests/test_telemetry.py` | Tests for OTel tracer, NoOp fallbacks, trace context manager |
| `tests/test_storage.py` | Tests for InMemoryTaskStore, TaskStore protocol, TaskManager with custom store |
| `tests/test_adapter_extended.py` | Tests for push notifications, resubscription, extended cards, retry-on-disconnect |
| `tests/test_server_phase2.py` | Tests for create_server new params, transforms, ToolResult, TaskConfig |
| `tests/test_router_structured.py` | Tests for LLMRouter structured sampling with result_type |

### New Files (Phase 1 — Previously Reported)
| File | Description |
|------|-------------|
| `src/agentique/core/registry.py` | Adapter registry with entry-point discovery |
| `src/agentique/core/tool_mapper.py` | DefaultToolMapper, PerSkillToolMapper, FlatHierarchyToolMapper |
| `src/agentique/bridge/middleware.py` | MiddlewareChain and built-in middleware |
| `src/agentique/bridge/context_manager.py` | MCP session ↔ A2A context ID mapping |
| `src/agentique/adapters/http/__init__.py` | Generic HTTP adapter package |
| `src/agentique/adapters/http/adapter.py` | HttpAgentAdapter implementation |
| `src/agentique/testing/__init__.py` | Public test utilities package |
| `src/agentique/testing/mocks.py` | MockAdapter, MockStreamingAdapter, helpers |
| `tests/test_middleware.py` | Middleware chain and built-in middleware tests |
| `tests/test_tool_mapper.py` | ToolMapper implementation tests |
| `tests/test_context_manager.py` | ContextManager and ContextMapping tests |
| `tests/test_router_extended.py` | LLMRouter, WeightedKeywordRouter tests |
| `tests/test_registry.py` | Adapter registry tests |
| `tests/test_testing_utils.py` | Testing utility tests |

### Modified Files (This Iteration)
| File | Changes |
|------|---------|
| `src/agentique/server.py` | Refactored all tools to use `Depends()` for DI; added `mount_bridge()` composition helper |
| `src/agentique/__init__.py` | Exports DI factories, `mount_bridge`, and new bridge dependencies |
| `src/agentique/bridge/__init__.py` | Exports DI factory functions from dependencies module |
| `src/agentique/core/config.py` | Added `extensions` and `supported_transports` fields to `AgentiqueConfig` |
| `src/agentique/adapters/a2a/client.py` | Added `extensions`, `supported_transports`, and `grpc_channel_factory` support |
| `src/agentique/adapters/a2a/adapter.py` | Added `extensions` param, `get_agent_extensions()`, extension propagation in messages, extension extraction from responses |

### Modified Files (Phase 2/3 — Previous Iteration)
| File | Changes |
|------|---------|
| `src/agentique/core/__init__.py` | Exports telemetry module (get_tracer, set_span_attribute, trace_agent_call) |
| `src/agentique/bridge/router.py` | LLMRouter enhanced with structured sampling via `result_type` |
| `src/agentique/bridge/task_manager.py` | Accepts pluggable `TaskStore` backend (default: `InMemoryTaskStore`) |
| `pyproject.toml` | Bumped version to 0.4.0 |

### Modified Files (Phase 1 — Previously Reported)
| File | Changes |
|------|---------|
| `src/agentique/core/types.py` | Added `ContextMapping`, `BridgeContext.replace()` |
| `src/agentique/core/protocols.py` | Added `AdapterFactory` protocol |
| `src/agentique/bridge/router.py` | Added `LLMRouter`, `WeightedKeywordRouter`, `aresolve()`, `unregister()` |
| `src/agentique/adapters/__init__.py` | Registers built-in adapters via `@register_adapter` |
| `tests/test_core.py` | Added `BridgeContext.replace()` and `ContextMapping` tests |
