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

**Dependency injection via `Depends()`** — `agentique.bridge.dependencies` module provides a registry-based DI system compatible with FastMCP 3.0's `Depends()`:
- `configure()` populates the registry during `create_server()` construction
- `clear()` resets the registry for testing isolation
- Factory functions: `get_router()`, `get_adapter()`, `get_task_manager()`, `get_config()`, `get_emitter()`, `get_context_manager()`
- All server tools use `Depends()` for clean parameter injection instead of closure-captured variables
- Each factory raises `RuntimeError` with a descriptive message when not configured

**Session-scoped dependency injection** — Per-session overrides stored via FastMCP's `ctx.set_state()` / `ctx.get_state()` mechanism:
- `set_session_override(ctx, key, value)` — stores a per-session dependency override
- `get_session_override(ctx, key)` — retrieves a session override (or None)
- `clear_session_overrides(ctx)` — removes all overrides for the current session
- Session-aware factories: `get_session_router(ctx)`, `get_session_adapter(ctx)`, `get_session_config(ctx)` — check session overrides first, fall back to server-scope registry
- Enables multi-tenant configurations where different MCP sessions get different adapters, routers, or configs
- Uses `_agentique_dep_` key prefix for session state isolation

**Composition via `mount()`** — `mount_bridge()` helper enables multi-bridge architectures:
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

**Visibility transform integration** — `AgentVisibility` wraps FastMCP's `Visibility` transform for per-session agent management:
- `apply(mcp, agents=["a", "b"])` — applies the Visibility transform to the server
- `enable_agent(ctx, name)` / `disable_agent(ctx, name)` — session-level enable/disable via `ctx.enable_components()` / `ctx.disable_components()`
- `reset_visibility(ctx)` — resets to server defaults for the current session
- `get_visible_agents(ctx)` — queries current session visibility state
- `register_tools(mcp)` — adds `enable_agent` and `disable_agent` MCP tools for client-driven visibility control
- Graceful fallback when FastMCP Visibility transform is not available

**Elicitation for input-required states** — When an A2A agent returns `input-required` task state during streaming, the `agent` tool uses `ctx.elicit()` to prompt the MCP client for additional input. The user's response is sent back to the agent, continuing the conversation loop. Handles `AcceptedElicitation`, `DeclinedElicitation`, and `CancelledElicitation` gracefully.

**Structured sampling in LLMRouter** — `LLMRouter.aselect()` now uses `ctx.sample()` with `result_type` (list of agent names) for structured agent selection. Falls back gracefully:
1. First attempts structured sampling with `result_type=agent_names`
2. If `TypeError` (structured not supported), falls back to plain-text sampling
3. If all sampling fails, falls back to `KeywordRouter`

**Structured content via ToolResult** — The `agents`, `task`, and `inspect` tools now return `ToolResult` with both `content` (JSON string for backward compatibility) and `structured_content` (dict for MCP's `outputSchema`/`structuredContent` spec). The `agent_background` tool also returns structured results with `task_id`, `agent`, `state`, and `event_count`.

**Structured tool output models** — Pydantic output models in `agentique.bridge.output_models` map to MCP's `outputSchema` and `structuredContent`:
- `AgentMessageOutput` — structured response from the `agent` tool (agent, text, task_id, state, event_count)
- `AgentListOutput` / `AgentSummary` — structured response from the `agents` tool
- `TaskStatusOutput` — structured response from the `task` tool
- `AgentInspectOutput` / `SubAgentSummary` — structured response from the `inspect` tool
- `HealthCheckOutput` / `AgentHealthOutput` — structured health check results
- `WebhookNotificationOutput` — structured webhook receipt confirmation
- `ErrorOutput` — structured error responses with error code and details
- All models expose `json_schema()` classmethod for MCP `outputSchema` registration
- Compatible with `ToolResult(content=model.model_dump_json(), structured_content=model.model_dump())`

**TaskConfig API** — Background task tool (`agent_background`) uses `TaskConfig(mode="optional")` from `fastmcp.server.tasks.config` for fine-grained control. Falls back to `task=True` with `Progress` dependency if `TaskConfig` is unavailable.

**OpenTelemetry integration** — New `agentique.core.telemetry` module:
- `get_tracer()` returns an agentique OTel tracer (no-op if SDK not installed)
- `trace_agent_call()` context manager creates spans with `agentique.agent_name`, `agentique.protocol`, `agentique.task_id`, and `agentique.task_state` attributes
- `set_span_attribute()` adds attributes to the current active span
- `AgentiqueMiddleware` injects `agentique.tool_name` and `agentique.protocol` on tool calls
- Zero overhead when OpenTelemetry SDK is not installed (uses no-op fallbacks)

**Storage backend abstraction** — `TaskStore` protocol in `agentique.bridge.storage`:
- `InMemoryTaskStore` — default in-memory implementation
- `TaskManager` accepts a pluggable `store` parameter
- `TaskStore` protocol defines `save()`, `load()`, `delete()`, `list_ids()` async methods
- `create_server()` accepts `task_store` parameter

**Persistent TaskStore implementations** — `agentique.bridge.persistent_stores` provides production-ready backends:
- `RedisTaskStore` — Redis-backed store using `redis.asyncio`:
  - Configurable key prefix (`agentique:task:` default) and TTL (86400s default)
  - Accepts pre-configured Redis client or connection URL
  - Full CRUD: `save()`, `load()`, `delete()`, `list_ids()` via `SCAN`
  - Graceful error handling for deserialization failures
- `DynamoDBTaskStore` — DynamoDB-backed store using `aiobotocore`:
  - Configurable table name, region, and custom endpoint URL (for local development)
  - Partition key: `task_id` (String), with `data` (JSON) and `state` columns
  - Paginated `list_ids()` via DynamoDB scan
  - Accepts pre-configured `aiobotocore.AioSession`
- Both implementations use shared `_serialize_tracker()` / `_deserialize_tracker()` for TaskTracker JSON serialization with full event, hierarchy, and metadata round-trip support

**Webhook receiver for push notifications** — `agentique.bridge.webhook` accepts push notifications from agents:
- `WebhookReceiver` receives, stores, and dispatches push notification events
- Parses both A2A-style nested notifications (with `result.status.state`) and simple flat payloads
- `PushNotification` dataclass with id, agent_id, task_id, kind, state, text, data, timestamp
- Automatic task state updates via TaskManager when notification includes task_id and state
- Emits events: `webhook.received`, `task.state_changed`, `message.received`
- Subscription management: `subscribe(agent_id, task_id)` / `unsubscribe()`
- Filtered retrieval: `get_notifications(agent_id=, task_id=, limit=)`
- `register(mcp)` adds `webhook_notify` tool and `webhook://notifications` resource to the server
- Configurable history limit (default 1000 notifications)

**Adapter health monitoring** — `agentique.bridge.health` provides periodic health checks:
- `HealthMonitor` probes agents at configurable intervals with configurable failure threshold
- `AgentHealth` dataclass tracks: healthy, last_check, last_success, consecutive_failures, last_error, latency_ms
- Health probe uses `get_agent_card()` (lightweight) or falls back to `discover_agents()`
- Events emitted: `adapter.healthy` (recovery), `adapter.unhealthy` (failure), `adapter.health_check` (every check)
- Automatic router integration: removes unhealthy agents, re-registers recovered agents
- `check_agent(id)` for single agent, `check_all()` for all known agents
- `get_healthy_agents()` / `get_unhealthy_agents()` for status queries
- `start()` / `stop()` for background monitoring loop
- `register_tools(mcp)` adds `agent_health` tool for client-driven health checks

**MCP proxy adapter** — `agentique.adapters.mcp` wraps remote MCP servers as agent-like adapters:
- `MCPProxyAdapter` implements the `AgentAdapter` protocol via FastMCP's `create_proxy()`
- Each registered agent maps to a remote MCP server endpoint
- Lazy proxy creation per agent via `_get_proxy()` with caching
- `send_message()` discovers and calls tools on the remote MCP server via `proxy.test_client()`
- `stream_message()` yields single event (MCP tool calls are request/response)
- `get_agent_card()` builds synthetic agent cards from remote server's tool list
- `MCPAdapterFactory` registered as `@register_adapter("mcp")` for auto-discovery
- Graceful error handling for connection failures and missing tools

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
- `test_mcp_adapter.py` — MCPProxyAdapter discover/send/stream/card/close, unknown agent errors, no-tools response, _extract_text helper
- `test_webhook.py` — Notification parsing (simple, A2A-style, edge cases), WebhookReceiver receive/emit/task-update/filter/subscribe/clear/max-history
- `test_persistent_stores.py` — Serialization round-trips (basic, with events, hierarchy, unknown state, metadata), RedisTaskStore CRUD/list/TTL with mock client, DynamoDBTaskStore CRUD/list/overwrite with mock client
- `test_visibility.py` — AgentVisibility enable/disable/reset/get-visible/register-tools, fallback without support, apply without Visibility transform
- `test_health.py` — AgentHealth defaults/to_dict, HealthMonitor check-healthy/unhealthy/events/check-all/discover/auto-remove/auto-re-register/start-stop/always-emits
- `test_output_models.py` — All output model construction, model_dump, model_dump_json, json_schema for AgentMessageOutput, AgentListOutput, TaskStatusOutput, AgentInspectOutput, HealthCheckOutput, WebhookNotificationOutput, ErrorOutput

**Total: 245 tests passing (1 skipped)**

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

**✦ Extension-to-OTel bridge** — Since both A2A extensions and OpenTelemetry are now supported, a built-in "tracing extension" could automatically propagate OTel trace context as an A2A extension (`urn:agentique:ext:otel`), creating end-to-end distributed traces across MCP → agentique → A2A agents.

**✦ Client interceptor middleware** — The a2a-sdk supports `ClientCallInterceptor` middleware on the client side. Agentique could expose this to users for request/response interception at the transport level (e.g., adding auth headers, logging raw A2A payloads).

**✦ Server-side output_schema registration** — Wire the Pydantic output models to FastMCP's `@mcp.tool(output_schema=...)` parameter so that MCP clients receive JSON Schema definitions for each tool's structured output. Requires FastMCP to fully support `output_schema` on tool registration.

**✦ Webhook endpoint as HTTP server** — The current webhook receiver works as a FastMCP tool/resource. A standalone HTTP endpoint (e.g., via Starlette/ASGI) would allow agents to POST push notifications directly without going through MCP, enabling standard webhook patterns.

**✦ Health monitor dashboard resource** — Expose health status as a FastMCP resource (`health://agents`) with live-updating JSON for MCP clients to display health dashboards.

**✦ Task store migration tooling** — Utilities for migrating task data between store backends (e.g., InMemory → Redis, Redis → DynamoDB) for deployment transitions.

**✦ Visibility rules engine** — Extend the Visibility integration with declarative rules (e.g., "enable agent-X if user has role admin") that can be evaluated per-session based on context metadata.

**✦ Adapter connection pooling** — The MCP proxy adapter creates one `FastMCPProxy` per agent. For high-throughput scenarios, connection pooling and multiplexing across proxy instances would improve efficiency.

**✦ Multi-store task manager** — A `TaskManager` that writes to multiple stores simultaneously (e.g., InMemory for fast reads + Redis for persistence) with configurable write-through/write-behind strategies.

**✦ Webhook authentication** — Add HMAC signature verification to the webhook receiver so that only authorized agents can send push notifications. Support multiple signing secrets per agent.

---

## File Inventory

### New Files (This Iteration)
| File | Description |
|------|-------------|
| `src/agentique/adapters/mcp/__init__.py` | MCP proxy adapter package |
| `src/agentique/adapters/mcp/adapter.py` | MCPProxyAdapter — wraps remote MCP servers as agents via `create_proxy()` |
| `src/agentique/bridge/webhook.py` | WebhookReceiver — push notification ingestion, parsing, event dispatch |
| `src/agentique/bridge/persistent_stores.py` | RedisTaskStore and DynamoDBTaskStore implementations |
| `src/agentique/bridge/visibility.py` | AgentVisibility — per-session agent enable/disable via Visibility transform |
| `src/agentique/bridge/health.py` | HealthMonitor — periodic adapter health checks with auto-remove/re-register |
| `src/agentique/bridge/output_models.py` | Pydantic output models for structured MCP tool responses |
| `tests/test_mcp_adapter.py` | Tests for MCPProxyAdapter (discover, send, stream, card, close, error handling) |
| `tests/test_webhook.py` | Tests for WebhookReceiver and notification parsing |
| `tests/test_persistent_stores.py` | Tests for Redis/DynamoDB stores with mock clients, serialization round-trips |
| `tests/test_visibility.py` | Tests for AgentVisibility enable/disable/reset/tools |
| `tests/test_health.py` | Tests for HealthMonitor checks, events, auto-remove/re-register, start/stop |
| `tests/test_output_models.py` | Tests for all Pydantic output models |

### New Files (Previous Iteration — DI, Composition, Extensions, Transport)
| File | Description |
|------|-------------|
| `src/agentique/bridge/dependencies.py` | Dependency injection registry — configure/clear/get_* factories + session-scoped DI |
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
| `src/agentique/__init__.py` | Exports session-scoped DI, health monitor, output models, persistent stores, visibility, webhooks |
| `src/agentique/bridge/__init__.py` | Exports all new bridge modules (health, output_models, persistent_stores, visibility, webhook) |
| `src/agentique/adapters/__init__.py` | Added MCPProxyAdapter import and MCPAdapterFactory registration |
| `src/agentique/bridge/dependencies.py` | Added session-scoped DI (set/get/clear session overrides, session-aware factories) |

### Modified Files (Previous Iteration)
| File | Changes |
|------|---------|
| `src/agentique/server.py` | Refactored all tools to use `Depends()` for DI; added `mount_bridge()` composition helper |
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
