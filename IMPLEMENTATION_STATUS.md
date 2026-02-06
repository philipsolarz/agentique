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

**Pydantic Settings configuration** — `AgentiqueConfig` uses `pydantic_settings.BaseSettings` with `AGENTIQUE_` env prefix, `.env` file support, and type validation. Covers transport, host/port, feature flags, performance tuning, and agent routing.

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

**A2AAgentAdapter** — Full implementation with `send_message`, `stream_message`, `get_agent_card`, and `close`. Handles both streaming and non-streaming A2A communication. Translates A2A SDK events into `AgentEvent` types.

**A2AClientPool** — Creates, caches, and manages A2A SDK clients by base URL with configurable timeout and client config.

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

### Test Coverage

- `test_core.py` — TaskState, TaskTracker, AgentInfo, BridgeContext (including `.replace()`), AgentEvent, StreamChunk, AgentHierarchy, ContextMapping, Router, Events
- `test_middleware.py` — MiddlewareChain ordering, LoggingMiddleware, ErrorMappingMiddleware, RateLimitMiddleware, MetricsMiddleware
- `test_tool_mapper.py` — DefaultToolMapper, PerSkillToolMapper, FlatHierarchyToolMapper
- `test_context_manager.py` — ContextMapping, ContextManager async operations
- `test_router_extended.py` — WeightedKeywordRouter, LLMRouter (sync/async/fuzzy/error), aresolve, unregister
- `test_registry.py` — adapter registration, discovery, creation
- `test_testing_utils.py` — MockAdapter, MockStreamingAdapter, RecordingMiddleware, factory helpers
- `test_bridge.py` — end-to-end MCP→A2A integration tests

---

## 2. Pending

This section covers features from the research report that are not yet implemented, organized by priority and effort.

### Phase 2 — FastMCP 3.0 Deep Integration (High Impact, Low-Medium Effort)

**Transform classes for component modification** — FastMCP 3.0's `Transform` classes (`Namespace`, `ToolTransform`, `Visibility`, `ResourcesAsTools`, `PromptsAsTools`) are not yet used. Agentique should expose transform hooks so users can:
- Apply `Namespace` transforms per agent to prevent tool name collisions
- Use `Visibility` with session-level control (`ctx.enable_components()` / `ctx.disable_components()`) for dynamic agent availability
- Apply custom `ToolTransform` instances for tool renaming

**FastMCP Middleware integration** — While agentique has its own `MiddlewareChain` at the bridge layer, it does not yet use FastMCP 3.0's own `Middleware` class with `on_call_tool`, `on_list_tools`, and `on_read_resource` hooks. These should be used for server-level cross-cutting concerns.

**Dependency injection via `Depends()`** — The codebase manually constructs dependencies rather than using FastMCP 3.0's `Depends()` for clean injection of agent clients, configuration, and services into tools.

**OpenTelemetry integration** — No tracing spans are emitted. Should add `agentique.agent_name`, `agentique.protocol`, `agentique.task_state` attributes to FastMCP's built-in OpenTelemetry spans.

**Storage backends for task persistence** — TaskManager uses in-memory dicts. FastMCP 3.0 supports pluggable storage backends (Redis, DynamoDB, filesystem). Task state should use these for production persistence.

**Elicitation with response types** — FastMCP 3.0's `ctx.elicit()` with Pydantic models is not yet used. Should enable structured confirmation dialogs for agent actions (e.g., when an A2A agent returns `input-required` state).

**Sampling with tool loop** — `ctx.sample()` now supports `tools` and `tool_choice` parameters. The `LLMRouter` uses basic sampling but does not leverage the tool loop for structured agent selection via `result_type`.

**Lifespan composition** — FastMCP's pipe operator (`lifespan_a | lifespan_b`) for composing startup/shutdown logic across multiple agent connections is not yet used.

**Structured content** — Tools returning dicts/Pydantic models should use FastMCP's automatic structured JSON alongside traditional content, aligning with MCP's `outputSchema`/`structuredContent` spec.

**`TaskConfig` API** — Currently uses `task=True` but should leverage `TaskConfig(mode="optional", poll_interval=timedelta(seconds=2))` for fine-grained control.

**Composition via mounting** — FastMCP's `mount()` could enable mounting separate bridges (A2A, HTTP, local) under a single MCP server with automatic namespace isolation.

### Phase 3 — A2A Protocol Completeness (Medium Impact, Moderate Effort)

**Push notification support** — A2A supports webhook-based push notifications for long-running tasks (`PushNotificationConfig`, JWT signing). Agentique should:
- Accept push notification config from MCP clients
- Configure push notifications on A2A servers
- Translate incoming webhooks into MCP `notifications/tasks/status`
- Use SDK's `InMemoryPushNotifier` and `PushNotificationConfigStore`

**Task resubscription** — A2A's `tasks/resubscribe` method allows reconnecting to active streams after disconnection. The bridge should implement reconnection logic instead of failing on SSE drops.

**gRPC transport** — A2A SDK includes `GrpcTransport`. Agentique should support gRPC as a backend option via `ClientConfig(ordered_transports=["gRPC", "JSONRPC"])`.

**Extended agent cards** — A2A distinguishes between public cards (unauthenticated, at `/.well-known/agent-card.json`) and extended cards (authenticated, revealing additional skills). Agentique should fetch extended cards when credentials are available.

**A2A extensions mechanism** — Users should be able to define custom A2A extensions that propagate to agents and are received back in responses (e.g., a tracing extension carrying OpenTelemetry span context).

**Auth-required and rejected state handling** — A2A's `auth-required` and `rejected` task states should be translated into MCP elicitation or appropriate error flows, not just generic errors.

### Phase 4 — Ecosystem & Community (High Long-Term Impact, Higher Effort)

**Published `agentique-core` package** — Extract the `core/` module as a standalone zero-dependency package so adapter authors don't need to depend on FastMCP or a2a-sdk.

**OpenAI Agents API adapter** — A third adapter targeting OpenAI's Agents API would further validate the protocol abstraction and expand the ecosystem.

**LangChain Runnable adapter** — An adapter wrapping LangChain Runnable endpoints would connect to the largest agent framework ecosystem.

**Comprehensive documentation site** — API reference, tutorials, adapter development guide, and deployment patterns.

**CI/CD pipeline** — Automated testing, type checking (mypy), linting (ruff), and publishing to PyPI.

**MCP Tasks spec alignment** — MCP's November 2025 spec added experimental Tasks support with states matching A2A's lifecycle. As this stabilizes, agentique's task bridge should align with native MCP task primitives.

**MCP extensions framework** — Define an `agentique://` extension carrying metadata like `agent_protocol`, `original_task_id`, and `agent_card_url` through MCP interactions.

**Property-based testing** — Use Hypothesis for property-based tests validating protocol compliance across all adapters.

---

## File Inventory

### New Files
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

### Modified Files
| File | Changes |
|------|---------|
| `src/agentique/__init__.py` | Exports new components |
| `src/agentique/core/__init__.py` | Exports new types, mappers, registry |
| `src/agentique/core/types.py` | Added `ContextMapping`, `BridgeContext.replace()` |
| `src/agentique/core/protocols.py` | Added `AdapterFactory` protocol |
| `src/agentique/bridge/__init__.py` | Exports middleware, context manager, new routers |
| `src/agentique/bridge/router.py` | Added `LLMRouter`, `WeightedKeywordRouter`, `aresolve()`, `unregister()` |
| `src/agentique/adapters/__init__.py` | Registers built-in adapters via `@register_adapter` |
| `src/agentique/server.py` | Integrates middleware chain, context manager, `aresolve()` |
| `tests/test_core.py` | Added `BridgeContext.replace()` and `ContextMapping` tests |
| `pyproject.toml` | Added httpx dependency, entry points, bumped version to 0.3.0 |
