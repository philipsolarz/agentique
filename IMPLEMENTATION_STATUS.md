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

**FastMCP server factory** with seven core tools:
- `agent` — send message and return structured task output (with LLM-capable routing via `aresolve`)
- `agents` — list available agents
- `task` — query task state
- `inspect` — view agent hierarchy
- `agent_background` — background task with `Progress` reporting
- `enable_components` / `disable_components` — session-scoped visibility controls using FastMCP component visibility APIs

**Task manager** — In-memory task tracking with state machine, conversation continuity, hierarchy tracking.

**Streaming** — Full streaming support via `stream_message` with `StreamChunk` types.

**Sub-agent visibility** — `AgentHierarchy` and `SubAgentInfo` track nested agent structures with branch path resolution.

**Background task support** — Uses FastMCP's `TaskConfig(mode="optional", poll_interval=...)` and `Progress` dependency injection for progress tracking.

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

### Phase 2 — FastMCP 3.0 Deep Integration (Substantially Complete)

**Transform classes integrated** — Agentique now wires FastMCP transforms through configuration:
- `Namespace` via `AGENTIQUE_COMPONENT_NAMESPACE`
- `ToolTransform` via `AGENTIQUE_TOOL_TRANSFORMATIONS`
- `Visibility` via `AGENTIQUE_DISABLED_COMPONENT_NAMES`
- Optional `ResourcesAsTools` / `PromptsAsTools` flags

**FastMCP middleware hooks integrated** — Added `FastMCPBridgeMiddleware` to bridge `MiddlewareChain` into FastMCP server hooks:
- `on_call_tool`
- `on_list_tools`
- `on_read_resource`

**Dependency injection via `Depends()`** — Core tools now inject shared services (`AgentRouter`, `TaskManager`, `ContextManager`, `AgentProvider`, adapter, event emitter) through FastMCP dependencies.

**Structured output & output schema alignment** — `agent` and `agent_background` tools return `ToolResult` with structured content, and `agent` declares an explicit output schema.

**Task API modernization** — Background tasks now use `TaskConfig` poll interval controls from config (`AGENTIQUE_BACKGROUND_TASK_POLL_INTERVAL_SECONDS`).

**Lifespan composition** — Server lifespan now composes lifecycle hooks with external lifespan callables using FastMCP `combine_lifespans`.

**OpenTelemetry span attributes** — Agent tool lifecycle emits optional span attributes when tracing is enabled:
- `agentique.agent_name`
- `agentique.protocol`
- `agentique.task_id`
- `agentique.task_state`

**Elicitation with typed response model** — When agents emit `input-required` / `auth-required` states, Agentique now uses `ctx.elicit()` with a Pydantic response type for structured client feedback.

**Sampling tool loop improvements** — `LLMRouter` now calls `ctx.sample(..., tools=..., result_type=...)` and parses structured results before falling back.

### Phase 3 — A2A Protocol Completeness (Partially Complete)

**Push notification config passthrough** — Adapter now maps context-level options into `MessageSendConfiguration.pushNotificationConfig`.

**Task resubscription support** — Streaming now attempts automatic `tasks/resubscribe` recovery for retryable stream drops when a task ID is known.

**gRPC/transport preference support** — `A2AClientPool` now supports `ClientConfig.supported_transports` and `use_client_preference` via config.

**A2A extension passthrough** — Adapter and client pool now propagate configured/per-request extensions to A2A SDK calls.

**Auth/rejected state handling** — `auth-required` and `rejected` states now map to task transitions and elicitation flow in the MCP tool path.

### Test Coverage

- `test_core.py` — TaskState, TaskTracker, AgentInfo, BridgeContext (including `.replace()`), AgentEvent, StreamChunk, AgentHierarchy, ContextMapping, Router, Events, config parsing helpers
- `test_middleware.py` — MiddlewareChain ordering, LoggingMiddleware, ErrorMappingMiddleware, RateLimitMiddleware, MetricsMiddleware, FastMCP middleware bridge hooks
- `test_tool_mapper.py` — DefaultToolMapper, PerSkillToolMapper, FlatHierarchyToolMapper
- `test_context_manager.py` — ContextMapping, ContextManager async operations
- `test_router_extended.py` — WeightedKeywordRouter, LLMRouter (sync/async/fuzzy/error/structured sampling), aresolve, unregister
- `test_registry.py` — adapter registration, discovery, creation
- `test_testing_utils.py` — MockAdapter, MockStreamingAdapter, RecordingMiddleware, factory helpers
- `test_a2a_adapter.py` — extension/config passthrough, resubscribe recovery, card extension forwarding
- `test_bridge.py` — end-to-end MCP→A2A integration tests

---

## 2. Pending

This section covers features from the research report that are not yet implemented, organized by priority and effort.

### Phase 2 — FastMCP 3.0 Deep Integration (High Impact, Low-Medium Effort)

**Storage backends for task persistence** — TaskManager uses in-memory dicts. FastMCP 3.0 supports pluggable storage backends (Redis, DynamoDB, filesystem). Task state should use these for production persistence.

**Provider/storage integration for tasks** — Server task metadata is still held in memory. Migration to FastMCP state stores (or Redis/DynamoDB-backed custom stores) remains open.

**Composition via mounting** — FastMCP's `mount()` could enable mounting separate bridges (A2A, HTTP, local) under a single MCP server with automatic namespace isolation.

**Transform ergonomics** — Transform hooks are wired, but higher-level presets (e.g., per-agent automatic namespacing policies and richer tool transform helpers) are still minimal.

### Phase 3 — A2A Protocol Completeness (Medium Impact, Moderate Effort)

**Push notification webhooks (server-side receiver)** — Client-side push config passthrough is implemented, but webhook ingestion and translation into MCP task notifications is still pending.

**Extended agent cards** — A2A distinguishes between public cards (unauthenticated, at `/.well-known/agent-card.json`) and extended cards (authenticated, revealing additional skills). Agentique should fetch extended cards when credentials are available.

**Extension contract UX** — Extension passthrough exists, but typed extension registration/validation APIs are not yet implemented.

**Task resubscription robustness** — Basic automatic resubscribe exists, but needs richer retry policy (backoff, resumable checkpoints, and explicit observability).

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

### New Files (This Pass)
| File | Description |
|------|-------------|
| `tests/test_a2a_adapter.py` | New unit tests for A2A extension/config passthrough and stream resubscribe behavior |

### Modified Files (This Pass)
| File | Changes |
|------|---------|
| `src/agentique/__init__.py` | Exports new components |
| `src/agentique/core/config.py` | Added FastMCP transform/task settings and A2A transport/extension/push config parsing helpers |
| `src/agentique/bridge/__init__.py` | Exports middleware, context manager, new routers |
| `src/agentique/bridge/middleware.py` | Added `FastMCPBridgeMiddleware` for `on_call_tool`, `on_list_tools`, `on_read_resource` integration |
| `src/agentique/bridge/router.py` | Upgraded `LLMRouter` to use `ctx.sample(..., tools=..., result_type=...)` structured routing |
| `src/agentique/adapters/a2a/client.py` | Added transport preference, extension, push config, and resolver options support in client pool |
| `src/agentique/adapters/a2a/adapter.py` | Added per-call A2A config/extension parsing, push config passthrough, resubscribe recovery, and state handling enhancements |
| `src/agentique/server.py` | Added FastMCP transforms, middleware bridge, `Depends()` injection, `TaskConfig`, structured tool outputs, session visibility tools, lifespan composition, OTel attributes |
| `tests/test_core.py` | Added config parsing and background poll-interval tests |
| `tests/test_middleware.py` | Added tests for `FastMCPBridgeMiddleware` hooks |
| `tests/test_router_extended.py` | Added structured-sampling test and `ctx.sample` kwargs compatibility |
| `tests/test_bridge.py` | Updated for FastMCP 3 result shapes and modern A2A client behavior |
| `tests/support/a2a_server.py` | Updated request handler to modern A2A request/response contracts |
| `tests/support/adk_agent.py` | Added compatibility fallback for evolving Google ADK callable interfaces |
| `IMPLEMENTATION_STATUS.md` | Updated completion/pending status and file inventory |
