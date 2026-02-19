# TODO

## Focus #1 — Typed LLMRouter with RoutingDecision ✅ DONE

## Focus #2 — Transform-based ToolMapper migration ✅ DONE

Audit results: ToolMapper classes existed but were NOT wired into AgentProvider.
Both pillars implemented:

1. **ToolMapper wired into AgentProvider + create_server()**:
   - [x] Audit: ToolMapper dead code, not wired in
   - [x] Add `tool_mapper` param to `AgentProvider.__init__()`
   - [x] Add `_make_tool_from_mapper_def()` method (creates FunctionTools from mapper defs)
   - [x] `_list_tools()` uses mapper when provided; falls back to default one-per-agent
   - [x] Wire `tool_mapper` through `create_server(tool_mapper=...)`
   - [x] Tests in `tests/unit/test_tool_mapper_provider.py`

2. **FastMCP transform factory functions** (`bridge/tool_transforms.py`):
   - [x] `namespace_transform_for_agent(agent_name)` → `Namespace` transform
   - [x] `visibility_transform(names)` → `Visibility` transform
   - [x] `resources_as_tools_transform(server)` → `ResourcesAsTools` transform
   - [x] `prompts_as_tools_transform(server)` → `PromptsAsTools` transform
   - [x] `default_transform_stack()` convenience factory
   - [x] Wire `resources_as_tools=True` / `prompts_as_tools=True` into `create_server()`
   - [x] Tests in `tests/unit/test_tool_transforms.py`

## Phase 4 — Gateway polish ✅ PARTIALLY DONE

- [x] **Structured `agent` tool output** — returns `ToolResult` with `task_id`, `agent`,
      `state`, `artifact_uris`, `response` in `structured_content` (was plain string)
- [x] **`auth-required` → `ctx.elicit()` flow** — A2A auth-required state now triggers
      structured elicitation; falls back to `ctx.warning` if elicitation fails
- [x] **`agentique.testing` module** — `MockAdapter`, `assert_adapter_protocol`,
      `InMemoryBridge` in `src/agentique/testing.py`

## Focus #3 — Protocol evolution items ✅ DONE

Implemented Phase 1 items from the second compass document:

1. **Complete A2A error code mapping** — all 5 codes:
   - [x] `ContentTypeNotSupportedError` (a2a_code=-32002, mcp_code=-32600)
   - [x] `UnsupportedOperationError` (a2a_code=-32003, mcp_code=-32601)
   - [x] `TaskNotCancelableError` (a2a_code=-32004, mcp_code=-32603)
   - [x] `PushNotificationNotSupportedError` (a2a_code=-32005, mcp_code=-32601)
   - [x] `ErrorMappingMiddleware.A2A_ERROR_MAP` updated to use specific types
   - [x] Tests in `tests/unit/test_error_mapping.py` (20 tests)

2. **Sampling fallback handler in LLMRouter** — "second inflection" safety net:
   - [x] `sampling_fallback: Callable | None` parameter on `LLMRouter`
   - [x] Invoked when `ctx` is None / lacks `.sample`, or when `ctx.sample()` raises
   - [x] Supports both sync and async callables
   - [x] Tests in `tests/unit/test_router_enhancements.py`

3. **Agentic planning loop** — "first inflection" via `ctx.sample(tools=[...])`:
   - [x] `enable_introspection: bool = False` parameter on `LLMRouter`
   - [x] `_make_introspection_tools(available)` returns `[inspect_agent, list_agents]`
   - [x] `tools=introspection_tools` passed to `ctx.sample()` when enabled
   - [x] Tests in `tests/unit/test_router_enhancements.py`

4. **Provider composition** — "third inflection" foundation:
   - [x] `extra_providers: list[Any] | None` parameter on `create_server()`
   - [x] AgentProvider + extra_providers combined in `providers=[...]`
   - [x] Tests in `tests/unit/test_router_enhancements.py`

## Focus #4 — Lifespan + session state + extensions ✅ DONE

Implemented remaining Phase 1/2 items from the second compass document:

1. **Lifespan composition** (`bridge/lifespans.py`):
   - [x] `make_cleanup_lifespan(pool, clear_deps_fn)` — closes A2AClientPool on shutdown
   - [x] `make_health_monitor_lifespan(adapter, interval, on_degraded)` — optional background health loop
   - [x] `compose_lifespans(*ls)` — composes with `|`, skips None entries
   - [x] Wired into `create_server()` via `health_check_interval` flag
   - [x] anyio-compatible (works under both asyncio and trio)

2. **Session state store + sampling handler** (`server.py`):
   - [x] `session_state_store: AsyncKeyValue | None` → passed to `FastMCP(session_state_store=...)`
   - [x] `sampling_handler: Any | None` → `FastMCP(sampling_handler=..., sampling_handler_behavior=...)`
   - [x] `sampling_handler_behavior: str = "fallback"` — "fallback" or "always"

3. **A2A Extensions passthrough** (`extensions.py`):
   - [x] `ROUTING_METADATA_URI`, `POLICY_CONTEXT_URI`, `MCP_SESSION_URI`, `TRACE_CONTEXT_URI`
   - [x] `pack_*` / `unpack_*` helpers for all 4 channels
   - [x] `current_trace_context()` — extracts OTel W3C trace context
   - [x] `build_gateway_metadata(...)` — convenience builder for all channels

4. **Tests** (3 new test files, +75 tests):
   - [x] `test_lifespans.py` (29 tests)
   - [x] `test_extensions.py` (34 tests)
   - [x] `test_server_new_params.py` (12 tests)

## Focus #5 — Session state unification, TaskConfig, capability advertisement ✅ DONE

- [x] **Session state unification** — `ContextManager.resolve_context()`, `track_task()`,
      `cleanup_session()` accept `ctx: Any = None` and persist/restore via
      `ctx.get_state`/`ctx.set_state`/`ctx.delete_state` with graceful degradation.
- [x] **MCP Tasks annotation** — `TaskConfig(mode="optional")` on `agent` tool when
      `enable_background_tasks=True` AND docket is installed; `mcp_related_task` field
      in structured_content for SEP-1686 correlation.
- [x] **Gateway capability advertisement** — `@mcp.resource("a2a://capabilities")` returns
      JSON with extensions, transports, and feature flags (workaround for FastMCP 3.0
      not yet exposing `capabilities.extensions` negotiation).

## Backlog

- JWT validation in `WebhookReceiver`
- Persistent artifact store (backed by `TaskStore` abstraction)
- `ToolMapper` protocol deprecation notice (now superseded by FastMCP transforms)
- Entry point discovery for third-party adapters (`agentique.adapters` entry point group)
- Background task execution via Docket (SQLite/Postgres persistence, horizontal scaling)
