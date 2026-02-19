# DONE

## Focus #1 — Typed LLMRouter with RoutingDecision

**Completed**: 2026-02-18

### What changed
- `src/agentique/bridge/router.py`: Added `RoutingDecision(BaseModel)` with five fields
  (agent_id, confidence, reasoning, fallback_agents, requires_decomposition). Rewrote
  `LLMRouter.aselect()` to use `result_type=RoutingDecision` (fixing the silent bug where
  a list was passed instead of a Pydantic model). Added fallback agent cascade, OTel span
  attributes, and improved system/plan prompts. Improved `_build_agent_manifest()` to use
  numbered multi-line entries with richer context.
- `tests/unit/test_router.py`: 20 new tests covering RoutingDecision validation, structured
  sampling, plain-text fallback, fallback cascade, all-fallbacks-fail path, requires_
  decomposition logging, manifest format, and _match_agent edge cases.
- `tests/__init__.py`, `tests/unit/__init__.py`: Created test package structure.

### Why it matters
The `result_type=agent_names` bug meant every routing call silently fell back to string
matching, defeating the purpose of structured sampling. With `RoutingDecision`, the LLM
now produces validated, typed routing decisions with confidence scores and fallback plans,
making routing more reliable, observable, and testable.

---

## Focus #2 — Transform-based ToolMapper migration + Phase 4 gateway items

**Completed**: 2026-02-19

### What changed

**`src/agentique/bridge/provider.py`**:
- Added `tool_mapper` parameter to `AgentProvider.__init__()`.
- Added `_make_tool_from_mapper_def()` — creates `FunctionTool` instances from
  mapper-produced dicts (name, description, message handler).
- `_list_tools()` now routes through the tool mapper when one is set; falls back
  to the previous default one-tool-per-agent behaviour when `tool_mapper=None`.
  Card-derived proxy tools are always appended regardless of mapper choice.

**`src/agentique/bridge/tool_transforms.py`** (new):
- `namespace_transform_for_agent(agent_name)` → FastMCP `Namespace` transform.
- `visibility_transform(names, *, enabled, tags)` → FastMCP `Visibility` transform.
- `resources_as_tools_transform(server)` → FastMCP `ResourcesAsTools` transform
  (server-bound; must be called after server creation).
- `prompts_as_tools_transform(server)` → FastMCP `PromptsAsTools` transform (same).
- `default_transform_stack(agent_name, *, server, resources_as_tools, prompts_as_tools)`
  — convenience factory producing a composable transform list.

**`src/agentique/server.py`**:
- `create_server()` gains three new parameters:
  - `tool_mapper` — passed to `AgentProvider`.
  - `resources_as_tools=False` — applies `ResourcesAsTools(mcp)` post-build.
  - `prompts_as_tools=False` — applies `PromptsAsTools(mcp)` post-build.
- `agent_tool` return type changed from `str` to `ToolResult`. `structured_content`
  now carries: `task_id`, `agent`, `state`, `artifact_uris`, `response`.
- `auth-required` state now triggers `ctx.elicit()` (structured credential prompt)
  with graceful fallback to `ctx.warning()` if elicitation is unavailable.

**`src/agentique/testing.py`** (new):
- `MockAdapter` — `AgentAdapter`-protocol-compliant in-memory adapter with
  configurable per-agent responses, per-agent event streams, and a call log.
- `assert_adapter_protocol(adapter)` — validates structural Protocol compliance
  via `isinstance(adapter, AgentAdapter)`.
- `InMemoryBridge` — wraps `create_server()` + `MockAdapter` for integration tests.

**Tests** (4 new test files, +85 tests):
- `tests/unit/test_testing_module.py` (36 tests) — covers MockAdapter, protocol
  assertion, InMemoryBridge.
- `tests/unit/test_tool_mapper_provider.py` (20 tests) — covers DefaultToolMapper,
  PerSkillToolMapper, FlatHierarchyToolMapper wired into AgentProvider.
- `tests/unit/test_tool_transforms.py` (20 tests) — covers all transform factories
  and `default_transform_stack`.
- `tests/unit/test_structured_output.py` (9 tests) — covers TaskManager artifact
  API, structured_content field validation, auth-required event detection.

### Why it matters
- **ToolMapper wired**: the three mapper implementations were previously dead code.
  Now they can be used via `create_server(tool_mapper=PerSkillToolMapper())` to
  expose per-skill tools without touching provider internals.
- **Transform factories**: users can compose FastMCP-native transform chains with
  a single import, achieving the same goals as custom ToolMapper but with
  composability guarantees and correct ordering from FastMCP's transform pipeline.
- **Structured agent output**: `agent_tool` now returns a machine-readable envelope
  with task ID, terminal state, and artifact URIs — enabling clients to immediately
  query artifacts, chain calls, and perform audit logging without parsing free-form text.
- **Auth elicitation**: `auth-required` A2A state now flows through MCP's first-class
  elicitation mechanism rather than a bare warning, enabling interactive credential
  collection in clients that support elicitation.
- **Testing module**: every production adapter and gateway configuration is now
  testable via `MockAdapter` + `InMemoryBridge` without real agent processes.

---

## Focus #3 — Protocol evolution (Phase 1 roadmap)

**Completed**: 2026-02-19

### What changed

**`src/agentique/core/errors.py`**:
- Added `ContentTypeNotSupportedError` (mcp_code=-32600, a2a_code=-32002)
- Added `UnsupportedOperationError` (mcp_code=-32601, a2a_code=-32003)
- Added `TaskNotCancelableError` (mcp_code=-32603, a2a_code=-32004)
- Added `PushNotificationNotSupportedError` (mcp_code=-32601, a2a_code=-32005)

**`src/agentique/bridge/middleware.py`**:
- `ErrorMappingMiddleware.A2A_ERROR_MAP` now covers all 5 A2A error codes
  (-32001 through -32005) using the precise typed error classes instead of
  generic fallbacks for -32002 and -32003.

**`src/agentique/bridge/router.py`**:
- `LLMRouter` gains two new parameters:
  - `sampling_fallback: Callable | None` — invoked when `ctx.sample()` is
    unavailable (no sampling capability) or raises. Accepts both sync and async
    callables. Signature: `(message: str, available: list[AgentInfo]) -> str`.
  - `enable_introspection: bool = False` — when `True`, passes
    `tools=[inspect_agent, list_agents]` to every `ctx.sample()` call, enabling
    the routing LLM to query agent details before committing to a decision (the
    "first inflection" agentic planning loop from the architecture document).
- New private helpers:
  - `_make_introspection_tools(available)` — returns `[inspect_agent, list_agents]`
    callables bound to the current agent registry.
  - `_call_fallback(fallback, message, available)` — invokes the fallback,
    awaiting it if async.

**`src/agentique/server.py`**:
- `create_server()` gains `extra_providers: list[Any] | None = None` — combines
  `AgentProvider` with additional FastMCP providers (e.g., `OpenAPIProvider`,
  `FileSystemProvider`) in the `providers=[...]` list, establishing the foundation
  for the "third inflection" provider composition pattern.

**Tests** (2 new test files, +44 tests):
- `tests/unit/test_error_mapping.py` (20 tests) — all 5 A2A error types, all 5
  middleware mappings, HTTP fallbacks, connection errors, pass-throughs, async
  `process()` path, and completeness assertion on `A2A_ERROR_MAP`.
- `tests/unit/test_router_enhancements.py` (24 tests) — sampling fallback (sync +
  async + ctx=None + ctx.sample raises + unknown-agent error), introspection
  (`enable_introspection=True/False`, `_make_introspection_tools` functions),
  and provider composition (`extra_providers` in `create_server()`).

---

## Focus #4 — Lifespan composition, session state, and A2A Extensions

**Completed**: 2026-02-19

### What changed

**`src/agentique/bridge/lifespans.py`** (new):
- `make_cleanup_lifespan(pool, *, clear_deps_fn)` — `@lifespan`-decorated factory
  that closes the `A2AClientPool` (and optionally clears the deps registry) on
  server shutdown. Handles `pool=None` gracefully for pre-built adapter setups.
- `make_health_monitor_lifespan(adapter, *, interval, on_degraded)` — starts a
  background health-check loop using `anyio.create_task_group()` (works under
  both asyncio and trio). Optional `on_degraded` callback for alerting.
- `compose_lifespans(*lifespans)` — combines `Lifespan` instances with `|`,
  silently skips `None` entries. Returns `None` for empty lists (FastMCP-safe).

**`src/agentique/server.py`**:
- `create_server()` gains four new parameters:
  - `session_state_store: AsyncKeyValue | None` — pluggable KV backend for
    `ctx.get_state`/`ctx.set_state`; passed directly to `FastMCP()`.
  - `sampling_handler: Any | None` — `AnthropicSamplingHandler` or
    `OpenAISamplingHandler`; enables `ctx.sample()` when MCP clients lack
    sampling support.
  - `sampling_handler_behavior: str = "fallback"` — controls when the handler
    is used (``"fallback"`` or ``"always"``).
  - `health_check_interval: float | None` — when set, composes a
    `make_health_monitor_lifespan` into the server lifespan chain.
- The server is now always constructed with a composed lifespan (at minimum,
  `make_cleanup_lifespan` for pool close on shutdown).
- `pool` variable hoisted out of the `if adapter is None:` block so the cleanup
  lifespan can always reference it (may be `None` for pre-built adapters).

**`src/agentique/extensions.py`** (new):
- Four URI constants: `ROUTING_METADATA_URI`, `POLICY_CONTEXT_URI`,
  `MCP_SESSION_URI`, `TRACE_CONTEXT_URI` (all `com.agentique/` namespaced).
- `ALL_EXTENSION_URIS` list for capability negotiation advertisement.
- `pack_*` / `unpack_*` helpers for all four channels (pure `dict`-based,
  no Pydantic coupling at call sites).
- `current_trace_context()` — extracts active OTel span's W3C trace context
  via `opentelemetry.propagate.inject`; returns `{}` when OTel is unconfigured.
- `build_gateway_metadata(...)` — one-shot convenience function that composes
  routing decision, policy context, session info, and trace context into a
  complete A2A message metadata dict.

**Tests** (3 new test files, +75 tests):
- `tests/unit/test_lifespans.py` (29 tests) — lifespan factories, composition,
  cleanup, health monitor, pipe operator, anyio compatibility.
- `tests/unit/test_extensions.py` (34 tests) — URI constants, all pack/unpack
  round-trips, `current_trace_context`, `build_gateway_metadata`.
- `tests/unit/test_server_new_params.py` (12 tests) — `create_server()` new
  params, import accessibility, error types.

### Why it matters
- **Complete error semantics**: clients and middleware can now distinguish all 5
  A2A failure modes — task missing, content type mismatch, unsupported operation,
  non-cancellable task, and no push support — instead of collapsing -32002/-32003
  into generic `TranslationError`/`AgentUnavailableError`.
- **Sampling resilience**: gateways deployed against MCP clients that lack sampling
  capability (e.g., older clients or test harnesses) can now register a
  `sampling_fallback` and remain functional rather than hard-failing at routing time.
- **Agentic planning loop**: with `enable_introspection=True`, the routing LLM can
  call `inspect_agent("billing")` or `list_agents()` mid-sample before finalising its
  `RoutingDecision`, enabling multi-turn agent-detail retrieval within a single
  routing step — the "first inflection" pattern described in the architecture document.
- **Provider composition**: `extra_providers` unlocks heterogeneous server topologies
  (A2A agents + OpenAPI endpoints + file system resources) from a single
  `create_server()` call, setting the stage for the "third inflection" in the roadmap.

---

## Focus #5 — Session state unification, TaskConfig, capability advertisement

**Completed**: 2026-02-19

### What changed

**`src/agentique/bridge/context_manager.py`**:
- `resolve_context()` gains `ctx: Any = None` parameter. When provided, reads the
  persisted context list from `ctx.get_state("agentique:contexts")` before falling
  back to in-memory mapping, enabling cross-restart continuity. Writes new context
  IDs back to state (idempotent — skips if already present).
- `track_task()` gains `ctx: Any = None`. When provided, appends the task ID to
  `ctx.get_state("agentique:ctx_tasks:{context_id}")` (idempotent append).
- `cleanup_session()` gains `ctx: Any = None`. Calls `ctx.delete_state("agentique:contexts")`
  on disconnect.
- All ctx operations use `getattr(ctx, "get_state", None)` for graceful degradation
  when ctx is absent or lacks state methods. Errors in state I/O are silently caught.
- Added `_STATE_KEY_CONTEXTS` and `_STATE_KEY_CTX_TASKS_PREFIX` constants.

**`src/agentique/server.py`**:
- `agent_tool` threads `ctx=ctx` through to both `_ctx_mgr.resolve_context()` and
  `_ctx_mgr.track_task()`, activating persistence for every tool invocation.
- Pre-computes `_agent_task_config`: `TaskConfig(mode="optional")` when
  `enable_background_tasks=True` AND `docket` (pydocket) is installed; falls back to
  `task=False` when either is absent. Applied to the `@mcp.tool(name="agent", task=...)`
  decorator, advertising MCP Tasks support to clients that understand SEP-1686.
- `agent_tool` structured_content now includes `"mcp_related_task": task_id` — the
  `io.modelcontextprotocol/related-task` hint correlating the A2A task with its MCP
  background task counterpart.
- Added `@mcp.resource("a2a://capabilities")` returning a JSON document advertising
  gateway name, version, all 4 extension URIs, supported transports, feature flags
  (`background_tasks`, `elicitation`, `tool_confirmation`, `push_notifications`), and
  `session_state_persistent` flag. This is the canonical capability advertisement
  channel since FastMCP 3.0 does not yet expose `capabilities.extensions` negotiation.

**Tests** (2 new test files + 8 tests added to `test_server_new_params.py`, +47 tests):
- `tests/unit/test_context_manager_state.py` (33 tests) — resolve_context without/with
  ctx, cross-restart continuity, idempotency, missing state methods, error recovery,
  track_task persistence and idempotency, cleanup with delete_state, state key constants.
- `tests/unit/test_agent_task_config.py` (6 tests) — background tasks enabled/disabled,
  docket missing fallback, TaskConfig import missing fallback, mcp_related_task in
  ToolResult schema, server code inspection.
- `tests/unit/test_server_new_params.py` (8 new tests) — capabilities resource URI
  registered, valid JSON, all 4 extension URIs, gateway name, features dict, persistent
  false/true by store presence, transports default.

### Why it matters
- **Session state unification**: `ContextManager` mappings now survive gateway restarts
  and work across replicas when a persistent `session_state_store` (Redis/DynamoDB) is
  configured. Without ctx, the in-memory path is unchanged — zero regression.
- **MCP Tasks advertisement**: clients that understand SEP-1686 now see
  `taskSupport="optional"` on the `agent` tool, enabling them to submit long-running
  A2A invocations as background MCP tasks with status polling and push notifications.
- **Capability advertisement**: any MCP client can `read_resource("a2a://capabilities")`
  to discover the gateway's extension URIs, transport preferences, and feature flags
  without requiring out-of-band documentation.
