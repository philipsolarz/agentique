# Agentique: an Intelligent Agent Gateway

**Agentique is an Intelligent Agent Gateway — a smart orchestration layer that sits between MCP clients and backend agent ecosystems.** It is not a passive bridge that forwards messages verbatim. It is an active, opinionated gateway that applies intelligence to routing decisions, enforces policy at session boundaries, normalises backend transports, and promotes agent-produced artifacts to first-class MCP resources. Built on FastMCP 3.0, A2A SDK, and the Model Context Protocol, it treats the client's LLM as the always-available decision engine and relies entirely on it for every non-trivial routing choice.

---

## Mission statement and architectural vision

**Mission statement**: *Agentique is a Python framework that transforms any collection of A2A (or other protocol) agents into a fully-featured, intelligently-routed MCP server. It applies LLM-driven routing, policy-based access control, and automatic resource registration so that MCP clients — Claude, Cursor, ChatGPT, or custom hosts — interact with backend agents as if they were native MCP capabilities.*

The gateway sits at the intersection of three maturing specifications: **MCP** (client-facing, now with structured output, tasks, and elicitation), **A2A** (the agent-to-agent protocol at v0.3, heading toward v1.0 under Linux Foundation stewardship), and **FastMCP 3.0** (provider/transform/middleware server framework). Rather than treating these as transport pipes, Agentique exploits each specification's richest semantics: MCP sampling for routing, MCP resources for artifact exposure, FastMCP middleware for policy enforcement, and A2A task IDs for resilient stream reconnection.

The architecture follows three layers:

- **Protocol Layer** (top): FastMCP 3.0 server exposing MCP primitives — tools, resources, prompts — with middleware and transforms applied
- **Bridge Layer** (middle): Intelligent routing (LLM-first, Plan→Execute→Verify), policy enforcement, task management, artifact registry
- **Adapter Layer** (bottom): Pluggable backends — A2A as the primary adapter, with HTTP and MCP proxy adapters also available

A key architectural insight: the MCP client's LLM and the backend A2A agents together form a **closed reasoning loop**. The gateway orchestrates this loop — it does not merely pass messages through it. This changes the design constraints: every gateway decision (routing, visibility, verification) should leverage LLM reasoning rather than heuristics, and every piece of information the LLM might need (agent capabilities, task state, artifacts) should be a first-class MCP primitive.

---

## The four pillars of the Intelligent Gateway

### Pillar 1 — Pure LLM Routing: Plan → Execute → Verify

**Before**: `AgentRouter` defaulted to `KeywordRouter` — a deterministic keyword-matching algorithm that scored agents by skill overlap with the user message. `LLMRouter` existed but fell back to `KeywordRouter` when sampling failed. This meant the system could route without ever consulting an LLM.

**After**: Keyword-matching routers (`KeywordRouter`, `WeightedKeywordRouter`) have been deleted entirely. `LLMRouter` is now the only non-direct routing strategy, and it hard-requires a live `ctx.sample()` context. The synchronous `select()` method raises `RuntimeError` to force callers onto the async path. There is no deterministic fallback.

**The Plan → Execute → Verify loop:**

*Plan* — `LLMRouter.aselect()` builds a structured **capability manifest** from every visible agent — name, description, skills, and endpoint — and presents it to the client LLM via `ctx.sample()`. The prompt constrains the response to one of the valid agent names. Structured sampling (`result_type=agent_names`) is attempted first for cleaner extraction, with plain-text sampling as fallback. The LLM can see the full agent graph and makes an informed, capability-aware routing decision.

*Execute* — `AgentRouter.aresolve()` calls the selected adapter and streams events back to the MCP client. In single-agent setups or when an explicit `target` is provided, no LLM call is made — the shortcut path avoids unnecessary sampling latency.

*Verify* (optional) — `LLMRouter.averify()` calls `ctx.sample()` a second time after the agent responds, asking the client LLM whether the response adequately addresses the original request. When `enable_verification=True`, a `NO` verdict can trigger re-routing to another agent. This gate is opt-in and fail-open (unavailable `ctx` returns `True`).

```python
# Default configuration — fully LLM-driven
router = AgentRouter(agents)                    # uses LLMRouter() by default

# Optional verification gate
router = AgentRouter(agents, strategy=LLMRouter(enable_verification=True))

# Inside the agent tool:
selected = await router.aresolve(message=msg, ctx=ctx)   # Plan
# ... stream from adapter ...                              # Execute
ok = await router.strategy.averify(msg, response, ctx=ctx)  # Verify
```

**Key invariants:**
- `AgentRouter.resolve()` (sync) raises for multi-agent routing — callers must use `aresolve()`
- Single-agent setups and explicit `name=` targets skip LLM entirely (zero sampling overhead)
- `_match_agent()` performs exact → case-insensitive → substring matching to handle LLM verbosity
- `AgentNotFoundError` is raised (not swallowed) when the LLM names an unknown agent

**Future extension — sampling with tool loops**: FastMCP 3.0's `ctx.sample()` supports a `tools` parameter and `tool_choice`, enabling a deeper orchestration loop where the LLM can call gateway tools during routing — for example, calling `inspect` on a candidate agent before committing to it, or calling `agents` to dynamically discover new capabilities. The current Plan→Verify loop is the first step; a full agentic planning loop is a natural next evolution.

---

### Pillar 2 — Policy-Driven Visibility: Auto-Configuration

**Before**: `AgentVisibility` required explicit manual calls from the MCP client (`enable_agent`, `disable_agent` tools) to control which agents were visible per session. Multi-tenant configuration meant the client had to know about the policy and execute the right tool calls at session start.

**After**: Visibility is **automatically applied** at the gateway boundary based on session metadata. No client cooperation is required. The policy is configured once server-side; the gateway enforces it transparently.

**Three-tier control model:**

1. **Server-level static visibility** — `AgentVisibility.apply()` registers a FastMCP `Visibility` transform that sets the baseline for all sessions.

2. **Policy-driven session visibility** — `AgentVisibility.configure_policy()` defines a `VisibilityPolicy`: a mapping of tenant identifier → allowed agent names. At the start of each `agent` tool invocation, `apply_session_policy()` reads the tenant ID from session metadata and calls `ctx.enable_components()` / `ctx.disable_components()` to configure exactly the right set of agents for that session.

3. **FastMCP middleware filtering** — `TenantVisibilityMiddleware` intercepts every `list_tools` request and filters the tool list to only include tools belonging to the tenant's permitted agents. This is the strongest enforcement layer: the client never even sees tools it cannot use.

```python
vis = AgentVisibility()
vis.configure_policy(
    tenant_header="X-Tenant-ID",
    tenant_agents={
        "acme":   ["billing-agent", "crm-agent"],
        "beta":   ["analytics-agent"],
    },
    default_visible=False,   # deny-by-default for unlisted tenants
)
server = create_server(agents=agents, visibility=vis)
```

Header extraction follows a three-step lookup: `context.state` dict → HTTP `request.headers` → `ctx.get_state()` coroutine, with raw, lowercase, and snake_case variants tried at each step.

**Future extension — dynamic policy from agent cards**: Rather than statically defining `tenant_agents`, the gateway could fetch extended agent cards (authenticated, per-tenant) and derive the visibility policy from the card's declared capabilities. An agent card might include `metadata.tenant_tags`, and the gateway would apply `tag_based=True` visibility automatically. This ties agent self-declaration to access control.

---

### Pillar 3 — Transport Refactoring: Streamable HTTP

**Before**: `A2AAgentAdapter` contained SSE-specific terminology: `retry_on_disconnect`, `max_resubscribe_attempts`, comments about "SSE disconnection", and a `ConnectionError | OSError` catch tied to EventSource behaviour.

**After**: The adapter speaks transport-agnostically. SSE-specific language is replaced with generic HTTP stream resilience semantics.

| Before | After |
|---|---|
| `retry_on_disconnect: bool` | `enable_reconnect: bool` |
| `max_resubscribe_attempts: int` | `max_reconnect_attempts: int` |
| `except (ConnectionError, OSError)` | `except (ConnectionError, OSError, asyncio.TimeoutError, EOFError)` |

The underlying resubscription logic is unchanged — when a task ID is known and the stream is interrupted, the adapter calls `client.resubscribe(TaskIdParams(id=task_id))` to resume. This is an A2A protocol feature (`tasks/resubscribe`), not an SSE feature.

**Transport priority**: Callers configure the A2A SDK's `ClientConfig` with `supported_transports` to prefer Streamable HTTP and gRPC over legacy JSONRPC. Agentique's adapter is fully transport-agnostic — the A2A SDK handles negotiation.

**Future extension — gRPC as first-class transport**: The A2A SDK now includes `GrpcTransport`. For high-throughput agent deployments, gRPC offers lower latency and bidirectional streaming without HTTP overhead. The gateway should expose `grpc_channel_factory` in `AgentiqueConfig` to allow users to configure gRPC channels with custom credentials, interceptors, and load balancing — all of which are already supported in `A2AClientPool` but not yet surfaced in `AgentiqueConfig` or `create_server()`.

---

### Pillar 4 — Artifacts as First-Class Resources

**Before**: Artifact events from agents were treated identically to message events — their text was appended to the response string and lost after the streaming loop.

**After**: When an agent emits an `artifact` event, the gateway captures it in `TaskManager` and registers it as an MCP Resource with a stable URI.

**MCP Resource URI scheme:**
```
a2a://{task_id}/artifacts/{artifact_id}    — raw content
a2a://{task_id}/artifacts                  — task artifact catalog
```

During streaming, `ctx.info()` notifies the client immediately when an artifact is ready:
```python
if chunk.kind == "artifact" and chunk.text:
    uri = _tasks.capture_artifact(task_id, art_id, chunk.text, name=event.artifact_name)
    await ctx.info(f"Artifact available: {uri}")
```

**`TaskManager` artifact API:**
```python
uri     = tasks.capture_artifact(task_id, artifact_id, content, name="report.md")
content = tasks.get_artifact(task_id, artifact_id)
meta    = tasks.get_artifact_metadata(task_id, artifact_id)
listing = tasks.list_artifacts(task_id)
```

**Future extension — persistent artifact stores**: Artifacts are currently in-memory, lost on server restart. Production deployments need the artifact store backed by the same pluggable `TaskStore` abstraction (Redis, DynamoDB). The MIME type field (`mime_type`) is already captured; full binary artifact support (images, PDFs, code files) requires extending the resource handler to negotiate content type and stream binary data, which is something FastMCP 3.0's resource model supports.

---

## FastMCP 3.0 capabilities not yet fully exploited

Several FastMCP 3.0 features are partially used but could be pushed much further within the gateway model.

### Component transforms beyond Visibility

FastMCP 3.0's `Transform` pipeline supports `Namespace`, `Visibility`, `ToolTransform`, `ResourcesAsTools`, and `PromptsAsTools`. The gateway currently uses `Namespace` and `Visibility`. The remaining transforms offer interesting gateway scenarios:

- **`ToolTransform`**: Rename verbose agent skills into concise, client-friendly tool names. An A2A agent card might expose `billing_agent_generate_invoice_v2`; a `ToolTransform` can present it as `create_invoice` without touching the adapter.
- **`ResourcesAsTools`**: Expose MCP resources (including artifact resources) as tools, letting clients that cannot read resources directly still access artifact content through a tool call.
- **`PromptsAsTools`**: Agents with rich prompt libraries can expose their prompts as tools, letting clients compose complex workflows from agent-defined prompt templates.

These are zero-code additions to `create_server()` — users can pass them via the `transforms` parameter today.

### Composition via `mount()` for multi-protocol gateways

FastMCP's `mount()` enables separate bridges — one for A2A agents, one for HTTP agents, one for local agents — composed under a single MCP server with namespace isolation. The `mount_bridge()` helper already exists in `server.py`. The missing piece is a higher-level configuration DSL that lets users declare protocol-per-namespace without writing Python:

```python
# Desired: declarative multi-protocol gateway
server = create_server(
    mounts=[
        BridgeMount(namespace="a2a",  adapter=A2AAgentAdapter(a2a_agents)),
        BridgeMount(namespace="http", adapter=HTTPAdapter(http_agents)),
        BridgeMount(namespace="mcp",  adapter=MCPProxyAdapter("http://remote-mcp")),
    ]
)
```

This is particularly powerful for the `create_proxy()` pattern — wrapping a remote MCP server behind the gateway gives it LLM routing, visibility policies, and artifact registration for free.

### Lifespan composition for agent connection management

FastMCP 3.0's lifespan pipe operator (`lifespan_a | lifespan_b`) enables composing startup/shutdown sequences. The gateway should use this to manage A2A client pool lifecycle, prefetch agent cards on startup, and warm up gRPC channels — all composable without manual ordering. Currently `AgentProvider.lifespan()` handles prefetching; a full lifespan composition would tie `A2AClientPool.close()`, health monitor teardown, and webhook receiver shutdown into a single composable sequence.

### Structured tool output with `outputSchema`

The `agent` tool currently returns a plain string. MCP's `outputSchema` / `structuredContent` allows tools to declare a schema for their return value, giving clients typed access to the response. The `agent` tool should return a structured payload:

```python
# Current
return "".join(text_parts)

# Target — with structured content
return ToolResult(
    content="".join(text_parts),
    structured_content={
        "task_id": task_id,
        "agent": resolved.name,
        "state": tracker.state.value,
        "artifact_uris": [f"a2a://{task_id}/artifacts/{a['artifact_id']}"
                          for a in tasks.list_artifacts(task_id)],
        "response": "".join(text_parts),
    },
)
```

This gives clients a machine-readable response they can use to immediately query artifacts, track tasks, and chain to subsequent calls — closing the loop between the gateway's resource model and its tool surface.

---

## A2A protocol evolution targets

### The bidirectional bridge: MCP ↔ A2A ↔ ADK

Google ADK's dual-direction integration is architecturally important for the gateway's positioning. `McpToolset` consumes MCP servers as ADK tools; `to_a2a()` exposes ADK agents as A2A servers. Agentique sits at the complementary position: consuming A2A agents as MCP tools. Together these create a full bidirectional stack:

```
MCP Client → Agentique Gateway → A2A Agent (ADK) → McpToolset → other MCP Servers
```

An Agentique gateway in this stack is not just a translator — it is the policy and intelligence layer for the entire chain. Routing decisions made at the gateway propagate through the entire ADK agent's tool graph.

### A2A v1.0 readiness under Linux Foundation stewardship

A2A's donation to the Linux Foundation signals long-term protocol stability and multi-vendor adoption. The v1.0 specification is expected to formalise the three-binding approach (JSON-RPC, gRPC, REST) and the extensions mechanism. Agentique should track these changes in `adapter.py` and `client.py` rather than absorbing them at higher layers — the adapter is the correct isolation boundary.

The remaining v0.3 → v1.0 gaps to close in the gateway:

- **Typed error codes**: A2A defines specific error codes (`-32001` TaskNotFound, `-32002` ContentTypeNotSupported, `-32003` UnsupportedOperation). The `ErrorMappingMiddleware` partially handles these, but the mapping should be complete and documented — especially for `-32002` and `-32003`, which have no current handler.
- **Context ID contract enforcement**: A2A mandates that agents reject messages with mismatching `contextId` and `taskId`. The gateway's `ContextManager` tracks the mapping but does not actively validate it on outgoing messages. Adding this validation would prevent hard-to-debug A2A errors from propagating as opaque failures.
- **`auth-required` → MCP elicitation**: The `auth-required` A2A state currently emits a `ctx.warning()`. The correct MCP response is `ctx.elicit()` with a structured authentication prompt — the gateway should turn the A2A auth challenge into a first-class MCP interaction.

### Push notifications for long-running agent tasks

Push notifications are critical for production deployments where backend agents run for minutes or hours. The current implementation configures push notifications on the A2A server and receives them via `WebhookReceiver`. The missing piece is the MCP side of this loop: when a push notification arrives at the gateway, it should emit an MCP `notifications/tasks/status` event (when MCP Tasks support stabilises) or at minimum write the update into the `TaskManager` store so clients polling `task(id=...)` see current state.

The JWT signing of push notification payloads (via the `token` field in `PushNotificationConfig`) should also be validated on receipt in `WebhookReceiver` — currently the token is generated but not verified.

---

## Expanding the adapter ecosystem (Phase 4)

The adapter layer is the gateway's extensibility surface. Everything above the adapter — routing, visibility, artifact registration, middleware — is protocol-agnostic. The A2A adapter proves the pattern; the second adapter validates it.

### The ToolMapper protocol for agent card translation

The `A2ACardParser` maps agent cards to MCP components (tools, resources, prompts). This mapping is currently fixed. A `ToolMapper` protocol would let users control how agent capabilities become MCP primitives:

```python
class ToolMapper(Protocol):
    def map_tools(self, agent: AgentInfo, card: Any) -> list[ToolDefinition]: ...
    def map_resources(self, agent: AgentInfo, card: Any) -> list[ResourceDefinition]: ...
    def map_prompts(self, agent: AgentInfo, card: Any) -> list[PromptDefinition]: ...
```

A default implementation creates one tool per agent (current behaviour). Alternative implementations could create one tool per skill (exposing agent granularity), flatten sub-agent hierarchies (collapsing nested agent graphs), or apply custom naming conventions (removing version suffixes, normalising casing). This is the highest-leverage extensibility point for enterprise deployments with complex agent card schemas.

### Entry point discovery for third-party adapters

The adapter registry pattern (already present as `register_adapter` / `discover_adapters`) should be backed by setuptools entry points:

```toml
# Third-party adapter (pyproject.toml)
[project.entry-points."agentique.adapters"]
openai = "agentique_openai:OpenAIAgentsAdapter"
langchain = "agentique_langchain:LangChainAdapter"
```

Discovery becomes automatic: `pip install agentique-openai` makes the adapter available without any configuration change. This mirrors pytest's pluggy-based discovery and is the correct pattern for a framework that wants to be genuinely unopinionated about backends.

### Test utilities for adapter and gateway validation

Every production use of Agentique requires testing routing decisions, middleware behaviour, and artifact capture. The framework should provide:

```python
# Protocol compliance: any class with matching methods satisfies AgentAdapter
from agentique.testing import MockAdapter, assert_adapter_protocol

def test_custom_adapter_satisfies_protocol():
    adapter = MyCustomAdapter(config)
    assert_adapter_protocol(adapter)  # validates runtime_checkable Protocol

# Integration: FastMCP in-process client
from agentique.testing import InMemoryBridge

async def test_routing_selects_correct_agent():
    bridge = InMemoryBridge(agents=[coding_agent, research_agent])
    async with bridge.client() as client:
        result = await client.call_tool("agent", {"message": "write a Python function"})
        assert "coding" in result.structured_content["agent"]
```

The `runtime_checkable` Protocol on `AgentAdapter` already makes structural validation possible — the test utility just needs to surface this cleanly.

---

## Design principles

### Intelligence is always-on, never optional

The system assumes an LLM is always available for routing. Removing keyword-matching fallbacks is not a loss of functionality; it is a gain in honesty. Deterministic keyword routing produced misleading confidence in routing quality. LLM routing is inherently higher quality and its failure mode (no `ctx.sample()` support) is explicit: `RuntimeError` rather than silent wrong routing.

### Policy enforcement is automatic, not delegated

Multi-tenant visibility used to rely on clients calling `enable_agent` / `disable_agent` tools at session start. The new model enforces policy at the gateway boundary, transparently. Clients cannot bypass it.

### Transports are negotiated, not hardcoded

The adapter is transport-agnostic. The A2A SDK handles transport negotiation. Adding a new transport requires no changes above the adapter layer.

### Artifacts are resources, not ephemeral text

Agents produce valuable artifacts — reports, generated files, structured outputs. Losing them at the end of a streaming call is a failure of the protocol contract. MCP resources with stable URIs make agent-produced content a durable, addressable part of the gateway's capability surface.

### The Protocol class is the extensibility boundary

Every public interface — `AgentAdapter`, `RoutingStrategy`, `ToolMapper`, `TaskStore`, `BridgeMiddleware` — is a `typing.Protocol` with `@runtime_checkable`. Users extend the gateway by implementing these protocols, not by subclassing internal classes. This structural subtyping approach (no forced inheritance) is the correct pattern for a framework that will have third-party ecosystem adapters it cannot anticipate.

---

## Lessons from the framework landscape

Analysis of LangChain, CrewAI, AutoGen, Semantic Kernel, and PydanticAI reveals consistent patterns Agentique should adopt and specific anti-patterns to avoid.

**PydanticAI's `AbstractToolset` is the closest model.** Its `get_tools()` and `call_tool()` mirror exactly what Agentique's adapter layer needs. PydanticAI's generic typing (`Agent[DepsT, OutputT]`) ensures full type safety through the pipeline. Its `MCPServer` and `FastMCPToolset` demonstrate clean MCP integration. Agentique occupies the complementary position of PydanticAI's `FastA2A`.

**Semantic Kernel's Kernel-as-DI-container** pattern maps directly onto Agentique's server factory. The `create_server()` function is Agentique's composition root — the place where adapters, middleware, transforms, routing strategy, and visibility policy converge. It should be designed with the same care as SK's `Kernel`: every dependency injectable, every component replaceable.

**LangChain's Runnable interface** shows the power of a universal composable unit. Every agent in Agentique's registry should support both `send_message()` (collect) and `stream_message()` (yield) — the async equivalents of `.invoke()` and `.stream()`. The `AgentAdapter` protocol already enforces this; it should be the invariant that gates all adapter acceptance.

**What to avoid:**
- CrewAI's role-based abstractions — too opinionated for a gateway that must be backend-agnostic
- AutoGen's conversation-as-workflow model — powerful for collaboration, wrong abstraction for a protocol gateway
- LangChain's proliferating callback handlers — Agentique should have at most 8-10 well-defined lifecycle events: `task.created`, `task.completed`, `task.state_changed`, `tool.called`, `tool.completed`, `tool.failed`, `stream.chunk`, `agent.discovered`, `agent.lost`, `error`

**The callback system pattern** appears in every framework. Agentique's `AsyncEventEmitter` covers this correctly. The remaining gap is making it easy to subscribe from outside the server factory — users should be able to call `server.on("task.completed", my_handler)` rather than passing the emitter instance through `create_server()`.

---

## Protocol convergence: MCP × A2A

### MCP Tasks alignment

MCP's experimental Tasks specification (November 2025) adds asynchronous, long-running operations with states (`working`, `input_required`, `completed`, `failed`, `cancelled`) and `notifications/tasks/status`. This maps almost perfectly to A2A's task lifecycle:

| A2A `TaskState` | MCP Task state |
|---|---|
| `submitted` | `working` |
| `working` | `working` |
| `input_required` | `input_required` |
| `completed` | `completed` |
| `canceled` | `cancelled` |
| `failed` | `failed` |
| `auth_required` | `input_required` (with auth elicitation) |

The gateway's `TaskManager` already models these states. When FastMCP stabilises its Tasks support, the `agent_background` tool should emit native MCP task notifications rather than requiring clients to poll the `task(id=...)` tool.

### `agentique://` extension definition

MCP's extensions framework (2025-11-25) enables optional, independently versioned protocol extensions. Agentique should define its own extension URI that carries gateway metadata through MCP interactions:

```python
# Extension URI: agentique://gateway/v1
# Payload:
{
    "agent_protocol": "a2a",
    "original_task_id": "...",
    "agent_card_url": "http://agent:9000/.well-known/agent-card.json",
    "gateway_version": "0.4.0",
    "routing_strategy": "llm",
}
```

This allows MCP clients aware of the extension to display richer metadata about which agent handled a request and why, enabling debuggability and audit logging at the client layer.

### MCP structured output for the `agent` tool

The `agent` tool should return a structured payload alongside its text content:

```python
ToolResult(
    content="".join(text_parts),
    structured_content={
        "task_id": task_id,
        "agent": resolved.name,
        "state": tracker.state.value,
        "artifact_uris": [...],
        "response": "".join(text_parts),
    },
)
```

This gives clients a machine-readable response envelope that enables: immediate artifact retrieval, task state tracking, chaining to subsequent calls, and audit logging — all without parsing free-form text.

---

## Architecture diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  MCP Client (Claude, Cursor, ChatGPT, custom)                        │
│  - Has sampling capability (required for LLM routing)                │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  MCP Protocol (Streamable HTTP / stdio)
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastMCP Server + Middleware Stack                                    │
│                                                                       │
│  ┌─────────────────────┐  ┌──────────────────────────────────────┐   │
│  │ AgentiqueMiddleware  │  │ TenantVisibilityMiddleware           │   │
│  │ (logging, OTel,     │  │ (filters list_tools by X-Tenant-ID)  │   │
│  │  event emission)    │  │                                      │   │
│  └─────────────────────┘  └──────────────────────────────────────┘   │
│                                                                       │
│  Core Tools:                    Resources:                            │
│  ● agent     (LLM-routed)       ● a2a://agents                       │
│  ● agents    (list registry)    ● a2a://{task}/artifacts/{id}        │
│  ● task      (query state)      ● a2a://{task}/artifacts             │
│  ● inspect   (agent hierarchy)  ● a2a://agents/{name}                │
│  ● agent_background (optional)  + enable_agent / disable_agent tools │
│                                                                       │
│  Transforms (composable):                                             │
│  ● Namespace  ● Visibility  ● ToolTransform  ● ResourcesAsTools      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐
  │  LLMRouter       │  │ TaskManager  │  │ Context      │
  │  aselect() PLAN  │  │ + Artifacts  │  │ Manager      │
  │  averify() VERIFY│  │   capture()  │  │ (session ↔   │
  │  (ctx.sample())  │  │   get()      │  │  context IDs)│
  └────────┬─────────┘  └──────┬───────┘  └──────────────┘
           │                   │
           └────────┬──────────┘
                    │
                    ▼
     ┌──────────────────────────────────────┐
     │  MiddlewareChain (Bridge Layer)      │
     │  Logging → ErrorMapping →            │
     │  RateLimit → Metrics → custom        │
     └──────────────────────────────────────┘
                    │
                    ▼
     ┌──────────────────────────────────────┐
     │  A2AAgentAdapter                     │
     │  Streamable HTTP / gRPC / JSON-RPC   │
     │  enable_reconnect=True               │
     │  tasks/resubscribe on disruption     │
     │  push notification config            │
     │  extended card support               │
     └──────────────────────────────────────┘
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
  A2A Agent    A2A Agent    A2A Agent
  (gRPC)     (HTTP+JSON)  (JSON-RPC)
     │
     ▼
  ADK Agent → McpToolset → other MCP Servers
  (bidirectional loop back into any MCP gateway)
```

---

## Public API changes (Phase 3.5)

### Removed

| Symbol | Reason |
|---|---|
| `KeywordRouter` | Replaced by LLM routing entirely |
| `WeightedKeywordRouter` | Replaced by LLM routing entirely |
| `LLMRouter.fallback` parameter | Hard-deprecated; no deterministic fallback |
| `A2AAgentAdapter.retry_on_disconnect` | Renamed to `enable_reconnect` |
| `A2AAgentAdapter.max_resubscribe_attempts` | Renamed to `max_reconnect_attempts` |

### Added

| Symbol | Description |
|---|---|
| `VisibilityPolicy` | Dataclass defining tenant → agent visibility rules |
| `TenantVisibilityMiddleware` | FastMCP middleware enforcing `list_tools` filtering |
| `RoutingStrategy` | Protocol type for custom routing strategies |
| `AgentVisibility.configure_policy()` | Configure tenant policy on an existing instance |
| `AgentVisibility.apply_session_policy()` | Apply policy at session init from metadata |
| `AgentVisibility.build_tenant_middleware()` | Build `TenantVisibilityMiddleware` from policy |
| `LLMRouter.averify()` | VERIFY phase: validate response via `ctx.sample()` |
| `AgentRouter.strategy` property | Expose the active routing strategy |
| `TaskManager.capture_artifact()` | Store an artifact and return its MCP Resource URI |
| `TaskManager.get_artifact()` | Retrieve artifact content by task + artifact ID |
| `TaskManager.get_artifact_metadata()` | Full artifact payload including MIME type |
| `TaskManager.list_artifacts()` | List all artifact metadata for a task |
| `create_server(visibility=...)` | Wire visibility policy into server at creation time |

---

## Roadmap

**Phase 1 — Foundation** ✅ Complete
Layered package structure, `AgentAdapter` protocol, `AgentiqueConfig`, `AsyncEventEmitter`, FastMCP Middleware, `Depends()` injection.

**Phase 2 — FastMCP 3.0 deep integration** ✅ Complete
Transforms, Visibility, TaskConfig, structured output, Elicitation, OTel, storage backends, session-scoped DI, `mount()` composition.

**Phase 3 — A2A protocol completeness** ✅ Complete
Push notifications, task resubscription, extended agent cards, auth states, gRPC transport, A2A extensions, context ID mapping.

**Phase 3.5 — Gateway intelligence** ✅ Complete
Pure LLM routing (Plan→Execute→Verify), policy-driven visibility auto-configuration, Streamable HTTP transport framing, artifacts as first-class MCP Resources.

**Phase 4 — Ecosystem and capability depth** (next)

*Gateway polish:*
- Structured `outputSchema` on `agent` tool (artifact URIs in response envelope)
- `agentique://` MCP extension definition for audit metadata
- Complete A2A error code mapping (`-32002`, `-32003`)
- `auth-required` → `ctx.elicit()` flow
- JWT validation in `WebhookReceiver`
- Persistent artifact store (backed by `TaskStore` abstraction)

*FastMCP depth:*
- `ToolTransform` / `ResourcesAsTools` / `PromptsAsTools` in `create_server()` helpers
- Full lifespan composition (pipe operator across pool, health monitor, webhook receiver)
- `ctx.sample(tools=[...])` for agentic planning loops in the router
- MCP native task notifications via `notifications/tasks/status`

*Adapter ecosystem:*
- `ToolMapper` protocol with pluggable implementations (per-skill, flat-hierarchy, custom naming)
- Entry point discovery for third-party adapters (`agentique.adapters` entry point group)
- Second adapter (OpenAI Agents API or generic HTTP REST) to validate protocol abstraction
- `agentique.testing` module: `MockAdapter`, `InMemoryBridge`, `assert_adapter_protocol`

*Packaging and distribution:*
- `agentique-core` as a separate zero-dependency package (protocols + types only)
- PyPI packaging, versioned releases, changelog
- CI/CD pipeline with adapter compliance test matrix
- Documentation site

---

## Conclusion

Agentique has evolved from a transparent protocol bridge into an **Intelligent Agent Gateway**. The three defining properties of the gateway model are:

1. **Intelligence is structural, not optional** — LLM routing is the default, not a feature flag. The gateway leverages the client's LLM as a first-class decision engine for every routing choice, and optionally for response verification too.

2. **Policy is enforced at the boundary** — Tenant isolation, access control, and session configuration happen automatically at the gateway. The architecture is zero-trust toward clients regarding policy compliance.

3. **Agent outputs become platform resources** — Artifacts are not ephemeral side effects. They are MCP Resources with stable URIs, making agent-produced content a durable, addressable part of the gateway's capability surface.

The deeper insight is that the gateway model changes the *unit of composition*. In the bridge model, the unit was a message: send one in, receive one out. In the gateway model, the unit is a **session**: the gateway maintains state (task registry, artifact store, context mapping, visibility policy) across the entire session lifetime, and every interaction enriches that state. The MCP client's LLM can reason over that accumulated state — querying artifacts, inspecting agent hierarchies, tracking task progress — making the gateway a shared memory space for human-agent collaboration.

This is the architecture that makes Agentique genuinely valuable at scale: not merely connecting protocols, but providing the intelligence, policy, and persistence layer that turns a collection of independent agents into a coherent, secure, and observable platform.
