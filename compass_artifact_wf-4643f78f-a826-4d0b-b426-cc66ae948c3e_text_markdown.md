# Agentique: architectural extensions and evolution paths across three interconnected systems

**Agentique sits at a uniquely powerful intersection — a gateway that bridges MCP's tool-centric client model with A2A's agent-centric communication model — and both underlying SDKs have evolved substantially since Agentique's initial design.** FastMCP 3.0 now offers a full Provider/Transform/Middleware pipeline with server-side agentic loops via `ctx.sample()`, while the A2A protocol has matured from v0.3 to a v1.0 Release Candidate under Linux Foundation governance with gRPC transport, formal security schemes, and a three-layer specification architecture. Meanwhile, MCP itself has added experimental Tasks, an Extensions framework, and Sampling with Tools — all features that directly expand Agentique's capability envelope. This analysis maps every meaningful extension surface, identifies structural gaps in the current architecture, and proposes a phased evolution grounded in what the code and specifications actually support.

---

## The protocol foundation has shifted beneath the gateway

Both MCP and A2A have undergone significant specification evolution that creates new extension surfaces for Agentique. Understanding these shifts is prerequisite to any architectural planning.

**FastMCP 3.0's Provider protocol** is the most consequential primitive for Agentique. The `Provider` base class defines seven async methods (`list_tools`, `get_tool`, `list_resources`, `get_resource`, `list_prompts`, `get_prompt`, `list_resource_templates`) that source components from any backing store. Agentique's `AgentProvider` already implements this interface, but FastMCP 3.0 ships **seven built-in providers** — `LocalProvider`, `FileSystemProvider` (with hot-reload), `OpenAPIProvider`, `ProxyProvider`, `FastMCPProvider`, and `SkillsProvider` — any of which could be composed alongside `AgentProvider` via `AggregateProvider`. The two-level transform system (provider-level then server-level) means each A2A agent adapter could carry its own `Namespace` transform while server-level `Visibility` transforms enforce tenant policy.

**The Transform pipeline** deserves special attention. Built-in transforms include `Namespace` (prefix isolation), `ToolTransform` (rename/redescribe/re-tag), `VersionFilter` (version-range gating), `Visibility` (blocklist/allowlist by tag/name/version), `ResourcesAsTools`, and `PromptsAsTools`. For Agentique, this means the current `ToolMapper` protocol (Default/PerSkill/FlatHierarchy) could be **reimplemented as composable Transform chains** rather than standalone mapper implementations — gaining composition, stacking, and interoperability with all other transforms for free.

**A2A's v1.0 RC** introduces breaking structural changes that Agentique must account for. The specification now uses a **three-layer architecture** — Data Model, Abstract Operations, Protocol Bindings — with Protocol Buffers as the normative source. The `kind` discriminator has been removed from data objects. A new `TASK_STATE_AUTH_REQUIRED` state enables in-task authentication escalation. `ListTasks` with filtering/pagination is now a formal operation. Security schemes have been overhauled to an OpenAPI-aligned model with `APIKeySecurityScheme`, `HTTPAuthSecurityScheme`, `OAuth2SecurityScheme` (including `DeviceCodeOAuthFlow` for CLI tools), `OpenIdConnectSecurityScheme`, and `MutualTlsSecurityScheme`. The `A2A-Version` and `A2A-Extensions` HTTP headers formalize protocol negotiation.

**MCP's November 2025 specification** added three features that directly affect Agentique's design space. First, **MCP Tasks** (experimental, SEP-1686) are durable state machines that augment any request with async tracking — tools declare `taskSupport` as `"required"`, `"optional"`, or `"forbidden"`, and task creation returns a `CreateTaskResult` with `taskId`, `status`, `pollInterval`, and `ttl`. Second, the **Extensions framework** (SEP-1724) enables capability negotiation via `capabilities.extensions` during initialization, using reversed-domain identifiers (`io.modelcontextprotocol/oauth-client-credentials`). Third, **Sampling with Tools** (SEP-1577) allows MCP servers to include `tools` and `toolChoice` in `sampling/createMessage` requests, enabling server-side multi-turn tool loops — the exact mechanism FastMCP 3.0 wraps with `ctx.sample(tools=[...])`.

### Structural comparison reveals precise interoperability boundaries

The MCP-A2A structural mapping is not 1:1, and the mismatches define Agentique's translation responsibilities:

| Concept | MCP | A2A | Gateway translation required |
|---------|-----|-----|------------------------------|
| Task creation | Request augmentation (`task` param on any method) | Dedicated `message/send` endpoint | Agentique must synthesize MCP task params from A2A task responses |
| Task states | `working`, `input_required`, `completed`, `failed`, `cancelled` | `submitted`, `working`, `input-required`, `completed`, `failed`, `canceled`, `rejected`, `auth-required` | Map `submitted`→implicit, `rejected`→`failed` with metadata, `auth-required`→elicitation flow |
| Streaming | SSE via Streamable HTTP transport | SSE events (`TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`) | Bridge A2A SSE events to MCP progress notifications |
| Structured output | `outputSchema` + `structuredContent` on tools | Artifacts with MIME-typed `Part` objects | Convert A2A artifacts to MCP `structuredContent` when schema available |
| User input | Elicitation (`elicitation/create` with form/URL mode) | `input-required` task state | Translate `input-required` to `elicitation/create`; relay response as new `message/send` |
| Auth | OAuth 2.1 + extensions + incremental scope | SecuritySchemes (API Key, HTTP, OAuth2, OIDC, mTLS) | Manage credential lifecycle per A2A agent; map auth failures to elicitation |
| Discovery | `tools/list`, `resources/list`, `prompts/list` | Agent Card at `/.well-known/agent-card.json` | Parse Agent Cards → synthesize MCP tool/resource/prompt registrations |
| Session | `MCP-Session-Id` header, capability negotiation | `contextId` (server-generated, groups tasks) | `ContextManager` maps MCP sessions ↔ A2A context IDs |
| Extensions | `capabilities.extensions` negotiation | `AgentExtension` in Agent Card + `metadata` keyed by URI | Passthrough or translate extension metadata between protocols |

**The deepest structural mismatch** is in the communication model itself. MCP is fundamentally **tool-centric** — clients invoke named tools with typed inputs and receive typed outputs. A2A is fundamentally **message-centric** — clients send natural-language messages to agents that interpret intent and return artifacts. Agentique's LLM-first routing bridges this gap by using `ctx.sample()` to interpret tool invocations as intent, select appropriate A2A agents, and translate message responses back to tool results. This is not a limitation but a **core architectural feature** — the intelligence layer is necessary, not optional.

---

## Where extension surfaces align with design principles

Agentique's four design principles — LLM-first routing, automatic policy enforcement, protocol-agnostic adapters, artifacts as resources — create a filter for evaluating extension candidates. Extensions that reinforce these principles are architecturally coherent; those that contradict them require careful justification.

### Server-side agentic loops via ctx.sample() with tools

FastMCP 3.0's `ctx.sample()` API is the single most impactful underutilized capability. The full signature accepts `tools` (list of Python callables), `result_type` (Pydantic model for structured output validation), `tool_choice` ("auto"/"required"/"none"), and `tool_concurrency` (sequential/bounded/unlimited). This enables a **Plan→Execute→Verify loop** where the LLM reasons about which tools to call, FastMCP executes them, and the loop continues until a structured result is produced.

For Agentique, this means the current LLM routing could evolve from single-shot dispatch to **multi-step orchestration**. The `tools` parameter could include not just the A2A agent dispatch function but also introspection tools (query agent capabilities, check task status, retrieve partial artifacts), planning tools (decompose complex requests into sub-tasks), and verification tools (validate outputs against schemas, check policy compliance). The `result_type` parameter enables typed routing decisions:

```python
class RoutingDecision(BaseModel):
    agent_id: str
    message: str
    confidence: float
    fallback_agents: list[str]
    requires_decomposition: bool
```

`ctx.sample_step()` provides even finer control — single LLM turns with `execute_tools=False` for manual tool execution, enabling the gateway to intercept tool calls, apply policy, and decide whether to proceed. This is particularly valuable for implementing **approval workflows** where high-risk operations require human confirmation via elicitation before execution.

**Sampling fallback handlers** (`OpenAISamplingHandler`, `AnthropicSamplingHandler` with `sampling_handler_behavior="fallback"`) ensure Agentique can function even when MCP clients don't support sampling — the gateway falls back to its own configured LLM.

### Transform-based dynamic tool surface management

The current `ToolMapper` protocol with three implementations (Default, PerSkill, FlatHierarchy) maps A2A agent skills to MCP tool names. This could be **replaced by or composed with** FastMCP 3.0's transform system for substantially more flexibility:

- **`Namespace("agent_name")`** on each `A2AAgentAdapter`'s provider replaces the Default mapper's `{agent}_{skill}` pattern
- **`ToolTransform`** enables per-deployment tool curation — renaming tools, updating descriptions with context-specific information, adding tags for visibility filtering
- **`Visibility` transforms** with `mcp.enable(tags={"tier_1"}, only=True)` replace the current `VisibilityPolicy`/`TenantVisibilityMiddleware` with FastMCP-native filtering, gaining tag-based, name-based, and version-based filtering
- **`ResourcesAsTools`** and **`PromptsAsTools`** enable Agentique to expose its artifact resources and any prompt templates as tools for clients that only support the tools primitive

This is not merely a refactoring suggestion — transforms **compose and stack**, meaning tenant visibility, namespace isolation, version gating, and tool curation can all operate simultaneously with correct ordering guaranteed by FastMCP's pipeline.

### MCP Tasks alignment creates true async bridging

The MCP Tasks specification (experimental) and FastMCP's `TaskConfig` create a direct alignment path with A2A's task lifecycle. Currently, Agentique's `TaskManager` captures A2A task state and exposes it via the `task` MCP tool. With MCP Tasks support, this becomes native:

- Tools backed by A2A agents declare `task=TaskConfig(mode="optional")` or `mode="required"` for long-running agents
- MCP clients receive `CreateTaskResult` with `taskId` and `pollInterval` instead of blocking
- A2A `TaskStatusUpdateEvent` SSE events map to `notifications/tasks/status` MCP notifications
- A2A artifacts map to MCP task results retrieved via `tasks/result`
- The `input_required` state in both protocols maps naturally — A2A's `input-required` triggers MCP's `input_required` task state, which the client resolves via elicitation, which Agentique relays back as a new A2A `message/send`

The `io.modelcontextprotocol/related-task` metadata in MCP Task messages enables correlation — Agentique can tag all MCP messages with the corresponding A2A task ID for end-to-end traceability.

**FastMCP's Docket-backed background tasks** (SQLite or Postgres persistence, horizontal worker scaling) provide the infrastructure for durable task execution. This means Agentique could run A2A agent interactions as Docket tasks, surviving server restarts and enabling distributed processing across multiple gateway instances.

### Auth-required to elicitation flow

A2A v1.0 RC introduces `TASK_STATE_AUTH_REQUIRED` — when an A2A agent needs secondary credentials during task execution, it transitions to this state. MCP's elicitation (both form mode and URL mode) provides the client-facing mechanism to collect these credentials. The bridge flow:

1. A2A agent transitions task to `auth-required` with metadata indicating required auth scheme
2. Agentique detects state transition via SSE stream or push notification
3. Gateway parses the agent's `SecurityScheme` requirements from the Agent Card
4. For OAuth2: sends `elicitation/create` with `mode: "url"` pointing to the authorization endpoint
5. For API keys: sends `elicitation/create` with `mode: "form"` requesting the key
6. Client provides credentials → Agentique stores them in session state → resends message with credentials
7. A2A agent transitions back to `working`

This flow leverages MCP's URL-mode elicitation (v2025-11-25, experimental) for credential collection without credentials transiting through the gateway's memory longer than necessary. The `DeviceCodeOAuthFlow` in A2A v1.0 RC handles CLI/headless scenarios where redirect-based OAuth is impractical.

### A2A Extensions as gateway metadata channels

A2A's extension mechanism — URI-keyed metadata on messages, artifacts, and tasks — creates a **metadata passthrough channel** between MCP clients and A2A agents. Agentique could define gateway-specific extensions:

- `com.agentique/routing-metadata`: Carries routing decisions, confidence scores, fallback information
- `com.agentique/policy-context`: Carries tenant ID, visibility tier, rate limit budget remaining
- `com.agentique/mcp-session`: Carries MCP session context (session ID, client capabilities, negotiated extensions)
- `com.agentique/trace-context`: Carries OpenTelemetry trace context for end-to-end distributed tracing

The `A2A-Extensions` HTTP header (v1.0 RC) enables per-request extension activation, so Agentique can selectively enable extensions based on the A2A agent's declared support in its Agent Card.

---

## Lifecycle, middleware, and state patterns that shape feasible growth

### Lifespan composition governs resource management

FastMCP 3.0's lifespan pipe operator (`|`) enables clean composition of independent resource lifecycles. For Agentique, this means:

```python
server_lifespan = (
    a2a_client_pool_lifespan     # Initialize A2A client connection pool
    | health_monitor_lifespan     # Start health check loop
    | webhook_receiver_lifespan   # Start push notification receiver
    | redis_store_lifespan        # Connect to Redis for task/session storage
    | otel_lifespan               # Initialize OpenTelemetry providers
)
```

Lifespans enter in order and exit in reverse (LIFO), ensuring clean shutdown. The context dict merge means all tool handlers access shared resources via `ctx.fastmcp` — no global state needed. This pattern directly enables **feature-flag-driven lifecycle management**: optional components (gRPC transport, webhook receiver, metrics exporter) participate in the lifespan chain only when configured.

### Two-level middleware creates a cross-cutting concern architecture

FastMCP 3.0 middleware operates on **requests** (tool calls, resource reads) while transforms operate on **components** (tool definitions, resource lists). Agentique's existing `MiddlewareChain` (ErrorMapping, Logging, RateLimit, Metrics) maps to FastMCP middleware, but the bridge layer adds a second concern axis:

**FastMCP level** (request pipeline):
- `AuthMiddleware` — validate MCP client credentials
- `LoggingMiddleware` / `StructuredLoggingMiddleware` — request/response logging
- `ResponseLimitingMiddleware` — cap response sizes for large artifacts
- `PingMiddleware` — keep-alive for long-running connections
- Custom rate limiting middleware per session/tenant

**Bridge level** (A2A dispatch pipeline):
- Error mapping (A2A error codes → MCP error responses)
- A2A credential injection (attach stored credentials per agent)
- Request transformation (MCP tool args → A2A message parts)
- Response transformation (A2A artifacts → MCP tool results/resources)
- Circuit breaking (health-aware routing around failed agents)

The interaction between these levels is critical. A **`ToolInjectionMiddleware`** at the FastMCP level could dynamically inject tools based on runtime conditions — for example, injecting a `request_credentials` tool when an A2A agent signals `auth-required`, or injecting `cancel_task` and `check_status` tools for active background tasks.

### Transport evolution paths

Three transport evolution paths are feasible, ordered by implementation complexity:

**Streamable HTTP (immediate)**: FastMCP 3.0 defaults to Streamable HTTP, which Agentique should adopt as primary. This provides single-endpoint serving, SSE streaming for server-initiated messages, session management via `MCP-Session-Id` header, and resumability via SSE event IDs. The `MCP-Protocol-Version` header enables version negotiation.

**gRPC for A2A backends (medium-term)**: A2A v1.0 RC promotes gRPC to a first-class transport binding with the normative `a2a.proto`. For high-throughput agent clusters, Agentique's `A2AAgentAdapter` could negotiate gRPC transport when the Agent Card declares `preferredTransport: "GRPC"` or includes a gRPC `AgentInterface`. The A2A Python SDK's `[grpc]` extra provides the client implementation.

**WebSocket for bidirectional streaming (exploratory)**: Neither MCP nor A2A currently specifies WebSocket transport, but IBM's ContextForge MCP Gateway already supports it. WebSocket would enable true bidirectional streaming for interactive agent sessions — particularly valuable for the `input-required` ↔ `elicitation` loop where latency matters.

### State unification opportunities

Agentique currently manages four distinct state types: task state (A2A task lifecycle), artifact state (captured outputs), session state (MCP session ↔ A2A context mapping), and visibility state (tenant policies). FastMCP 3.0's session state API (`ctx.get_state`/`ctx.set_state`/`ctx.delete_state`) with pluggable backends (Redis, DynamoDB, MongoDB) could unify these:

- **Task state**: Store A2A task metadata keyed by `task:{task_id}` in session state, with TTL matching A2A task lifecycle
- **Artifact state**: Store artifact references (not content) as `artifact:{task_id}:{artifact_id}`, with actual content served via MCP resource URIs
- **Session state**: Map `session:{mcp_session_id}` → `{context_id, tenant_id, auth_tokens, active_tasks}`
- **Visibility state**: Store tenant policies as `policy:{tenant_id}` with transform configurations

The unified store enables **cross-cutting queries** — listing all active tasks for a tenant, finding all artifacts in a session, checking rate limit budgets across sessions. The `serializable=False` option handles non-JSON objects (A2A client instances, gRPC channels) that must remain in-memory.

---

## The capability envelope and a phased evolution roadmap

### Theoretical capability boundaries

Agentique's capability envelope is bounded by the **intersection** of what MCP clients can express and what A2A agents can perform, mediated by the gateway's intelligence layer. The theoretical maximum includes:

- **Any A2A-compliant agent** accessible via any supported transport (JSON-RPC, gRPC, REST) can be exposed as MCP tools/resources/prompts
- **Any MCP client** (Claude Desktop, Cursor, VS Code, custom) can interact with A2A agents through the gateway
- **Multi-turn conversations** are bounded only by context window limits (A2A contextId enables indefinite conversation threads)
- **Structured output** is constrained to what both sides can express — A2A's MIME-typed Parts and MCP's JSON Schema `outputSchema`
- **Authentication** can be fully delegated through the elicitation channel
- **Observability** is end-to-end via OpenTelemetry W3C trace context propagation

The **practical** boundary is narrower: MCP elicitation supports only flat object schemas (no nested objects, no arrays of objects), A2A gRPC transport requires TLS, session state TTL defaults to 1 day, and MCP Tasks are still experimental.

### Phased technical roadmap

**Phase 1 — Foundation alignment (0-3 months)**
This phase brings the existing architecture into alignment with current FastMCP 3.0 and A2A v0.3 capabilities without changing the public API.

- **Migrate ToolMapper to Transform chains**: Replace the three ToolMapper implementations with composed Transform stacks (`Namespace` + `ToolTransform` + `Visibility`). This is a refactoring that gains composability and interoperability with all FastMCP transforms.
- **Adopt Streamable HTTP transport**: Switch from SSE to Streamable HTTP as default, gaining session management, resumability, and single-endpoint serving.
- **Implement complete A2A error code mapping**: Map all five A2A-specific error codes (-32001 through -32005) to appropriate MCP error responses with structured error metadata.
- **Add `ctx.sample()` with `result_type`** to LLMRouter: Replace raw sampling with typed routing decisions, gaining validation and retry on malformed LLM responses.
- **Adopt FastMCP session state**: Migrate `ContextManager`'s session↔context mapping to `ctx.get_state`/`ctx.set_state` with pluggable backends.
- **Enable OpenTelemetry**: Zero-config instrumentation is available — just configure a `TracerProvider`. Add custom span attributes for A2A agent ID, task ID, and routing decisions.

**Phase 2 — Async and lifecycle capabilities (3-6 months)**
This phase adds genuinely new capabilities that expand the gateway's operational model.

- **MCP Tasks integration**: Declare `task=TaskConfig(mode="optional")` on A2A-backed tools. Implement the full `tasks/get` → `tasks/result` → `notifications/tasks/status` flow backed by A2A task lifecycle events.
- **Background task execution via Docket**: Long-running A2A interactions execute as Docket tasks with SQLite/Postgres persistence, surviving gateway restarts.
- **Elicitation bridge for input-required**: When A2A agents transition to `input-required`, trigger `elicitation/create` with form mode, relay user responses back as `message/send`.
- **Push notification consolidation**: Unify `WebhookReceiver` with MCP task notifications — A2A push notifications translate to `notifications/tasks/status` MCP messages.
- **Lifespan composition**: Refactor resource management to use pipe-composed lifespans for A2A client pool, health monitor, webhook receiver, and storage connections.
- **Agentic planning loops**: Extend `ctx.sample()` usage to support multi-step orchestration — the LLM can call introspection tools, decompose requests, dispatch to multiple agents, and verify results before returning.

**Phase 3 — Protocol advancement (6-12 months)**
This phase tracks A2A v1.0 and MCP specification evolution, implementing capabilities as they stabilize.

- **A2A v1.0 migration**: Adopt the three-layer spec architecture, protobuf types, new SecurityScheme model, `ListTasks` operation, and `TASK_STATE_AUTH_REQUIRED` handling.
- **gRPC transport for A2A backends**: Implement gRPC client in `A2AAgentAdapter` with transport selection based on Agent Card `preferredTransport` / `additionalInterfaces`.
- **Auth-required → elicitation flow**: Full implementation of the auth escalation bridge using URL-mode elicitation for OAuth2 and form-mode for API keys.
- **MCP Extensions negotiation**: Declare and negotiate gateway extensions during MCP initialization (`capabilities.extensions`), enabling metadata passthrough to A2A agents.
- **A2A Extensions passthrough**: Forward A2A extension metadata between MCP clients (via tool `_meta`) and A2A agents (via message `metadata`).
- **Agent Card signing verification**: Validate `AgentCardSignature` JWS signatures when registering A2A agents, ensuring agent identity integrity.
- **Agent Registry integration**: As A2A's Agent Registry specification matures, implement dynamic agent discovery beyond static configuration.

**Phase 4 — Advanced orchestration (12+ months)**
This phase explores capabilities at the boundary of what the protocol combination enables.

- **Multi-agent workflow composition**: Use `ctx.sample()` with planning tools to decompose complex requests across multiple A2A agents, with dependency tracking and partial result aggregation.
- **Structured output schema propagation**: When A2A agents declare output schemas in their skills, propagate these as MCP `outputSchema` on corresponding tools, enabling end-to-end typed data flow.
- **Cross-gateway federation**: Multiple Agentique instances expose their agent registries to each other via A2A Agent Cards, creating a federated mesh of gateway-mediated agents.
- **Provider composition**: Combine `AgentProvider` with `OpenAPIProvider` (for REST APIs), `ProxyProvider` (for remote MCP servers), and `FileSystemProvider` (for local tools), all behind a unified transform pipeline — making Agentique a universal capability aggregator, not just an A2A gateway.

### Feasibility and architectural coherence assessment

Each proposed direction can be evaluated on two axes — **technical feasibility** (how much underlying infrastructure already exists) and **architectural coherence** (how well it aligns with Agentique's design principles):

- **High feasibility, high coherence**: Transform-based ToolMapper migration, session state unification, OpenTelemetry enablement, complete error code mapping, `result_type` on LLMRouter. These are straightforward refactorings that leverage existing FastMCP 3.0 infrastructure.
- **High feasibility, medium coherence**: MCP Tasks integration, Docket background tasks, elicitation bridge. These are well-supported by the SDKs but expand gateway responsibilities beyond pure routing — justified by the "artifacts as first-class resources" principle.
- **Medium feasibility, high coherence**: Agentic planning loops, gRPC transport, A2A Extensions passthrough. The primitives exist but integration requires careful design to maintain the protocol-agnostic adapter pattern.
- **Medium feasibility, medium coherence**: Auth-required → elicitation flow, Agent Card signing, MCP Extensions negotiation. These depend on experimental or RC-stage specifications that may change.
- **Lower feasibility, exploratory**: Cross-gateway federation, multi-agent workflow composition, WebSocket transport. These push beyond current specification boundaries and require design decisions that could constrain future flexibility.

---

## Conclusion: where the inflection points are

Three architectural inflection points would meaningfully expand Agentique's capability:

**The first inflection** is moving from single-shot dispatch to multi-step orchestration via `ctx.sample()` with tools. This transforms the gateway from a router to an orchestrator — the LLM doesn't just pick an agent, it plans a strategy, executes it across multiple agents, and verifies the result. The infrastructure (`ctx.sample()`, `tools` parameter, `result_type`, `tool_concurrency`) already exists in FastMCP 3.0. The key design decision is whether orchestration tools are built-in or pluggable via the `AgentAdapter` Protocol.

**The second inflection** is MCP Tasks alignment. Once A2A task lifecycle maps natively to MCP Tasks (rather than being exposed through the `task` meta-tool), the gateway becomes truly asynchronous — clients can fire-and-forget, poll, or subscribe to updates. Combined with Docket persistence, this enables production-grade long-running workflows. The constraint is MCP Tasks' experimental status; the specification may change before stabilization.

**The third inflection** is provider composition — combining `AgentProvider` with `OpenAPIProvider`, `ProxyProvider`, and `FileSystemProvider` behind a unified transform pipeline. This shifts Agentique from "A2A-to-MCP gateway" to "universal capability aggregator" that can source tools from A2A agents, REST APIs, remote MCP servers, and local code, all presented through a consistent, policy-enforced, LLM-routed interface. This is the logical conclusion of the protocol-agnostic adapter principle: if the adapter interface is truly protocol-agnostic, then the gateway should serve all protocol types, not just A2A. The trade-off is maintaining gateway purity versus becoming a general-purpose platform — but FastMCP 3.0's Provider architecture makes this composition nearly zero-cost.