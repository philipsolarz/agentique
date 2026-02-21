# Agentique: Architecture Analysis & Evolution Roadmap (v2)

## What Agentique Is — and Isn't

---

### The Origin

Agentique exists because of a gap. You have A2A agents — built with Google ADK, with their own tools, sub-agents, and capabilities — and you have MCP clients like GitHub Copilot, Claude Code, Cursor, and others that speak MCP. There is no way to connect them. No bridge exists between A2A's agent-to-agent protocol and MCP's tool-and-resource protocol. Agentique is that bridge.

The project's reason for being is a single sentence:

> **Agentique is the protocol bridge that makes A2A agents first-class participants in MCP conversations.**

When you connect GitHub Copilot to an Agentique MCP server, every A2A agent registered with that server becomes available in your Copilot conversation — as if those agents were native MCP tools. The agents may have their own MCP tools, sub-agents, complex internal workflows — none of that matters to the MCP client. Agentique handles the translation.

### What This Means for Identity

Agentique is not a gateway, not an orchestrator, not a workflow engine, and not a policy platform. It is a **protocol bridge with intelligent routing**. The word "bridge" is precise: it connects two protocol worlds with the highest fidelity possible, so that high-value features of both MCP/FastMCP and A2A survive the crossing.

The routing capability — using MCP sampling to let the LLM choose which agent handles a message — is a natural extension of the bridge position. When you expose multiple agents through one MCP server, routing is needed. But routing serves the bridge; the bridge doesn't serve routing.

### The Quality Bar

The experience of using an A2A agent through Agentique should be as close as possible to using that agent directly. If an agent supports long-running tasks, the MCP client should be able to track them. If an agent requires authentication, the MCP client should be prompted for credentials. If an agent produces artifacts, the MCP client should be able to read them. Fidelity of translation is the primary quality metric.

### The Boundary

Agentique should NOT:

- Decompose user messages into sub-tasks dispatched to multiple agents (that's an orchestrator)
- Maintain its own planning or reasoning beyond agent selection (that's a planner)
- Enforce organizational policies, budgets, or compliance rules on tool usage (that's a governance platform)
- Manage complex multi-step workflows across agents (that's a workflow engine)
- Build agent UIs or dashboards (that's an application layer)

Agentique SHOULD:

- Translate between MCP and A2A with the highest possible fidelity
- Route messages to the right agent when multiple agents are available
- Preserve the full lifecycle of A2A interactions (streaming, tasks, artifacts, auth) through MCP
- Make the MCP client experience excellent — responsive, informative, correct
- Leverage FastMCP 3.0's composition primitives to present agent capabilities cleanly

---

## Current Architecture Assessment

### Three-Layer Design (Correct, Preserve)

**Core Layer** (`core/`): Protocol-agnostic types, configuration, error hierarchy, event system, telemetry, adapter registry, tool mapper protocols. All interfaces use `typing.Protocol` — no inheritance required. This layer has zero knowledge of MCP or A2A specifics.

**Bridge Layer** (`bridge/`): The mediation substrate. Routing, session/context management, task lifecycle, middleware chain, health monitoring, dependency injection, and lifespan composition. This is where the translation logic lives.

**Adapter Layer** (`adapters/`): Concrete protocol implementations — A2A (primary), generic HTTP, and MCP proxy. The A2A adapter handles push notifications, task resubscription, extended agent cards, extension propagation, and stream reconnection.

**Server Factory** (`server.py`): Single composition root via `create_server()`. Wires everything together and registers MCP tools, resources, and prompts.

### Architectural Invariants (Must Preserve)

1. **Protocol agnosticism at core.** No A2A or MCP types in `core/`. This is the foundation of extensibility.
2. **Structural subtyping everywhere.** All protocols use `typing.Protocol`. Third parties extend without importing Agentique.
3. **Single composition root.** `create_server()` is the only assembly point. Clean testing, clear wiring.
4. **Transform vs. Mapper separation.** ToolMappers control *creation* (what tools exist). FastMCP Transforms control *presentation* (how tools appear). Two-phase pipeline.
5. **Stateless bridge, stateful session.** Bridge holds no per-request state. State lives in TaskManager, ContextManager, session-scoped overrides.
6. **Event-driven side channels.** AsyncEventEmitter decouples lifecycle hooks from the request flow.

---

## Feature Audit: Keep, Evolve, or Remove

Every existing feature is evaluated against the bridge identity: does it serve high-fidelity protocol translation, intelligent routing, or excellent MCP client experience?

### KEEP — Core to Bridge Identity

| Feature | Why It Stays |
|---------|-------------|
| **LLM-driven routing** (sampling-based `RoutingDecision`) | Core bridge value-add. Multiple agents need routing. Using MCP sampling for this is elegant and protocol-native. |
| **Session/context management** (`ContextManager`) | Essential. Maps MCP sessions to A2A contexts. Without this, conversation continuity breaks. |
| **Task lifecycle tracking** (`TaskManager`, `TaskStore`) | Essential. A2A tasks have rich state machines (submitted → working → input-required → completed/failed/canceled). The bridge must track and expose this. |
| **Background tasks** (`agent_background` tool, `TaskConfig`) | A2A supports long-running tasks. Hiding this from MCP clients would be a fidelity loss. |
| **Auth elicitation** (security scheme parsing, `ctx.elicit()`) | A2A agents declare security requirements. The bridge must prompt MCP clients for credentials. Removing this breaks agents that require auth. |
| **Artifact capture/serving** (A2A artifacts → MCP resources) | A2A agents produce artifacts. Exposing them as MCP resources is a direct protocol translation. |
| **Health monitoring** (`HealthMonitor`) | The bridge needs to know if agents are reachable. Health checks are infrastructure, not feature creep. |
| **ToolMapper protocol** (Default, PerSkill, FlatHierarchy) | Controls how agent skills become MCP tools. Different mapping strategies serve different agent topologies. Core bridge concern. |
| **Transform pipeline** (Namespace, Visibility, ResourcesAsTools) | FastMCP's composition model. The right way to present agent capabilities. No custom code needed. |
| **Middleware chain** (`BridgeMiddleware`, `FastMCPMiddleware`) | Request/response processing pipeline. Enables observability, timing, validation without touching core logic. |
| **Event system** (`AsyncEventEmitter`) | Needed for streaming, observability, lifecycle hooks. Low-overhead, high-utility. |
| **Lifespan composition** (`compose_lifespans()`) | Background services (health checks, cleanup) compose cleanly. Infrastructure, not feature. |
| **Adapter registry** (entry points, `@register_adapter`) | Third-party protocol adapters. The bridge should support more than just A2A eventually. |
| **Testing infrastructure** (`MockAdapter`, `InMemoryBridge`, `assert_adapter_protocol()`) | Non-negotiable for a protocol bridge. Translation correctness requires rigorous testing. |
| **Direct routing** (`DirectRouter`) | When there's only one agent, skip LLM routing entirely. Reduces latency, respects simplicity. |

### EVOLVE — Useful but Needs Refocusing

| Feature | Current State | Recommended Evolution |
|---------|--------------|----------------------|
| **Visibility control** (server-level + session-level) | Implements tenant-based visibility policies with middleware enforcement. | Simplify. Session-level enable/disable of agents is useful (e.g., "I only want to talk to agent X right now"). Tenant-based policy enforcement is governance territory — strip the policy framework, keep the per-session toggle. |
| **Extension metadata** (`extensions.py`, 4 URI-keyed types) | Routing metadata, policy context, MCP session, trace context propagation. | Keep routing-metadata and trace-context (directly serve bridge function). Remove policy-context (governance concern). Simplify mcp-session to what's needed for context mapping. |
| **DI registry** (module-level + session-scoped overrides) | Full dependency injection with `Depends()` integration. | Keep but recognize this is implementation infrastructure, not a user-facing feature. Don't extend it further. |
| **Webhook reception** | Push notification receiver, subscription management. | Keep only what's needed for A2A push notifications. Remove generic webhook handling if it exists beyond A2A push support. |
| **Routing confidence scoring + fallback cascades** | `RoutingDecision` includes confidence, reasoning, fallback agents. | Keep. This directly serves routing quality. But don't evolve into "routing memory" or "adaptive routing" — those cross into learning/optimization systems. |
| **Output models** (`bridge/models.py`) | Structured output for routing decisions. | Keep. Part of the routing implementation. |

### REMOVE — Outside Bridge Identity

| Feature | Why It Should Go |
|---------|-----------------|
| **`RoutingDecision.requires_decomposition`** | This flag is a structural invitation to orchestration. If a message requires decomposition, that's the MCP client's (or user's) job — not the bridge's. Decomposition means planning, sub-task dispatch, result aggregation. Remove the field entirely. |
| **Policy transform layer** (as currently conceived) | External policy documents (JSON/YAML) filtering tools based on compliance rules, cost budgets, operational constraints → this is a governance platform. If someone needs this, they should build it as a FastMCP Transform *outside* Agentique and compose it in via the transform pipeline. The seam exists; the implementation shouldn't be in Agentique. |
| **`AgentEvent.requires_confirmation`** handling (as a gateway-enforced gate) | If an agent needs confirmation, that's between the agent and the user. The bridge should relay the agent's request, not insert its own confirmation step. If A2A defines this clearly in the protocol, bridge it. Don't invent a confirmation workflow. |
| **Routing memory / confidence feedback loops** (as previously proposed) | Storing `(message_signature, agent_id, confidence, outcome_quality)` tuples and feeding calibration data back into routing → this is a learning system. It introduces statefulness in the routing path and optimizes for something that should be the LLM's job. The LLM already gets agent descriptions and health status. That's enough. |
| **Per-agent routing statistics resource** | Exposes routing analytics as an MCP resource. This is a dashboard/monitoring concern, not a bridge concern. If someone wants this, they can build it from the event stream. |
| **Conversation-aware routing** (session history informing agent selection) | Crossing into planning territory. If the user wants to talk to a specific agent, they can say so. The LLM router already has the conversation context from the MCP client. |

### BORDERLINE — Decide Based on Practical Value

| Feature | Case For | Case Against | Recommendation |
|---------|---------|-------------|----------------|
| **Structured audit log resource** (`a2a://audit`) | Useful for debugging translation issues. Users can see what the bridge did. | Monitoring concern. Event stream already exists. | **Keep as opt-in.** Debugging the bridge itself is a legitimate user need when agent interactions go wrong. But keep it simple — recent events only, no persistence. |
| **Unified session resource** (`a2a://session/{id}`) | Shows the user what the bridge knows about their session. | Could expose internal state that confuses users. | **Keep, but simplify.** Expose active agents, active tasks, current context. Don't expose routing internals. |
| **Capabilities resource** (`a2a://capabilities`) | Tells MCP clients what the bridge supports. | Static JSON, limited utility. | **Keep.** Low cost, useful for tooling and discovery. |

---

## Cross-Protocol Mapping Depth

This is Agentique's core job. Current mapping fidelity:

| MCP Primitive | A2A Mapping | Fidelity | Notes |
|---------------|-------------|----------|-------|
| Tool call → `send_message` | Full | Clean translation. Message content, streaming, structured output all work. |
| Elicitation → input-required/auth-required | Full | Auth scheme parsing and credential prompting work well. |
| Sampling → LLM routing | Full | Elegant use of the MCP sampling capability for agent selection. |
| Output schema → structured ToolResult | Full | Agent response structure preserved. |
| Background task → A2A task lifecycle | Full | Task states mapped correctly. Long-running tasks work. |
| Resource read → agent cards, artifacts | Partial | Agent cards and artifacts served as resources. Missing: reverse direction (MCP resources → A2A attachments). |
| Progress → `ctx.report_progress` | Partial | Event stream progress relayed, but granularity could improve. |
| Session state → context mapping | Partial | Bidirectional session ↔ context mapping works. Persistence is manual, not systematic. |
| Prompt → card-derived templates | Shallow | Generic templates from agent cards. Agent skills have richer metadata (inputModes, outputModes) that could generate better prompts. |
| MCP notifications → task state changes | Missing | No mechanism to push task state updates to MCP clients. Clients must poll. |
| MCP resources → A2A attachments | Missing | One-directional only. Clients can't send files/data to agents via the bridge. |
| Task cancellation | Missing | A2A defines `tasks/cancel`. No cancellation tool exposed. |

**Priority for bridge fidelity:** Complete the missing and partial mappings before adding any new features. This IS the product.

---

## FastMCP 3.0 GA — New Opportunities

FastMCP 3.0 shipped February 18, 2026. Several capabilities are directly relevant to the bridge mission:

### High Relevance — Should Adopt

**ResponseLimitingMiddleware.** Caps tool response sizes with UTF-8-safe truncation. When an A2A agent returns a massive response, the MCP client's context window shouldn't be overwhelmed. This is a translation quality concern — the bridge should ensure responses are consumable. Low effort, high impact.

**Component Versioning** (`@tool(version="2.0")`). When an A2A agent's capabilities change (new skills, modified input schemas), the bridge could expose versioned tool surfaces. The MCP client sees the latest version by default; legacy integrations can pin to older versions. This is particularly useful for agents that evolve frequently. Medium effort, medium impact — worth adopting once agents actually start versioning their capabilities.

**Granular Authorization** (per-component `authorize` callbacks). Currently Agentique handles auth at the agent level. FastMCP 3.0 allows per-tool auth. This maps naturally to A2A agents that have different security requirements per skill. For example, an agent's "read" skills might be public while its "write" skills require OAuth. Medium effort, high fidelity improvement.

**Progressive Disclosure** (Visibility + Auth + Session State composition). FastMCP 3.0's blog specifically describes this pattern: mount tools hidden by default, provide an authenticated unlock tool, and the session evolves as trust increases. For Agentique, this could mean: initially show agent descriptions and read-only tools. When the user authenticates with an agent, dynamically reveal that agent's full tool set for the session. This is a *much* better UX than showing all tools upfront and failing on auth at call time. Medium effort, significant UX improvement.

**CLI Tools** (`fastmcp list`, `fastmcp call`, `fastmcp discover`, `fastmcp generate-cli`). Not a runtime feature but invaluable for development and debugging. `fastmcp call` lets you test individual agent tools from the terminal. `fastmcp generate-cli` could generate a standalone CLI for the entire Agentique server — every agent tool becomes a typed CLI subcommand. Should be documented and integrated into the development workflow. Low effort.

### Medium Relevance — Worth Investigating

**MCP Apps / `ui://` resource scheme.** If A2A agents can provide interactive UIs (forms, dashboards, visualizations), the bridge could serve them through MCP Apps. This is future-facing — most MCP clients don't render apps yet. But it's protocol-level support that costs little to enable. The question is whether any A2A agents actually produce UI content. Low effort to enable, value depends on agent ecosystem.

**OpenTelemetry native instrumentation.** FastMCP 3.0 has built-in OTel. Agentique already has OTel span attributes, but the traces could be richer if they leveraged FastMCP's native instrumentation rather than custom middleware. Worth unifying. Low-medium effort.

**Concurrent sampling** (`context.sample()` with `tool_concurrency=0`). Could enable parallel agent capability discovery during routing. Instead of sequentially checking agent availability, the router could sample multiple agents concurrently. Low effort, potential latency improvement.

### Low Relevance — Not Needed Now

**FileSystemProvider hot-reload.** Useful for development but not for the bridge runtime. Agents are registered programmatically, not from files.

**Playbooks** (chained progressive disclosure creating workflows). This is orchestration territory. Interesting as a FastMCP concept, but outside bridge scope.

**Docket background task coordination (Redis-based).** Agentique already has task lifecycle management. The Redis coordination layer is for distributed FastMCP deployments, not for the bridge's task tracking.

---

## A2A SDK 0.3 — Missing Protocol Coverage

### Should Implement

**Task cancellation.** A2A defines `tasks/cancel` and `TaskNotCancelableError`. Agentique defines the error type but exposes no cancellation tool. This is a protocol gap — if the user wants to cancel a long-running task, the bridge should allow it. Low effort.

**Trace context propagation.** The `TRACE_CONTEXT_URI` extension and `pack_trace_context()` / `current_trace_context()` are implemented but not wired into the adapter's `_build_message()`. Completing this enables end-to-end distributed traces. Low effort.

**Protocol versioning.** A2A v0.3 requires `A2A-Version` headers. Agentique should send and negotiate protocol versions. If an agent only supports v0.2, the bridge should know and adjust. Low effort.

### Should Investigate

**gRPC transport.** A2A v0.3 added gRPC as an alternative transport. Lower latency, bidirectional streaming. The `A2AClientPool` has infrastructure for `supported_transports` but no gRPC logic. Worth implementing when performance-sensitive agents adopt gRPC. Medium-high effort.

**Agent card signature verification.** A2A v0.3 supports signed cards. For deployments where agents come from untrusted sources, the bridge should verify card signatures before registering the agent. Medium effort, enterprise concern.

### Not Needed

**Multi-stream task subscription.** Multiple concurrent streams for the same task is an edge case. Single-stream reconnection is sufficient for the bridge use case.

**Full context storage semantics.** A2A's full conversation trace as a queryable resource is more than the bridge needs. Session-to-context mapping is sufficient.

---

## Revised Roadmap

### Phase 1: Translation Fidelity (Immediate — Complete the Bridge)

*Goal: Close every gap in the MCP ↔ A2A mapping. This is the core product.*

| Item | What | Effort | Impact |
|------|------|--------|--------|
| Task cancellation tool | Expose `cancel_task` calling A2A `tasks/cancel` | Low | Completes task lifecycle |
| Trace context propagation | Wire `current_trace_context()` into `_build_message()` | Low | End-to-end distributed tracing |
| ResponseLimitingMiddleware | Enable FastMCP 3.0's response size caps | Low | Prevents context window overflow |
| Protocol version negotiation | Send `A2A-Version` headers, handle version mismatches | Low | Protocol correctness |
| Artifact TTL/eviction | Configurable TTL with background cleanup | Low | Prevents memory growth |
| Richer prompt generation | Use agent skill `inputModes`/`outputModes` for typed MCP prompts | Medium | Better agent discovery UX |
| MCP notifications for task states | Relay A2A task state changes as MCP notifications | Medium | Eliminates client polling |
| Remove `requires_decomposition` | Delete the field from `RoutingDecision` | Trivial | Removes orchestration temptation |

**Why this is Phase 1:** Every item here directly improves the fidelity of A2A ↔ MCP translation. No architectural changes needed. All are derivable from existing code.

### Phase 2: Client Experience (Near-term — Make It Feel Great)

*Goal: The MCP client experience should be excellent, not just correct.*

| Item | What | Effort | Impact |
|------|------|--------|--------|
| Progressive disclosure | Hide agent tools by default, reveal on auth. Use FastMCP 3.0's Visibility + Auth + Session State pattern. | Medium | Major UX improvement — no more auth failures on first tool call |
| Granular authorization | Per-tool auth via FastMCP 3.0 `authorize` callbacks, mapped to A2A skill-level security | Medium | Fidelity: different agent skills can have different auth requirements |
| Component versioning | `@tool(version=...)` when agent capabilities change | Medium | Graceful evolution of agent tool surfaces |
| Bidirectional artifact mapping | MCP resources → A2A message attachments (send files to agents) | Medium-high | Completes the artifact bridge |
| Concurrent sampling for routing | Use FastMCP 3.0's `tool_concurrency` for parallel agent probing | Low | Routing latency reduction |
| CLI integration | Document and integrate `fastmcp list/call/discover` for development | Low | Developer experience |

**Why this is Phase 2:** These improve the quality of the bridge experience. Progressive disclosure in particular transforms the UX — instead of showing 50 tools from 10 agents upfront, the user sees a clean surface that expands as they authenticate.

### Phase 3: Production Robustness (Medium-term — Scale and Harden)

*Goal: The bridge should be reliable and observable in production deployments.*

| Item | What | Effort | Impact |
|------|------|--------|--------|
| gRPC transport | A2A SDK gRPC support in `A2AClientPool` | Medium-high | Performance for latency-sensitive agents |
| Agent card signature verification | Verify A2A v0.3 signed cards before registration | Medium | Trust boundary for untrusted agent sources |
| Unified OTel instrumentation | Merge custom spans with FastMCP 3.0 native OTel | Low-medium | Cleaner observability |
| Task resumption across sessions | Persist `input-required` tasks, resume in new sessions | Medium | Long-running task resilience |
| Session state persistence | Systematic cross-restart session continuity | Medium | Production reliability |
| MCP Apps passthrough | If A2A agents provide UI content, serve via `ui://` | Low-medium | Future-proofing for rich agent UIs |

**Why this is Phase 3:** These are production concerns. The bridge works correctly (Phase 1) and feels great (Phase 2); now it needs to be robust under real-world conditions.

---

## Simplification: What to Remove from the Codebase

Based on the feature audit, the following should be actively removed or significantly simplified:

### Remove

1. **`RoutingDecision.requires_decomposition` field** — Delete from `bridge/router.py`. This flag has no implementation and its presence invites orchestration scope creep. If a future contributor sees it, they'll build decomposition. Remove the temptation.

2. **`POLICY_CONTEXT_URI` extension** — Delete from `extensions.py`. Policy context propagation serves governance use cases outside the bridge identity. If someone needs policy enforcement, they compose a Transform externally.

3. **Per-agent routing statistics resource** (if implemented or planned) — This is analytics, not bridging. The event stream provides the raw data if someone wants to build monitoring.

### Simplify

4. **Visibility system** — Keep session-level agent enable/disable (useful: "only show me agent X"). Remove tenant-based policy infrastructure and `VisibilityPolicy` middleware if it goes beyond per-session toggles. The progressive disclosure pattern from FastMCP 3.0 is a better approach.

5. **Extension metadata** — Keep `ROUTING_METADATA_URI` and `TRACE_CONTEXT_URI`. Evaluate whether `MCP_SESSION_URI` can be simplified to just what ContextManager needs. Remove `POLICY_CONTEXT_URI`.

6. **Webhook reception** — Verify this only handles A2A push notifications. If there's generic webhook infrastructure beyond what A2A push requires, simplify to just the A2A push handler.

---

## What Agentique Is Called

"Gateway" implies a network boundary device. "Orchestrator" implies workflow management. "Router" implies just routing and nothing else.

The best terms for what Agentique actually is:

- **Protocol bridge** — accurate, technical, describes the core function
- **A2A-to-MCP bridge** — maximally specific
- **Agent bridge** — more marketable, implies connecting agent worlds

Suggested positioning: *"Agentique is an agent bridge that connects A2A agents to MCP clients. It translates between the protocols with high fidelity so your A2A agents — with all their tools, sub-agents, and capabilities — become first-class participants in GitHub Copilot, Claude Code, Cursor, and any other MCP-compatible environment."*

---

## Architectural Principles (Revised)

1. **Translation fidelity is the product.** Every A2A capability that can be meaningfully expressed in MCP should be. Every MCP capability that can enhance agent interaction should be bridged back. Close the mapping gaps before adding features.

2. **The bridge serves the MCP client.** Design decisions should optimize for the experience of the person in GitHub Copilot or Claude Code. If they can't tell the difference between using an agent directly and using it through Agentique, you've succeeded.

3. **Protocol capabilities before bridge inventions.** Complete A2A and FastMCP feature coverage first. Don't invent novel abstractions when protocol features exist.

4. **Compose, don't embed.** Use FastMCP 3.0's Provider/Transform composition model. Features that can be external Transforms should not be built into Agentique's core.

5. **Routing is a bridge feature, not the bridge.** Smart routing serves the bridge. The moment routing becomes the primary concern, the project has drifted.

6. **The adapter boundary is sacred.** No MCP types in adapters. No A2A types in the bridge. Core types are the universal currency.

7. **If in doubt, don't build it.** Every feature not built is a feature that doesn't need maintenance, documentation, or debugging. The bridge should be small, correct, and reliable.

---

*Analysis based on Agentique v0.4.0 codebase, FastMCP 3.0.0 GA (February 18, 2026), and A2A Python SDK 0.3.23 (February 17, 2026). Revised with corrected project identity and scope.*
