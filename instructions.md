# Agentique Console: Architecture and Development Plan

**Agentique Console** is a desktop agentic coding and research product built around a Recursive LLM (RLM) execution engine called **Ripple**. Unlike existing tools that stuff entire contexts into a single LLM call or rely on lossy summarization, Agentique Console treats complex assets as dependency graphs, operating over them programmatically through a REPL with recursive sub-agent handoffs. The RLM approach — validated by Zhang, Krassa, and Khattab (MIT CSAIL, arXiv:2512.24601v2, January 2026) — enables processing inputs **up to two orders of magnitude beyond context windows**, with GPT-5 + RLM outperforming base GPT-5 by **28.4%** on the OOLONG benchmark and achieving **58.0% F1** on OOLONG-Pairs where base GPT-5 scores below 0.1%. This document provides a complete, implementable architecture: product definition, system design with component diagrams, technology decisions with explicit rationale, recursive execution guardrails, development milestones, and a starter blueprint with repo structure and pseudocode.

---

## A. Product definition

### Primary user personas

**Platform engineers** use Agentique Console to understand sprawling infrastructure-as-code repositories, trace dependency chains across Terraform modules and Kubernetes manifests, and generate policy-compliant configurations. Their workflows involve multi-hop traversal of dependency graphs — exactly the scenario where stuffing tokens fails and RLM's programmatic decomposition excels.

**Software developers** use it for large-codebase reasoning: understanding unfamiliar repositories, implementing features that touch many files, debugging cross-cutting issues, and reviewing pull requests that span architectural boundaries. The Ripple engine's ability to chunk a codebase, recursively process segments, and synthesize findings maps directly to how experienced developers manually navigate complex codebases.

**Analysts and researchers** use it for multi-document synthesis: comparing regulatory filings, analyzing policy documents, extracting structured data from unstructured collections. The RLM paper demonstrates this use case explicitly — OOLONG-Pairs requires cross-referencing pairs of Wikipedia articles, and RLM's chunking + recursive sub-calling pattern is the emergent solution.

### Key workflows

- **Coding assistance**: File editing, code generation, test writing, refactoring across multiple files. The agent reads relevant code via REPL variables rather than loading entire files into context, applies edits surgically, and verifies changes by running tests.
- **Repository understanding**: Answering architectural questions about large codebases by building and traversing dependency graphs. Ripple decomposes the question, fans out sub-agents to investigate relevant modules, and synthesizes findings.
- **Policy and document analysis**: Processing documents that exceed context windows by chunking, extracting structured facts per chunk via sub-agents, then merging and deduplicating results — the exact pattern the RLM paper observes emerging naturally.
- **Multi-document synthesis**: Cross-referencing multiple sources to answer questions that require reasoning across documents — the OOLONG-Pairs scenario where RLM transforms near-zero performance into 58% F1.

### Non-goals and when RLM is unnecessary

The RLM paper's Observation 3 notes that **on short, simple tasks, RLM slightly underperforms** the base model due to overhead. Agentique Console explicitly does not use Ripple for: single-file edits under ~500 lines, simple Q&A answerable from context, code completion/autocomplete (latency-critical), and straightforward refactors where the change pattern is mechanical. For these, the system routes to a direct (non-recursive) LLM call. The routing decision is a first-class architectural concern described in Section D.

---

## B. System architecture

### High-level component topology

```
┌──────────────────────────────────────────────────────────────────────┐
│                        AGENTIQUE CONSOLE                             │
│                                                                      │
│  ┌──────────────────────────────────────────┐                        │
│  │          DESKTOP SHELL (Tauri v2)         │                        │
│  │  ┌────────────────────────────────────┐  │                        │
│  │  │  WebView Frontend (React + ag-ui)  │  │                        │
│  │  │  • Chat panel + streaming render   │  │                        │
│  │  │  • Artifact viewer + diff display  │  │                        │
│  │  │  • Recursion tree visualizer       │  │                        │
│  │  │  • Cost/budget dashboard           │  │                        │
│  │  │  • Tool approval dialogs           │  │                        │
│  │  └──────────────┬─────────────────────┘  │                        │
│  │                 │ Tauri IPC (Commands/Events/Channels)             │
│  │  ┌──────────────┴─────────────────────┐  │                        │
│  │  │     RUST CORE PROCESS               │  │                        │
│  │  │  ┌──────────────────────────────┐  │  │                        │
│  │  │  │   Agent Runtime Core         │  │  │                        │
│  │  │  │  • Planning/execution loop   │  │  │                        │
│  │  │  │  • Session state machine     │  │  │                        │
│  │  │  │  • Tool invocation layer     │  │  │                        │
│  │  │  │  • Artifact store            │  │  │                        │
│  │  │  └──────┬──────────┬────────────┘  │  │                        │
│  │  │         │          │               │  │                        │
│  │  │  ┌──────▼───┐ ┌───▼────────────┐  │  │                        │
│  │  │  │ Ripple   │ │ MCP Client     │  │  │                        │
│  │  │  │ Engine   │ │ Manager        │  │  │                        │
│  │  │  │ (REPL +  │ │ • Multi-server │  │  │                        │
│  │  │  │ Recursion│ │ • Namespacing  │  │  │                        │
│  │  │  │ Runtime) │ │ • Permissions  │  │  │                        │
│  │  │  └──────┬───┘ └───┬────────────┘  │  │                        │
│  │  │         │          │               │  │                        │
│  │  │  ┌──────▼──────────▼────────────┐  │  │                        │
│  │  │  │  Provider-Agnostic LLM Layer │  │  │                        │
│  │  │  │  • Model router (cost/cap)   │  │  │                        │
│  │  │  │  • Streaming + structured    │  │  │                        │
│  │  │  │  • Retry + fallback          │  │  │                        │
│  │  │  └──────────────────────────────┘  │  │                        │
│  │  │                                     │  │                        │
│  │  │  ┌──────────────────────────────┐  │  │                        │
│  │  │  │  Data Layer                  │  │  │                        │
│  │  │  │  • Tantivy index (code/docs) │  │  │                        │
│  │  │  │  • Petgraph dep graphs       │  │  │                        │
│  │  │  │  • SQLCipher local DB        │  │  │                        │
│  │  │  │  • Keyring secrets           │  │  │                        │
│  │  │  └──────────────────────────────┘  │  │                        │
│  │  │                                     │  │                        │
│  │  │  ┌──────────────────────────────┐  │  │                        │
│  │  │  │  Observability & Governance  │  │  │                        │
│  │  │  │  • tracing spans per step    │  │  │                        │
│  │  │  │  • Cost accounting           │  │  │                        │
│  │  │  │  • Provenance citations      │  │  │                        │
│  │  │  │  • Safety guardrails         │  │  │                        │
│  │  │  └──────────────────────────────┘  │  │                        │
│  │  └─────────────────────────────────────┘  │                        │
│  └──────────────────────────────────────────┘                        │
│                                                                      │
│  ┌──────────────────────────┐   ┌──────────────────────────┐         │
│  │  MCP Servers (external)  │   │  LLM API Providers       │         │
│  │  • filesystem            │   │  • OpenAI / Anthropic    │         │
│  │  • github                │   │  • Google / local        │         │
│  │  • database              │   │  • Custom endpoints      │         │
│  │  • custom tools          │   │                          │         │
│  └──────────────────────────┘   └──────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
```

### B1. Desktop shell: Tauri v2 + ag-ui

**Why Tauri v2**: Tauri's Rust core + OS-native WebView architecture produces **~3-10 MB** binaries (vs Electron's 150 MB+), provides memory-safe IPC via `#[tauri::command]` with JSON-RPC-like serialization, and ships a fine-grained **ACL capabilities system** where each window gets explicit permissions via JSON capability files. The security model is critical for an agent that executes code and accesses files — every tool invocation path must pass through Tauri's permission layer.

**UI architecture**: The WebView frontend uses React with the **ag-ui protocol** for agent-UI communication. ag-ui provides ~16 standardized event types organized in 5 categories: lifecycle events (`RUN_STARTED`, `STEP_STARTED`), text message streaming (`TEXT_MESSAGE_CONTENT` with delta tokens), tool call visualization (`TOOL_CALL_START/ARGS/END/RESULT`), state synchronization (`STATE_SNAPSHOT` + `STATE_DELTA` via JSON Patch RFC 6902), and reasoning events. The Rust core acts as the ag-ui event producer, streaming events to the WebView via Tauri's **Channel API** (`tauri::ipc::Channel`) for backpressure-aware streaming.

**State model**: Application state lives in the Rust core process as `Mutex<AppState>` managed via Tauri's dependency injection. The WebView maintains a read-only projection synchronized via ag-ui `STATE_SNAPSHOT` and `STATE_DELTA` events. Agent session state (conversation history, REPL variables, recursion tree) persists in the Rust core; the frontend receives a rendered view. This ensures a single source of truth with crash isolation — a WebView crash loses no state.

**Plugin surfaces**: Tauri v2's plugin architecture supports user-installable extensions. Custom MCP server configurations, model provider credentials, and UI themes are all plugin-scoped. Each plugin declares its own permissions, preventing a theme plugin from accessing the filesystem.

**Security sandboxing**: Tauri's **Isolation Pattern** injects an AES-GCM-encrypted iframe between frontend JavaScript and the Rust core, preventing supply-chain attacks in NPM dependencies from directly accessing IPC. File system access is scoped via `tauri-plugin-fs` permissions to project directories only. The agent's tool execution never runs in the WebView process — all execution happens in the Rust core or in sandboxed child processes.

### B2. Agent runtime core (Rust)

**Planning/execution loop**: The core agent loop follows the proven `prompt → reason → act → observe → repeat` pattern, but with a critical difference: the Ripple engine replaces the "stuff everything into prompt" step with programmatic REPL-based asset access. The loop is implemented as an async state machine:

```
enum AgentState {
    Idle,
    Planning { task: Task, context: SessionContext },
    LlmCall { messages: Vec<Message>, tools: Vec<ToolDef> },
    ToolExecution { calls: Vec<ToolCall>, pending: Vec<oneshot::Sender<Value>> },
    RippleExecution { repl: ReplSession, depth: u8 },
    Streaming { chunks: mpsc::Receiver<AgUiEvent> },
    AwaitingUser { prompt: String, reply: oneshot::Sender<UserResponse> },
    Complete { result: Artifact },
    Error { error: AgentError, recoverable: bool },
}
```

**Communication uses the Op/Event queue pattern** (inspired by Codex CLI's architecture): clients submit `Op` variants (UserTurn, Interrupt, ExecApproval) and receive `Event` variants (AgentMessage, StepProgress, TaskComplete) asynchronously via tokio channels. This enables **mid-task course correction** — users can inject instructions while the agent is actively working, matching Claude Code's h2A async dual-buffer pattern.

**Tool invocation layer**: Tools are registered via a `ToolRouter` that maps tool names to implementations of an async `Tool` trait. The router aggregates tools from three sources: built-in tools (file read/write, shell execution, search), MCP tools (discovered via `tools/list` from connected MCP servers), and Ripple-internal tools (REPL variable access, sub-agent spawn). Each tool call is wrapped in a tracing span for observability and a budget check for cost governance.

**Conversation/session state**: Sessions persist as append-only **JSONL rollout files** in `$AGENTIQUE_HOME/sessions/`, enabling session resumption and forking from earlier transcript points. The conversation state includes: message history, current REPL variable map, recursion tree state, cumulative cost, and TODO/plan state. A **compressor** triggers at ~85% context utilization, summarizing older messages while preserving REPL variable references and plan state — adapting Claude Code's proven compaction approach but preserving Ripple's symbolic handles.

**Artifact store and provenance**: Every agent output (generated code, analysis reports, extracted data) is stored as an `Artifact` with: content hash (SHA-256), creation timestamp, producing step ID, input references (which REPL variables / tool results fed into it), and the model + prompt that generated it. Artifacts are immutable once created. This enables full **provenance tracking** — any output can be traced back through the chain of reasoning steps, tool calls, and sub-agent invocations that produced it.

### B3. Recursive LLM ("Ripple") execution environment

This is the architectural centerpiece. Ripple implements the RLM algorithm from the paper as a first-class runtime within the agent.

**How the agent operates on large assets without stuffing context**: The RLM paper's key insight is three design choices missing from existing scaffolds: **(1)** a symbolic handle to the prompt P so the model manipulates it without copying into context, **(2)** output via REPL variables rather than autoregressive Finish, and **(3)** symbolic recursion enabling programmatic sub-calls in loops. Ripple implements all three:

1. **Symbolic handles**: When a user asks about a large codebase or document collection, Ripple loads it as a named variable in the REPL environment (e.g., `$REPO` or `$DOCUMENT_SET`). The LLM receives metadata about the variable (type, size, structure summary, available operations) but never the raw content. The model writes code to inspect, slice, filter, and transform these variables.

2. **Variable-based output**: The model sets REPL variables (e.g., `$RESULT`, `$FINDINGS`) rather than generating a monolithic text response. This enables incremental construction of complex outputs and prevents the "running out of output tokens" failure mode the paper identifies for thinking models.

3. **Symbolic recursion**: The model can invoke `sub_RLM(snippet, instruction)` as a REPL function, spawning a sub-agent that receives a portion of the data and a focused task. Sub-agent results are returned as REPL variables, available for the parent to aggregate.

**REPL primitives — Read/Evaluate/Print/Loop**:

```
READ:  Load asset into REPL variable; expose metadata (size, type, structure)
       Operations: $var.slice(start, end), $var.search(pattern),
                   $var.symbols(), $var.dependencies(), $var.chunks(n)
                   
EVAL:  Execute LLM-generated code in sandboxed environment
       Code can: inspect variables, transform data, call tools,
                 set output variables, invoke sub_RLM()
       Sandbox: wasmtime with epoch-based interruption + fuel metering
       
PRINT: Append execution results + metadata to session transcript
       Metadata includes: tokens consumed, wall time, cost, variable changes
       
LOOP:  Continue until model sets $FINAL variable (termination signal)
       Or: budget exhausted, max iterations reached, user interrupt
```

**Recursion/handoff model**: When the Ripple engine encounters a `sub_RLM(data_ref, instruction)` call in model-generated code:

1. A new REPL session is initialized with `data_ref` loaded as `$P` (the prompt variable)
2. The sub-agent receives the instruction plus a budget allocation (fraction of parent's remaining budget)
3. The sub-agent model is selected by the routing policy — by default, a **cheaper model** for sub-calls, following the paper's pattern of GPT-5 root with GPT-5-mini sub-calls
4. The sub-agent executes its own Read/Eval/Print/Loop cycle
5. On completion, the sub-agent's `$FINAL` variable is returned to the parent's REPL as the function return value
6. The parent continues execution with the result available as a REPL variable

**When to run non-recursive vs recursive Ripple**: This routing decision is critical and maps to the paper's observations about task characteristics:

| Scenario | Route | Rationale |
|----------|-------|-----------|
| Short task, low complexity (< 2K tokens input, simple question) | Direct LLM call (no Ripple) | RLM overhead > benefit (Observation 3) |
| Long input, low complexity (large file, simple extraction) | Non-recursive Ripple (chunking + code, no sub-agents) | Chunking via code is sufficient; recursion adds cost |
| Long input, high complexity (cross-referencing, multi-hop reasoning) | Recursive Ripple (full sub-agent spawning) | The OOLONG-Pairs scenario; recursion is essential |
| Short input, high complexity (deep reasoning about small code) | Direct LLM call with extended thinking | Model's native reasoning suffices; REPL adds overhead |

The routing heuristic uses: **input token count** (threshold: ~4K tokens for Ripple, ~32K for recursion), **estimated graph density** of the task (cross-references detected in the input), and **task type classification** (extraction vs. synthesis vs. reasoning).

### B4. MCP integration

**MCP client architecture**: The Rust core embeds an MCP client manager built on the **`rmcp` crate** (official Rust SDK, v0.15.0+). The manager maintains a `HashMap<ServerName, ClientSession>` where each `ClientSession` wraps a 1:1 connection to an MCP server. Connections use either stdio transport (for local servers launched as child processes) or Streamable HTTP transport (for remote servers). The manager implements the full MCP lifecycle: `initialize` with capability negotiation → operational phase → graceful shutdown.

**Discovery and namespacing**: On startup, the manager reads server configurations from three scopes (mirroring Claude Code's pattern): local project (`.agentique/mcp.json`), user global (`$AGENTIQUE_HOME/mcp.json`), and built-in defaults. Tool names are prefixed with server name to prevent collisions when presenting to the LLM: `github:create_issue`, `filesystem:read_file`. The aggregated tool list is cached and refreshed on `notifications/tools/list_changed`.

**Permission model**: Tool invocations follow a three-tier approval model:

- **Auto-approve**: Tools with `readOnlyHint: true` annotation from trusted servers (user-configured trust list)
- **Session-approve**: User approves once per session for a tool; subsequent calls auto-approved
- **Per-call approve**: Tools with `destructiveHint: true` or from untrusted servers require approval each time

Approval requests surface via ag-ui `TOOL_CALL_START` events with an approval flag, rendered as dialog prompts in the frontend.

**Auditing**: Every MCP tool invocation is logged as a tracing span with: server name, tool name, input arguments (redacted if sensitive), output summary, latency, and approval decision. The audit log persists in the session JSONL for post-hoc review.

**Caching**: `tools/list` results cached per-session (invalidated by `notifications/tools/list_changed`). `resources/read` results cached with TTL and invalidated by `notifications/resources/updated` subscriptions. Prompt templates cached indefinitely until `notifications/prompts/list_changed`.

### B5. Provider-agnostic LLM interface

**Core abstraction**: A Rust trait hierarchy provides clean separation:

```rust
#[async_trait]
trait CompletionProvider: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
    async fn complete_streaming(&self, request: CompletionRequest) 
        -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>>>>>;
    fn capabilities(&self) -> ProviderCapabilities;
    fn model_id(&self) -> &str;
}

struct CompletionRequest {
    messages: Vec<Message>,
    tools: Option<Vec<ToolDefinition>>,       // JSON Schema via schemars
    response_format: Option<ResponseFormat>,   // JSON schema for structured output
    max_tokens: u32,
    temperature: f32,
    stop_sequences: Vec<String>,
}

struct ProviderCapabilities {
    supports_tool_calling: bool,
    supports_structured_output: bool,
    supports_streaming: bool,
    supports_vision: bool,
    max_context_tokens: u32,
    cost_per_input_token: f64,
    cost_per_output_token: f64,
}
```

**Provider adapters**: Built on `reqwest` for HTTP with provider-specific request/response mapping. Initial adapters: OpenAI (via `async-openai` patterns), Anthropic (Messages API), Google Gemini, and an OpenAI-compatible adapter for local models (Ollama, vLLM). Each adapter handles its provider's streaming format (SSE with provider-specific JSON schemas), tool call format differences, and structured output mechanisms.

**Model router**: The router selects which provider/model to use for each request based on configurable policies:

- **Cost policy**: Route to cheapest model that meets capability requirements. Used for sub-agent calls in Ripple (paper's "cheaper model for sub-calls" strategy).
- **Capability policy**: Route based on required features (tool calling, vision, structured output, context window size).
- **Latency policy**: Route to fastest provider for interactive tasks.
- **Fallback chain**: If primary provider returns 429/500/timeout, automatically retry on next provider in chain with exponential backoff.

**Error handling**: Retries use jittered exponential backoff (initial 1s, max 60s, max 3 retries). Rate limit responses (429) extract `retry-after` header. Provider outages trigger fallback to next provider in chain. Malformed tool call JSON (a known issue with all models) triggers a re-prompt with the error and the schema.

### B6. Data layer

**Code and document indexing**: **Tantivy** (Rust's Lucene equivalent, ~2x faster) provides full-text search over indexed codebases and documents. The indexing pipeline: file watcher (`notify` crate) detects changes → **tree-sitter** parses source files into ASTs, extracting symbol definitions (functions, classes, imports) → Tantivy indexes both raw content and extracted symbols with file path, language, and symbol type as facets → **petgraph** maintains an in-memory dependency graph where nodes are files/symbols and edges are import/call/reference relationships.

**Graph representation**: The dependency graph uses petgraph's `StableGraph` (stable node indices survive deletions) with two edge types: `Imports` (file A imports symbol from file B) and `References` (symbol A calls/uses symbol B). This graph enables **PageRank-style relevance ranking** (the pattern Aider pioneered) — when a user asks about a function, the graph identifies the most-connected related symbols to include as context, far more precisely than vector similarity search. Incremental updates: on file change, tree-sitter re-parses only the changed file (incremental parsing), diffs the old and new symbol lists, and patches the graph.

**Local storage**: **SQLCipher** (SQLite + AES-256 encryption at rest) via `sqlx` + `libsqlite3-sys` with `bundled-sqlcipher` feature. Stores: session transcripts, artifact metadata, index metadata, user preferences, and MCP server configurations. Database key derived from OS keychain credential (`keyring` crate) — on macOS via Keychain, Windows via Credential Store, Linux via Secret Service.

**Secrets handling**: API keys and tokens stored in the OS keychain, never in plaintext config files. In-memory, sensitive values use the `memsecurity` crate for encrypted memory with zero-on-drop semantics. HTTP request headers carrying API keys are marked as sensitive in reqwest.

### B7. Observability and governance

**End-to-end tracing**: The `tracing` crate provides structured, span-based instrumentation. Every agent step creates a span: `agent.turn` → `ripple.loop_iteration` → `llm.completion` or `tool.execution` or `ripple.sub_agent`. Spans carry: step ID, parent step ID (for recursion tree), model used, token counts, wall time, cost. Spans are exported via `tracing-subscriber` to both the local session log and the frontend's recursion tree visualizer.

**Cost accounting**: Each LLM call records `input_tokens * cost_per_input_token + output_tokens * cost_per_output_token` from the provider capabilities registry. Costs aggregate hierarchically: sub-agent costs roll up to parent step, all steps roll up to session total. The frontend displays a running cost counter. **Budget enforcement** is a first-class guardrail: each session has a configurable budget ceiling (default: $1.00); each Ripple sub-agent receives a budget allocation; exceeding budget triggers graceful termination with partial results.

**Provenance**: Every artifact stores a `provenance` record: the chain of step IDs that produced it, the source data references (file paths, URLs, REPL variable names), and the model+prompt hash. When the agent cites a fact, the citation links to the specific step where that fact was extracted, which links to the specific source chunk. This creates **cite-able evidence** for all outputs.

**Safety guardrails**: Five layers of protection:

- **Recursion depth limit**: Configurable max depth (default: 3, paper used 1). Each sub-agent carries its current depth; spawning at max depth falls back to non-recursive.
- **Token budget**: Per-session and per-step ceilings prevent runaway costs. The paper notes high variance at the 95th percentile — budget caps are the primary mitigation.
- **Step limit**: Maximum iterations per Ripple loop (default: 50) prevents infinite loops.
- **Tool permissions**: The three-tier MCP approval model plus Tauri's capability ACL.
- **Content policy**: Output filtering for harmful content; input sanitization against prompt injection (tool descriptions from MCP servers treated as untrusted).

---

## C. Rust + Python mixing decision

### Option 1: Pure Rust (recommended)

Build the entire agent runtime in Rust using the mature crate ecosystem: `rig-core` or custom provider abstractions for LLM integration, `schemars` for JSON Schema generation, `rmcp` for MCP, `wasmtime` for sandboxed code execution, `tantivy` + `tree-sitter` + `petgraph` for indexing. No Python dependency.

**Pros**: Single binary distribution (~10 MB), no Python runtime to bundle, no GIL contention, maximum performance, Tauri-native (no sidecar lifecycle management), simplest packaging and cross-platform story, strongest security posture (memory-safe end-to-end, no interpreted code in the trusted core).

**Cons**: Rust LLM ecosystem is less mature than Python's (though `async-openai`, `rig-core`, and `genai` cover the major providers). No access to Python-only agent frameworks (pydantic-ai, LangGraph). Slower iteration on agent logic (compile times). Custom structured output parsing rather than Pydantic validators.

**Implementation sketch**: Provider adapters as Rust trait implementations using `reqwest` for HTTP. Tool definitions via `#[derive(JsonSchema)]` with `schemars`. Agent loop as async state machine on tokio. MCP client via `rmcp`. Code execution sandbox via `wasmtime` with WASI for filesystem access.

### Option 2: Rust core + Python agent kernel via PyO3

Embed CPython in the Rust process via PyO3. Agent logic written in Python (using pydantic-ai or similar), called from Rust via `Python::with_gil()`.

**Pros**: Full Python ecosystem access. Leverage pydantic-ai's model adapters and structured output validation. Fast iteration on agent logic.

**Cons**: **GIL blocks all Python-touching threads** — devastating for concurrent sub-agent execution in Ripple. Complex async bridging (pyo3-async-runtimes is experimental). Packaging nightmare: must bundle libpython + stdlib + site-packages (50-150 MB). PyO3 function call overhead (~40-70ns) is negligible, but GIL serialization is not. A Python crash takes down the entire Tauri process. PyOxidizer (the bundling tool) is effectively abandoned. Debugging requires simultaneous Python+Rust debuggers.

### Option 3: Sidecar agent service

Python agent service compiled to a standalone binary via PyInstaller, launched by Tauri as a sidecar process. Communication via HTTP (FastAPI on localhost) or stdio.

**Pros**: Clean process isolation (Python crash doesn't kill the app). Full Python ecosystem. Independent debugging. Proven Tauri sidecar pattern with community examples.

**Cons**: **30-150 MB** additional bundle size for the Python binary. IPC latency (~1-10ms per request) adds up across Ripple's many iterations. Sidecar lifecycle management complexity (startup, health checks, crash recovery, graceful shutdown). Two separate codebases to maintain. The Ripple REPL's tight loop (LLM → code → REPL → metadata → repeat) crosses the IPC boundary every iteration, amplifying latency.

### Option 4: WASM boundary

Compile agent logic to WASM, host in wasmtime embedded in Rust.

**Pros**: Perfect sandboxing. Language-agnostic module system.

**Cons**: **Not viable for agent workloads**. WASM has no native networking (can't call LLM APIs), no threading (can't parallelize sub-agents), 4 GB memory limit, and no Python package compatibility. WASI 0.3 (which adds async I/O) is not yet stable. Pyodide (CPython on WASM) lacks threading and real networking.

### Recommendation: Option 1 (Pure Rust)

The decision hinges on a single question: **does the agent logic critically need the Python ecosystem?** For Agentique Console, the answer is no.

The Python LLM ecosystem's advantages — pydantic-ai's provider adapters, LangGraph's workflow orchestration — are replicable in Rust with `rig-core`/`genai` for provider abstraction, `schemars`+`jsonschema` for structured output validation, and custom workflow orchestration. The Rust ecosystem covers every architectural need: `async-openai` for OpenAI (production-ready, full API coverage), `rmcp` for MCP (official SDK), `schemars` for JSON Schema, `wasmtime` for sandboxing, `tantivy` for search, `tree-sitter` for parsing.

The pure Rust approach eliminates the three worst risks: GIL contention killing Ripple's concurrent sub-agents, Python bundling complexity inflating the install, and cross-process IPC latency degrading the tight REPL loop. The Ripple engine's loop — where the LLM generates code, the REPL executes it, results append to metadata, and the loop continues — executes potentially dozens of iterations per task. Every millisecond of per-iteration overhead compounds. In-process Rust execution with zero serialization overhead is the right choice.

**Migration path**: If a Python-only model SDK becomes critical later, add it as an optional sidecar that handles only LLM API calls (not the agent loop), keeping the latency-sensitive Ripple loop in Rust.

| Factor | Pure Rust | PyO3 | Sidecar | WASM |
|--------|-----------|------|---------|------|
| Bundle size | **~10 MB** | ~100 MB | ~50-160 MB | ~10 MB |
| Ripple loop latency | **~μs** | ~ms (GIL) | ~ms (IPC) | N/A |
| Concurrent sub-agents | **Excellent** | Poor (GIL) | Good | None |
| Python ecosystem | None | Full | Full | None |
| Crash isolation | N/A (single process) | None | **Excellent** | Excellent |
| Packaging complexity | **Trivial** | Very high | Medium | Trivial |
| Security posture | **Strongest** | Weakest | Good | Strongest |

---

## D. Recursive execution strategy and guardrails

### Recursion depth limits and budgets

The RLM paper used synchronous sub-calls with **max recursion depth of 1** (sub-calls are plain LMs, not recursive). Agentique Console extends this to configurable depth (default max: **3**) but with strictly decreasing budgets at each level:

```
Depth 0 (root):  100% of session budget, full model
Depth 1:         budget_fraction × remaining_parent_budget, cheaper model
Depth 2:         budget_fraction × remaining_parent_budget, cheapest model
Depth 3:         fixed_micro_budget, cheapest model, no further recursion
```

`budget_fraction` defaults to **0.15** per sub-agent call. This means a root agent with $1.00 budget allocates at most $0.15 to each sub-agent, and depth-2 sub-agents get at most ~$0.02 each. The paper's finding that **median RLM cost is cheaper than base** justifies generous root budgets, while the **high 95th-percentile variance** justifies strict per-sub-agent caps.

**Loop detection**: Each Ripple REPL session tracks a hash of `(instruction, data_reference)` for every sub-agent call. If the same hash appears twice at the same depth, the call is blocked and the model receives an error message prompting it to try a different approach. This prevents the degenerate case of recursive self-calls with identical inputs.

**Step/time/token budgets**: Each Ripple loop iteration increments a step counter. Limits enforced: max 50 iterations per REPL session, max 300 seconds wall time per session, max 200K input tokens cumulative per session. The wasmtime sandbox enforces **fuel-based metering** on executed code — a runaway regex or infinite loop in model-generated code is interrupted after the fuel budget expires (epoch-based interruption cannot be bypassed by malicious WASM).

### Synchronous vs asynchronous recursion

The paper used synchronous sub-calls and notes async as future work. Agentique Console implements **async-by-default with a synchronous fallback**:

**Async recursion** (default): When the model generates code containing multiple `sub_RLM()` calls (e.g., in a loop chunking a document), calls are dispatched concurrently via `tokio::spawn`. Results are collected into a `Vec` and bound to a REPL variable. This directly addresses the paper's identified negative result that "synchronous calls are slow." For a document chunked into 10 segments, async recursion processes all chunks in parallel rather than sequentially, reducing wall time by up to ~10x.

**Safety of async recursion**: Concurrent sub-agents share no mutable state — each gets an independent REPL session with its own variable namespace. The only shared resource is the budget pool, protected by an `Arc<AtomicU64>` for lock-free budget decrementing. If the budget pool is exhausted, subsequent sub-agent spawns fail gracefully with a "budget exhausted" error that the parent model can handle (e.g., by synthesizing from partial results).

**Synchronous fallback**: Used when sub-agent calls are data-dependent (output of sub-agent A is input to sub-agent B). The Ripple engine detects dependency by analyzing the model-generated code: if `sub_RLM()` result is used as input to a subsequent `sub_RLM()`, those calls are serialized.

### Policies for when to spawn sub-agents

The Ripple engine does not blindly recurse. The model decides when to call `sub_RLM()`, but the runtime enforces heuristic guardrails and provides the model with decision-relevant metadata:

**Complexity heuristics injected into model context**:
- **Input size**: If `$P.token_count > 4096`, the REPL injects a hint: "This input exceeds single-call capacity. Consider chunking via $P.chunks(n) and processing via sub_RLM()."
- **Graph density**: If the dependency graph for the relevant code has > 20 cross-module edges, inject: "High cross-reference density detected. Consider recursive decomposition."
- **Uncertainty signal**: If a previous iteration's output confidence was flagged as low (model self-reported or heuristic-detected hedging language), suggest recursion for verification.

**Hard routing rules**:
- Input < 2K tokens AND task classified as simple → block sub_RLM(), force direct completion
- Remaining budget < $0.05 → block sub_RLM(), force synthesis from available data
- Depth = max_depth → sub_RLM() calls execute as plain (non-recursive) LLM calls

### Termination criteria

The RLM paper notes that **FINAL answer detection is brittle**. Agentique Console uses a multi-signal approach:

1. **Explicit**: Model sets `$FINAL = <value>` in REPL — the primary termination signal
2. **Implicit convergence**: If the last 3 iterations produced no new REPL variable assignments and no tool calls, terminate with the most recent non-empty variable as the result
3. **Budget exhaustion**: Graceful termination with partial results + "budget exhausted" annotation
4. **Step limit**: Hard stop at max iterations with partial results
5. **User interrupt**: User can inject a "stop and give me what you have" signal at any time via the Op queue

**"Give up / ask user" behavior**: If the agent reaches 80% of its step budget without setting `$FINAL` and without apparent progress (measured by: no new variables, repeated tool calls, or oscillating outputs), it enters `AwaitingUser` state with a summary of what it's found so far and a request for guidance.

### "Cheaper model for sub-agent" routing

Following the paper's validated pattern of GPT-5 root with GPT-5-mini sub-calls, the model router implements a **tier-based routing policy**:

```
Tier 1 (Root agent):        Most capable model (e.g., Claude Sonnet, GPT-5)
Tier 2 (Depth-1 sub-agent): Mid-tier model (e.g., Claude Haiku, GPT-5-mini)
Tier 3 (Depth-2+ sub-agent): Cheapest viable model (e.g., GPT-4o-mini)
```

The paper validates this approach: sub-agents perform focused, narrower tasks (extract facts from a chunk, filter results) that don't require the full reasoning capability of the root model. The cost savings compound — if a root task spawns 10 sub-agents, using a 10x cheaper model for sub-calls reduces total cost by ~5x. Users can override this policy per-session.

**Model capability floor**: The paper notes that "models without sufficient coding capabilities struggle" with RLM. The router enforces a minimum capability threshold: sub-agent models must support tool calling and produce syntactically valid code at an acceptable rate (tracked per-model from historical success rates). If a cheap model's code-generation success rate drops below 80%, the router escalates to the next tier.

---

## E. Development plan

### Milestones

**MVP (Months 1-3)**: Single-session agent with Ripple REPL, one LLM provider (OpenAI), basic file tools, non-recursive Ripple loop, Tauri shell with chat UI. Validates the core REPL-based execution model. Key metric: successfully processes a 100K-token codebase that exceeds context window.

**Alpha (Months 4-6)**: Recursive Ripple with depth-1 sub-agents, MCP client (stdio transport, single server), provider-agnostic interface (add Anthropic), code indexing with tree-sitter + tantivy, cost tracking, basic budget enforcement. Key metric: reproduces RLM paper results on OOLONG-equivalent tasks.

**Beta (Months 7-10)**: Async recursion, multi-MCP-server support, full observability (tracing + provenance), dependency graph with petgraph, session persistence and resumption, SQLCipher encrypted storage, security audit pass 1. Key metric: end-to-end workflow (repo understanding → code change → test verification) completes reliably.

**Production (Months 11-14)**: Depth-3 recursion, model routing optimization, plugin system, auto-updater, performance optimization pass, security audit pass 2, cross-platform distribution (Windows, macOS, Linux). Key metric: 95th-percentile cost within 5x of median for standard workflows; mean time to useful result < 60s for repo-understanding tasks.

### Prioritized backlog

**Must-have (MVP through Beta)**: Ripple REPL engine with symbolic variables; agent execution loop with tool calling; single-depth recursion; provider-agnostic LLM interface (OpenAI + Anthropic); MCP client (stdio); file read/write/search tools; tree-sitter code parsing; tantivy code search; Tauri shell with streaming chat; session persistence; cost tracking and budget enforcement; basic observability (tracing spans).

**Should-have (Beta through Production)**: Async recursion; multi-server MCP; dependency graph (petgraph); model routing policies; provenance citations; encrypted storage; recursion tree visualizer in UI; approval dialogs for destructive tools; session forking; compressor for long sessions; Streamable HTTP MCP transport; user-configurable recursion policies.

**Later**: Plugin marketplace for MCP servers; team collaboration features; cloud sync for sessions; custom model fine-tuning pipeline (the paper shows 1000 trajectories → 28.3% improvement for RLM-Qwen3-8B); multi-modal inputs (images, diagrams); voice interaction; A2A (agent-to-agent) protocol support.

### Testing strategy

**Unit tests**: Each Rust module has unit tests. Critical focus areas: REPL variable management (set/get/slice/search), budget accounting arithmetic, model router selection logic, JSON Schema generation and validation round-trips, MCP message serialization/deserialization.

**Integration tests**: End-to-end tests that run a real Ripple loop against a mock LLM server (returning scripted responses). Tests verify: correct tool call sequencing, budget enforcement at exact thresholds, recursion depth limiting, session persistence and resumption, MCP tool discovery and invocation.

**Golden trace tests**: Recorded sessions (JSONL rollouts) serve as regression baselines. Each test replays a golden trace and verifies that the agent's behavior matches (tool calls made, variables set, final output). When model behavior changes, traces are reviewed and re-baselined manually.

**Adversarial recursion tests**: Specifically designed to stress the guardrails: model that always calls `sub_RLM()` (tests depth limiting); model that generates infinite loops in code (tests wasmtime fuel metering); model that never sets `$FINAL` (tests step limit + convergence detection); model that consumes maximum tokens per call (tests budget enforcement); concurrent sub-agents that collectively exceed budget (tests atomic budget decrement).

**Performance testing**: Automated benchmarks measuring: P50/P95/P99 latency per Ripple iteration, P50/P95 total cost per standard task suite, throughput under concurrent sessions. **Regression gates**: P95 cost must not exceed 10x P50 for any benchmark task (the paper's long-tail cost distribution is the primary risk). CI fails if a code change increases P95 cost by > 20%.

### Security review checklist

- **Local file access**: All file operations pass through Tauri's scoped filesystem permissions. Agent cannot access files outside declared project roots. Path traversal attacks validated against.
- **Tool permissions**: MCP tool descriptions treated as untrusted input. Sanitized before inclusion in LLM prompt. Destructive tools require per-call user approval.
- **Prompt injection**: Model-generated code executed only in wasmtime sandbox. Tool outputs treated as untrusted data (not injected raw into prompts). System prompts include injection-resistant framing.
- **Data exfiltration**: Sandboxed code has no network access by default. MCP servers' `openWorldHint` annotation triggers explicit user notification. API keys never included in LLM context.
- **Secrets**: API keys in OS keychain, encrypted in memory, zeroed on drop. SQLCipher for data at rest. No plaintext credentials in config files.
- **Supply chain**: Tauri's Isolation Pattern encrypts IPC. Cargo dependency audit via `cargo-vet`. CSP headers restrict WebView script sources.

---

## F. Concrete starter blueprint

### Proposed repo structure

```
agentique-console/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── agentique-core/                 # Agent runtime core
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── agent_loop.rs           # Planning/execution state machine
│   │   │   ├── session.rs              # Session state, persistence, compaction
│   │   │   ├── tool_router.rs          # Tool registry and dispatch
│   │   │   ├── artifact.rs             # Artifact store and provenance
│   │   │   └── budget.rs               # Cost accounting and enforcement
│   │   └── Cargo.toml
│   │
│   ├── ripple-engine/                  # Recursive LLM execution environment
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── repl.rs                 # REPL session: variables, operations
│   │   │   ├── executor.rs             # Code execution (wasmtime sandbox)
│   │   │   ├── recursion.rs            # Sub-agent spawning, budget allocation
│   │   │   ├── routing.rs              # Recursive vs non-recursive decision
│   │   │   └── termination.rs          # Convergence detection, FINAL extraction
│   │   └── Cargo.toml
│   │
│   ├── llm-provider/                   # Provider-agnostic LLM interface
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── traits.rs               # CompletionProvider trait
│   │   │   ├── router.rs               # Model routing (cost/capability/fallback)
│   │   │   ├── openai.rs               # OpenAI adapter
│   │   │   ├── anthropic.rs            # Anthropic adapter
│   │   │   ├── google.rs               # Gemini adapter
│   │   │   └── streaming.rs            # SSE parsing, chunk aggregation
│   │   └── Cargo.toml
│   │
│   ├── mcp-manager/                    # MCP client manager
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── client.rs               # Single-server client session
│   │   │   ├── manager.rs              # Multi-server lifecycle + namespacing
│   │   │   ├── permissions.rs          # Tool approval policies
│   │   │   └── config.rs               # Server configuration loading
│   │   └── Cargo.toml
│   │
│   ├── data-layer/                     # Indexing, graphs, storage
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── indexer.rs              # Tantivy indexing pipeline
│   │   │   ├── parser.rs               # Tree-sitter AST extraction
│   │   │   ├── graph.rs                # Petgraph dependency graph
│   │   │   ├── storage.rs              # SQLCipher database
│   │   │   └── secrets.rs              # Keyring + memsecurity
│   │   └── Cargo.toml
│   │
│   └── observability/                  # Tracing, cost, provenance
│       ├── src/
│       │   ├── lib.rs
│       │   ├── tracer.rs               # Tracing span management
│       │   ├── cost.rs                 # Per-step cost accumulation
│       │   ├── provenance.rs           # Citation chain tracking
│       │   └── guardrails.rs           # Safety policy enforcement
│       └── Cargo.toml
│
├── src-tauri/                          # Tauri application shell
│   ├── src/
│   │   ├── main.rs                     # Tauri app bootstrap
│   │   ├── commands.rs                 # IPC command handlers
│   │   └── events.rs                   # ag-ui event emitter
│   ├── capabilities/                   # Tauri ACL capability files
│   │   ├── default.json
│   │   └── agent-tools.json
│   ├── tauri.conf.json
│   └── Cargo.toml
│
├── frontend/                           # React + ag-ui WebView app
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ChatPanel.tsx           # Message streaming display
│   │   │   ├── RecursionTree.tsx       # Visual recursion tree
│   │   │   ├── ArtifactViewer.tsx      # Code diffs, documents
│   │   │   ├── CostDashboard.tsx       # Budget tracking display
│   │   │   └── ToolApproval.tsx        # Permission dialogs
│   │   ├── hooks/
│   │   │   └── useAgentEvents.ts       # ag-ui event consumer
│   │   └── lib/
│   │       └── ag-ui-client.ts         # ag-ui event types + state
│   ├── package.json
│   └── tsconfig.json
│
├── tests/
│   ├── golden-traces/                  # Recorded session JSONL files
│   ├── adversarial/                    # Recursion stress tests
│   └── benchmarks/                     # Performance regression tests
│
└── docs/
    ├── architecture.md
    ├── ripple-design.md
    └── mcp-integration.md
```

### Core Rust module boundaries

The workspace is organized around **six crates** with strict dependency direction (no cycles):

```
observability  ←── agentique-core ──→ ripple-engine
                        │                    │
                        ▼                    ▼
                   mcp-manager          llm-provider
                        │
                        ▼
                    data-layer
```

`agentique-core` is the orchestrator — it depends on all other crates. `ripple-engine` depends on `llm-provider` (to make LLM calls) but not on `mcp-manager` (MCP tools are injected via the `Tool` trait). `mcp-manager` depends on `data-layer` for configuration persistence. `observability` is a leaf crate depended on by all others for tracing macros.

### Minimal protocol definitions

**Tool call protocol** (between agent loop and tool implementations):

```rust
// Tool definition sent to LLM
struct ToolDefinition {
    name: String,             // "mcp:github:create_issue" or "builtin:file_read"
    description: String,
    parameters: Value,        // JSON Schema from schemars
    annotations: ToolAnnotations,
}

// Tool invocation from LLM response
struct ToolCall {
    id: String,               // Unique call ID for correlation
    name: String,
    arguments: Value,         // JSON matching parameters schema
}

// Tool result returned to agent loop
struct ToolResult {
    call_id: String,
    content: Vec<ContentBlock>,  // Text, image, or resource_link
    is_error: bool,
    cost: Option<CostRecord>,
    duration_ms: u64,
}
```

**Step trace protocol** (for observability):

```rust
struct StepTrace {
    step_id: Uuid,
    parent_step_id: Option<Uuid>,   // For recursion tree
    depth: u8,
    step_type: StepType,            // LlmCall, ToolExecution, SubAgent, ReplEval
    model_id: Option<String>,
    input_tokens: u32,
    output_tokens: u32,
    cost_usd: f64,
    wall_time_ms: u64,
    variables_changed: Vec<String>, // REPL variables modified
    status: StepStatus,             // Running, Completed, Failed, Cancelled
}
```

**Provider interface** (the core LLM abstraction):

```rust
#[async_trait]
trait CompletionProvider: Send + Sync {
    async fn complete(
        &self, 
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError>;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ProviderError>> + Send>>;

    fn capabilities(&self) -> &ProviderCapabilities;
}

enum StreamChunk {
    TextDelta(String),
    ToolCallStart { id: String, name: String },
    ToolCallArgsDelta { id: String, delta: String },
    ToolCallEnd { id: String },
    Usage { input_tokens: u32, output_tokens: u32 },
    Done,
}
```

### Sample Ripple loop pseudocode

This pseudocode implements the full RLM Algorithm 1 with a single recursive handoff, written in language-agnostic style that maps directly to the Rust implementation:

```
function ripple_execute(prompt, config, budget, depth):
    // === INIT REPL (Algorithm 1, line 1-2) ===
    repl = new ReplSession()
    repl.set("$P", prompt)                    // Symbolic handle to prompt
    repl.set("$P_META", {
        token_count: count_tokens(prompt),
        type: detect_type(prompt),            // "code_repo" | "document" | "data"
        structure: extract_structure(prompt),  // file list, headings, schema
    })
    
    // Add sub_RLM function to REPL (Algorithm 1, line 3)
    repl.define_function("sub_RLM", (data_ref, instruction) => {
        if depth >= config.max_depth:
            // At max depth, sub-calls are plain LLM calls (paper's design)
            return llm_complete(config.sub_model, instruction + "\n" + data_ref)
        
        sub_budget = min(budget.remaining * config.budget_fraction, 
                         config.max_sub_budget)
        if sub_budget < config.min_viable_budget:
            return Error("Budget exhausted for sub-agent")
        
        // RECURSIVE HANDOFF: spawn sub-agent with cheaper model
        sub_model = config.model_router.select_for_depth(depth + 1)
        return ripple_execute(data_ref, config, sub_budget, depth + 1)
    })
    
    // Add utility functions
    repl.define_function("search", (var, pattern) => regex_search(repl.get(var), pattern))
    repl.define_function("chunk", (var, n) => split_into_chunks(repl.get(var), n))
    repl.define_function("slice", (var, start, end) => substring(repl.get(var), start, end))
    
    // === MAIN LOOP (Algorithm 1, lines 4-8) ===
    messages = [system_prompt(config, repl.metadata())]
    step_count = 0
    
    while step_count < config.max_steps AND budget.remaining > 0:
        step_count += 1
        
        // READ: Present current REPL state to LLM
        messages.append(user_message(format_repl_state(repl)))
        
        // LLM generates code (Algorithm 1, line 5)
        trace = start_trace("ripple.iteration", depth, step_count)
        response = llm_complete(
            model: config.model_for_depth(depth),
            messages: messages,
            tools: repl.available_functions_as_tool_defs(),
            budget: budget,
        )
        budget.deduct(response.cost)
        trace.record(response.usage)
        
        // Check for FINAL in response (Algorithm 1, line 7)
        if response.sets_variable("$FINAL"):
            repl.set("$FINAL", response.extract_final_value())
            trace.complete("final_set")
            break
        
        // EVALUATE: Execute LLM-generated code in sandbox (Algorithm 1, line 6)
        code_blocks = extract_code_blocks(response)
        for block in code_blocks:
            sandbox_result = sandbox_execute(block, repl, config.fuel_limit)
            
            if sandbox_result.is_error:
                // PRINT: Append error metadata (Algorithm 1, line 6b)
                messages.append(system_message(
                    "Execution error: " + sandbox_result.error + 
                    "\nREPL state: " + format_repl_state(repl)
                ))
            else:
                // PRINT: Append execution result metadata
                repl.apply_changes(sandbox_result.variable_changes)
                messages.append(system_message(
                    "Executed successfully. Variables changed: " + 
                    sandbox_result.variable_changes.keys() +
                    "\nREPL state: " + format_repl_state(repl)
                ))
        
        // LOOP: Check convergence (implicit termination)
        if last_3_iterations_no_progress(messages):
            repl.set("$FINAL", repl.get_best_candidate_result())
            trace.complete("convergence")
            break
        
        trace.complete("continue")
    
    // === TERMINATION ===
    if not repl.has("$FINAL"):
        // Budget or step limit reached without result
        repl.set("$FINAL", repl.get_best_candidate_result())
        emit_warning("Terminated without explicit FINAL: " + termination_reason)
    
    return {
        result: repl.get("$FINAL"),
        cost: budget.total_spent,
        steps: step_count,
        depth: depth,
        trace: collect_traces(),
        provenance: repl.variable_lineage("$FINAL"),
    }


// === EXAMPLE: Multi-document synthesis with recursive handoff ===
// 
// User asks: "Compare the error handling strategies across all modules in this repo"
// 
// Root agent (depth 0, GPT-5):
//   1. repl.set("$P", <entire repo contents as symbolic handle>)
//   2. LLM generates: modules = $P.search("mod.rs|lib.rs"); chunks = chunk(modules, 5)
//   3. LLM generates: for i, chunk in chunks:
//                        $results[i] = sub_RLM(chunk, "Extract error handling patterns")
//   4. Sub-agents (depth 1, GPT-5-mini) each process ~5 modules, return structured findings
//   5. Root agent synthesizes: $FINAL = merge_and_compare($results)
```

This pseudocode directly implements the three key design choices from the RLM paper: **symbolic handle** (`$P` loaded as REPL variable, never copied into LLM context), **variable-based output** (`$FINAL` and intermediate `$results` as REPL variables), and **symbolic recursion** (`sub_RLM()` as a programmatic function callable in loops). The sandbox execution via wasmtime ensures model-generated code cannot escape its resource bounds, and the budget accounting at every step addresses the paper's identified risk of long-tailed cost distributions.

---

## Conclusion: what this architecture makes possible

Agentique Console's architecture resolves the fundamental tension in current AI coding tools: **context windows are necessary but not sufficient for complex tasks**. The RLM paper demonstrated that programmatic decomposition via a REPL with recursive sub-calls outperforms both context-stuffing and RAG approaches by a wide margin on complex, long-context tasks. This architecture makes that capability production-viable through three key innovations.

First, the **Ripple engine** transforms the RLM algorithm from a research prototype into an observable, budgeted, permission-controlled runtime. The paper's identified negative results — brittle FINAL detection, synchronous slowness, high cost variance — are each addressed by specific guardrails: multi-signal termination, async sub-agent dispatch, and hierarchical budget enforcement with per-step caps.

Second, the **pure Rust decision** eliminates the most dangerous architectural risk for a REPL-based agent: latency per iteration. Every Ripple loop iteration crosses the LLM → code → REPL → metadata boundary. In-process Rust execution with zero serialization overhead keeps this loop tight, while wasmtime sandboxing provides the security boundary that a REPL executing model-generated code demands.

Third, the **MCP-first tool architecture** means Agentique Console is not a monolithic product but a platform. Every tool — filesystem, git, database, API — is an MCP server that can be added, removed, and permissioned independently. The agent's capabilities grow with the MCP ecosystem rather than requiring first-party development for each integration. Combined with the provider-agnostic LLM layer, no single vendor dependency exists in the architecture.

The development plan sequences risk: MVP validates that the REPL-based execution model works for real coding tasks, Alpha validates recursion, Beta validates production readiness. The 95th-percentile cost regression gate in CI directly targets the RLM paper's identified risk of long-tailed cost distributions, ensuring that the capability gains of recursive execution do not come with unbounded cost exposure.