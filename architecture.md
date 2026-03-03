# Agentique Console: architectural blueprint for an agentic coding system

**Agentique Console is a desktop multi-agent coding system built on Rust, Tauri v2, and the Model Context Protocol, designed around a single unifying abstraction: the Adaptive Execution Tree.** This tree structure simultaneously models three concerns—concurrent task execution, cost budget propagation, and real-time observability—within one recursive data structure. Every agent invocation, every LLM call, every tool execution occupies a node in this tree. When a parent node cancels, its children cancel. When a child spends tokens, the cost propagates upward. When an observer inspects the tree, they see the full execution trace. One tree, three interpretations, zero impedance mismatch.

This blueprint synthesizes Google's eight documented multi-agent design patterns into six composable execution primitives that, when combined recursively, can express any multi-agent workflow—from a trivial single-shot code explanation to a complex generate→test→diagnose→fix→retest cycle spanning dozens of specialized agents. The architecture prioritizes simplicity over cleverness, composition over configuration, and deterministic orchestration augmented by LLM-driven routing only where dynamic judgment is genuinely required.

---

## Five principles that govern every architectural decision

The design philosophy for Agentique Console emerges from a critical observation across production multi-agent systems: **coordination failures, not model failures, are the dominant source of system-level problems.** The "prompting fallacy"—believing prompt tweaks can fix systemic architecture issues—leads teams to optimize the wrong layer. These five principles address coordination at the structural level.

**Principle 1: Structured concurrency is non-negotiable.** Every spawned agent task must have a well-defined parent, a bounded lifetime, and guaranteed cleanup on cancellation. In Rust, this means every agent subtask lives inside a `JoinSet` or equivalent scope. Dropping the parent aborts all children. No orphaned LLM calls burning tokens against stale context. No zombie tool executions completing after their results became irrelevant. The execution tree IS the concurrency tree—they are the same data structure.

**Principle 2: Deterministic orchestration by default, LLM-driven routing by exception.** Google's pattern taxonomy draws a sharp line between workflow agents (Sequential, Parallel, Loop) that orchestrate via predefined logic and coordinator agents that use LLM reasoning to route. The lesson: **LLM-driven orchestration is expensive, latency-adding, and error-prone.** Use it only at decision points where the task genuinely requires natural-language understanding to route—typically the top-level intent classification and ambiguous decomposition steps. Everything else should be deterministic: sequences execute in order, parallel fans execute concurrently, loops iterate until hard exit conditions are met.

**Principle 3: Budget is a first-class structural concern, not an afterthought.** Every node in the execution tree carries a budget allocation (tokens, dollars, wall-clock time, iteration count). Budget flows downward through the tree via allocation and upward via consumption reporting. When a budget exhausts, the node's `CancellationToken` fires, propagating cancellation through structured concurrency. This makes runaway cost impossible at the architectural level rather than requiring ad-hoc guards.

**Principle 4: Composition over taxonomy.** Rather than designing for a fixed set of "agent types," the system provides **six small, orthogonal execution primitives** that compose recursively. Any workflow—sequential pipelines, parallel fan-outs, iterative refinement loops, dynamic routing, human approval gates—is expressed as a tree of these primitives. New patterns emerge from composition without new framework abstractions.

**Principle 5: Tools are deterministic; agents reason.** Following the pattern validated in Google's Code Review Assistant codelab, deterministic operations (AST parsing, test execution, linting, file I/O) should be performed by tools with predictable behavior, not delegated to LLM reasoning. Agents decide *what* to do; tools do it with guaranteed correctness. This separation makes the system testable, reproducible, and cost-efficient—the LLM is never invoked when a `tree-sitter` query or a `cargo test` invocation would suffice.

---

## The Adaptive Execution Tree: a unified master pattern

Google's architecture documentation describes eight multi-agent patterns, each with distinct strengths. The Single Agent pattern is simplest but cannot handle complex tasks. Sequential pipelines are deterministic but rigid. Parallel fan-outs reduce latency but multiply cost. Loop patterns enable iterative refinement but risk runaway execution. The Coordinator pattern adds dynamic flexibility but introduces an LLM bottleneck. Iterative Refinement (Generator-Critic) maps directly to code generation workflows. Human-in-the-Loop provides safety gates. And the Custom Logic pattern—the most honest of the eight—admits that production systems compose all of the above.

The Adaptive Execution Tree (AXT) formalizes this composition. It defines **six execution primitives** that can be nested to arbitrary depth, creating a recursive tree that expresses any multi-agent workflow:

### The six primitives

**Leaf** executes a single agent: one LLM call (or chain of calls) with access to a set of tools. This is the atomic unit of work. A Leaf corresponds to Google's Single Agent pattern. In Rust, a Leaf is an async function that receives a context (containing budget, state, and cancellation token) and produces a result.

**Sequence** executes its children in order, threading state from one to the next. Each child's output enriches the shared state before the next child reads it. This maps to Google's Sequential pattern and is implemented as a loop over children with `await` between each. Sequence is the workhorse for deterministic multi-step pipelines: parse requirements → generate code → write tests → report results.

**Parallel** executes all children concurrently using a `JoinSet`, then gathers results into a synthesis step. This maps to Google's Parallel pattern. Each child receives its own budget allocation (the parent's remaining budget divided according to a configurable policy). A `Semaphore` limits concurrent LLM calls to prevent API rate-limit violations. Parallel is ideal for independent analysis tasks: run security audit, style check, and performance analysis simultaneously.

**Loop** executes its children repeatedly until an exit condition is met. Exit conditions include: a quality predicate on shared state (e.g., all tests pass), a maximum iteration count, a budget exhaustion signal, or a stagnation detector (no improvement across N iterations). This maps to Google's Loop and Iterative Refinement patterns. **Loop is the most critical primitive for coding workflows**—the generate→test→fix cycle is fundamentally iterative. Every Loop node carries a hard `max_iterations` ceiling; there is no configuration that allows unbounded looping.

**Route** selects one (or more) children to execute based on a routing decision. The router can be **rule-based** (a match expression on task metadata), **classifier-based** (a lightweight model classifying intent), or **LLM-driven** (the Coordinator pattern). Route is the only primitive where LLM-driven orchestration is used, and only when static routing is insufficient. This maps to Google's Coordinator pattern. In the coding system, the top-level Route classifies user intent (generate, refactor, debug, explain, test) and dispatches to the appropriate sub-tree.

**Gate** pauses execution and persists full tree state to durable storage, awaiting an external signal—typically human approval. This maps to Google's Human-in-the-Loop pattern. When the Gate resumes, execution continues from the checkpoint with potentially modified parameters. Gate enables deterministic replay: the persisted state is a checkpoint from which execution can be re-run with different inputs (a different model, a modified prompt, an increased budget).

### How composition works in practice

The coding workflow in Agentique Console composes these primitives into a tree that handles the full lifecycle of a coding task:

```
Route (intent classifier — lightweight model or rules)
├── "generate" → Loop (max_iterations=5, exit=tests_pass ∧ lint_clean)
│   └── Sequence
│       ├── Leaf (CodeGenerator — writes or modifies code)
│       ├── Parallel
│       │   ├── Leaf (TestRunner — executes test suite)
│       │   ├── Leaf (Linter — runs static analysis)
│       │   └── Leaf (SecurityScanner — checks vulnerabilities)
│       ├── Leaf (ResultSynthesizer — merges parallel outputs)
│       └── Leaf (LoopEvaluator — checks exit conditions, writes to state)
├── "debug" → Loop (max_iterations=5, exit=error_resolved)
│   └── Sequence
│       ├── Leaf (ErrorAnalyzer — diagnoses failure from stack trace + code)
│       ├── Leaf (HypothesisGenerator — proposes fixes with reasoning)
│       ├── Leaf (FixApplier — modifies code)
│       └── Leaf (TestValidator — confirms fix)
├── "refactor" → Sequence
│   ├── Leaf (RefactorPlanner — analyzes code, proposes changes)
│   ├── Gate (human approval of refactoring plan)
│   ├── Leaf (RefactorExecutor — applies transformations)
│   └── Leaf (RegressionTester — ensures no breakage)
├── "explain" → Leaf (CodeExplainer — single-shot analysis)
└── "test" → Sequence
    ├── Leaf (TestGenerator — writes test cases)
    ├── Leaf (TestRunner — executes tests)
    └── Leaf (CoverageAnalyzer — reports coverage gaps)
```

This tree is not a static configuration—it is constructed dynamically based on the task. Simple requests produce shallow trees (a single Leaf). Complex multi-file refactors produce deep trees with nested Loops and Parallels. The tree-construction logic itself is deterministic code, not LLM reasoning, except at Route nodes.

### Rust implementation sketch

The core type is a recursive enum that Rust's ownership model makes safe to compose:

```rust
pub enum ExecutionNode {
    Leaf { agent: AgentConfig, tools: Vec<ToolRef> },
    Sequence { children: Vec<ExecutionNode> },
    Parallel { children: Vec<ExecutionNode>, concurrency_limit: usize },
    Loop { body: Box<ExecutionNode>, exit: ExitCondition, max_iterations: u32 },
    Route { router: RouterKind, branches: IndexMap<String, ExecutionNode> },
    Gate { checkpoint: CheckpointConfig, continuation: Box<ExecutionNode> },
}
```

Execution is a recursive async function that pattern-matches on the node type, spawning child tasks into a `JoinSet` for Parallel nodes, iterating with state checks for Loop nodes, and persisting state for Gate nodes. Each execution call receives an `ExecutionContext` containing the parent's `CancellationToken` (child tokens are derived), the budget scope, the shared state handle, and the OTel span context.

---

## Agent roles in a coding-native workflow

The Adaptive Execution Tree defines *structure*. Agents define *behavior*. Following the ADK taxonomy, Agentique Console distinguishes three agent categories that serve different roles within the tree.

### LLM agents are the reasoning core

LLM agents wrap a model call with a system prompt, tool bindings, and output constraints. They occupy Leaf nodes in the execution tree. Each LLM agent has a **clear, narrow role** with explicit instructions about what it should and should not do. The ADK codelab's anti-pattern finding is critical here: without explicit negative instructions ("do NOT fix the code before analyzing it"), LLMs "helpfully" modify inputs, corrupting downstream agents' assumptions.

Key LLM agent roles for coding:

The **CodeGenerator** agent produces or modifies source code given requirements and existing context. It receives the current file state, relevant code snippets (retrieved via the code intelligence subsystem), and the user's intent. It outputs a structured diff or complete file content. This agent uses the most capable model available within the budget, because code generation quality has the highest leverage on overall task success.

The **ErrorAnalyzer** agent examines stack traces, test failures, and compiler errors to produce a structured diagnosis: root cause hypothesis, affected files/functions, and suggested fix approach. This agent benefits from chain-of-thought prompting and access to the project's error history (episodic memory).

The **CodeReviewer** agent evaluates generated code against configurable criteria: correctness, style conformance, security patterns, performance implications. It operates as the "critic" in the Generator-Critic pattern. Its output is a structured review with pass/fail verdict and specific, actionable feedback that the CodeGenerator can consume in the next loop iteration.

The **RefactorPlanner** agent analyzes code structure (via tree-sitter AST data) and proposes refactoring operations as a structured plan. This agent uses the critic-tier model (more capable, higher cost) because architectural decisions require nuanced reasoning.

### Workflow agents orchestrate without reasoning

Workflow agents correspond to the non-Leaf nodes: Sequence, Parallel, Loop, Route. They execute predefined logic—no LLM calls, no token consumption, no latency from model inference. Their logic is pure Rust code: iterate over children, spawn concurrent tasks, evaluate exit conditions, route based on classifiers. The Google documentation emphasizes this separation: **workflow agents handle control flow; LLM agents handle reasoning.** Conflating the two produces systems that are expensive, slow, and hard to debug.

### Custom agents bridge deterministic tools

Custom agents wrap deterministic operations—test execution, linting, file I/O, git operations, build system invocations—in the agent interface so they can participate in the execution tree. They consume no LLM tokens. The TestRunner custom agent executes `cargo test` (or the project's test command), captures stdout/stderr, and writes structured results (pass/fail per test, error messages, timing) to shared state. The Linter custom agent invokes language-specific linters and writes violations to state. These agents are fast, cheap, and reproducible—exactly what the "tools are deterministic" principle demands.

### Model routing optimizes cost across the tree

Not every Leaf node needs the same model. The **LLM Router** selects models based on three signals: **recursion depth** in the execution tree (deeper nodes handle simpler sub-tasks and can use cheaper models), **task classification** (code explanation → fast/cheap model; architecture decisions → capable/expensive model), and **remaining budget** (as budget depletes, route to cheaper models or refuse to proceed). This cascading classification approach means the system naturally uses expensive models only where they provide the most value.

The routing table is configurable but ships with sensible defaults. Initial intent classification and loop evaluation use a lightweight model (minimal token cost for a routing decision). Code generation and complex debugging use the most capable available model. Linting, formatting, and simple explanations use the cheapest tier. This maps to the ADK pattern of `worker_model` for mechanical tasks and `critic_model` for nuanced reasoning.

---

## Memory architecture spans four tiers

Agent memory is the single most underestimated subsystem in coding assistants. Without effective memory, every session starts cold, every context window fills with redundant information, and agents cannot learn from past interactions. Agentique Console implements a **four-tier memory architecture** inspired by the CoALA cognitive architecture taxonomy, with each tier optimized for different access patterns and lifetimes.

### Tier 1: Working memory is the active context window

Working memory is what the LLM can "see" right now—the contents of its context window for the current invocation. It includes the system prompt, the current conversation turn, retrieved code snippets, and recent tool outputs. Working memory is ephemeral and reconstructed for each Leaf node execution.

The critical challenge is **context rot**: LLM performance degrades as context fills, even within technical token limits. Research from JetBrains and the Manus team establishes a three-level management hierarchy that Agentique Console follows strictly. **Raw context** (verbatim tool outputs and code) is preferred when it fits. **Observation masking** replaces older tool results with compact references ("Previous 47 lines of test output elided; see artifact test-results-v3")—this is surprisingly effective, performing equal to or better than LLM summarization on coding benchmarks while consuming zero additional tokens. **LLM summarization** is the last resort, used only when masking cannot reduce context below the pre-rot threshold (typically **60-70% of the model's context limit**).

A hopping context window mechanism prevents stop-the-world compaction pauses. At 70% capacity, a background task checkpoints the conversation via summary into a secondary buffer. The agent continues working in the primary buffer. When the primary hits capacity, execution swaps to the secondary buffer (which contains the summary plus all messages since the checkpoint). This double-buffering technique, inspired by graphics rendering, eliminates the latency spike of mid-conversation summarization.

### Tier 2: Session memory persists within a task

Session memory stores the full execution trace of the current coding task—every agent invocation, every tool result, every state mutation. It is backed by an in-memory store (shared state accessible to all agents in the execution tree) augmented with durable append-only logs for crash recovery. Session state uses typed keys (`StateKey<T>`) to prevent the subtle bugs that plague stringly-typed shared state in production ADK systems.

The shared state acts as a **whiteboard**: agents write results to named keys, and downstream agents read them. The Sequence primitive naturally threads state—each child enriches the whiteboard for the next. The Parallel primitive requires careful key management—parallel children must write to distinct keys to avoid race conditions, with a synthesis step that reads all parallel outputs and writes a merged result.

### Tier 3: Project memory accumulates across sessions

Project memory stores knowledge that persists across coding sessions for a specific codebase. It includes: the code index (tree-sitter AST data + Tantivy full-text index), the project's architectural conventions, recurring error patterns and their solutions, test failure history, and consolidated learnings from past sessions.

Project memory is backed by a `.agentique/` directory within the project root, containing SQLite databases for structured data, the Tantivy index directory, and markdown files for human-readable project conventions (following the CLAUDE.md pattern, which benchmarks show is surprisingly competitive with vector-database approaches for project-level memory). An **episodic-to-semantic consolidation process** runs at session end: the session trace (episodic memory) is analyzed to extract generalizable patterns ("whenever tests fail with error X, the fix is approach Y"), which are written to semantic project memory for future retrieval.

### Tier 4: User memory personalizes across projects

User memory stores preferences, coding style patterns, frequently used tools, and model preferences that apply across all projects. It enables the system to learn that a specific developer prefers explicit error handling over `unwrap()`, favors functional patterns, or wants detailed explanations. User memory is stored in `~/.agentique/` and loaded at startup.

### Context compaction keeps costs bounded

The four-tier architecture creates a natural compaction pipeline. When working memory approaches capacity, the system consults session memory for relevant prior context rather than keeping everything in the window. When session memory grows large, key insights are promoted to project memory. This hierarchical offloading means **the context window stays focused on what matters right now**, while the full history remains accessible through retrieval.

For the iterative generate→test→fix loop specifically, context management is critical because each iteration adds code, test results, and error analysis. The compaction strategy: keep the *current* code and *most recent* test results in full (raw context), mask previous iteration results with references, and maintain a running summary of what was tried and why it failed (to prevent the agent from repeating failed approaches).

---

## Code intelligence: where tree-sitter meets Tantivy

A coding agent is only as effective as its understanding of the codebase. Agentique Console builds a **code intelligence pipeline** that combines structural understanding (via tree-sitter ASTs) with fast full-text retrieval (via Tantivy), creating a hybrid search system that understands both the meaning and the text of code.

### Incremental AST indexing with tree-sitter

Tree-sitter produces concrete syntax trees for **40+ programming languages** with incremental parsing—when a file changes, only the affected AST subtree is re-parsed, completing in sub-milliseconds. The code intelligence subsystem maintains a persistent AST cache per project, updated incrementally on file save events.

From each AST, the system extracts a **semantic skeleton**: function/method signatures, class/struct definitions with inheritance relationships, import/module dependency graphs, exported symbols, and documentation strings. This extraction uses tree-sitter's S-expression query language, with per-language query files that define the patterns to extract. The Aider project validates this approach: their graph-based AST retrieval achieves **4.3-6.5% context utilization** (meaning the agent needs to see only 4-6% of the codebase to complete tasks), with no GPU, no embedding model, and no vector database required.

The semantic skeleton serves two purposes. First, it provides the **repository map**—a compact representation of the codebase's structure that fits within a few thousand tokens and gives agents enough context to navigate the project without seeing every line of code. Second, it feeds the **dependency graph**, which the code intelligence subsystem uses to determine what code is affected by a change (the foundation for the Ripple Engine's change propagation analysis).

### Full-text code search with Tantivy

Tantivy provides Apache Lucene-grade full-text search as a native Rust library. The code search index uses a schema designed for code-specific queries:

- **`file_path`** (stored, filterable): enables scoping searches to directories or file types
- **`symbol_name`** (text, stored): function/class/variable names, searchable with language-aware tokenization
- **`symbol_kind`** (keyword): function, class, struct, enum, trait, interface—enables structural filtering
- **`content`** (text): the full source code of the semantic unit, searchable with code-aware tokenizers that handle snake_case, camelCase, and dotted.paths
- **`language`** (keyword): programming language for filtering
- **`line_range`** (fast field): start and end line numbers for jump-to-source navigation

Index updates are incremental: on file change, delete old documents for that file, re-parse with tree-sitter, create new documents per semantic unit, and let Tantivy's segment merge handle compaction in the background. The **per-semantic-unit granularity** (one Tantivy document per function/class rather than per file) means search results are precise—an agent searching for "authentication handler" gets the relevant function, not the entire 2000-line file.

### Retrieval pipeline for agent context

When an agent needs codebase context (during code generation, debugging, or refactoring), the retrieval pipeline executes a three-stage process. First, **keyword extraction** from the task description identifies search terms. Second, **Tantivy query** retrieves the top-N relevant code snippets by text relevance. Third, **graph expansion** uses the dependency graph to pull in structurally related code (callers, callees, type definitions) that the text search might miss. The combined results are ranked by a scoring function that weights text relevance, structural proximity to the edit target, and recency of modifications. This pipeline runs entirely on CPU, requires no external services, and completes in single-digit milliseconds for codebases up to ~1M lines.

---

## Budget governance propagates through the execution tree

Cost control in multi-agent systems is fundamentally a tree problem. A user says "spend at most $2 on this task." That $2 must be distributed across potentially dozens of agent invocations spanning multiple loop iterations and parallel branches. The Adaptive Execution Tree makes this natural: **budget is a resource that flows downward through allocation and upward through consumption, mirroring the tree's concurrency structure.**

### The budget scope model

Each execution node carries a `BudgetScope` containing: a **token ceiling** (maximum input + output tokens this subtree may consume), a **dollar ceiling** (derived from token ceiling × model pricing), a **wall-clock timeout**, and an **iteration ceiling** (for Loop nodes). When a node spawns children, it allocates portions of its budget to each child. The allocation policy is configurable: equal division for Parallel nodes, full budget passed sequentially (with remaining balance) for Sequence nodes, and per-iteration allocation for Loop nodes.

Budget consumption is tracked via an `AtomicU64` counter shared between the execution node and its `BudgetScope`. After every LLM call, the token count is atomically added. A background monitor compares consumption against the ceiling and fires the node's `CancellationToken` when the budget is **90% exhausted** (the remaining 10% is reserved for graceful termination—saving state, writing a partial result, and reporting what was accomplished before the budget ran out).

### Depth-aware model routing saves cost automatically

The LLM Router integrates with budget governance through a simple heuristic: **as recursion depth increases and remaining budget decreases, route to cheaper models.** The top-level Route node (intent classification) uses a lightweight model because routing decisions are structurally simple. The first iteration of a generate→test→fix loop uses the most capable model (highest probability of first-attempt success, which is the cheapest path). Subsequent iterations—which indicate the task is harder than expected—may downgrade to maintain budget headroom, or the system may surface a budget warning to the user and request approval to continue.

This creates a natural cost optimization: easy tasks complete in one iteration with a capable model (moderate cost), while hard tasks get multiple attempts with progressively cheaper models (bounded cost). The user always knows the worst case because the budget ceiling is hard.

### Exit conditions prevent runaway execution

Every Loop node in the execution tree enforces multiple exit conditions simultaneously. The loop terminates when **any** of these conditions becomes true: the quality predicate passes (all tests green, lint clean, review approved), the iteration count hits `max_iterations`, the budget scope signals exhaustion, or the **stagnation detector** fires. The stagnation detector compares the last N iterations' quality scores; if improvement plateaus (the same tests fail, the same lint errors persist), the loop exits early rather than burning budget on approaches that aren't converging. This directly addresses the Google documentation's warning that loop patterns "directly increase latency and operational costs with each cycle."

---

## The Agent Execution Observatory makes the tree visible

The Adaptive Execution Tree's most elegant property is that its structure directly maps to an observability trace. Each execution node is an **OpenTelemetry span**. Leaf nodes produce spans with `gen_ai.agent.name`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and model metadata following the OTel GenAI semantic conventions (v1.37+). Workflow nodes (Sequence, Parallel, Loop, Route, Gate) produce parent spans that contain their children's spans. The result: **the OTel trace IS the execution tree**, viewable in any OTel-compatible backend.

### Real-time execution visualization

The Agent Execution Observatory is Agentique Console's frontend visualization of the live execution tree. It renders as an interactive tree/flame chart in the Tauri webview, updated in real-time via Tauri IPC Channels as agents execute. Each node displays: agent name, current status (running/completed/failed/cancelled), token consumption, estimated cost, elapsed time, and a collapsible detail pane showing the agent's input/output.

The Channel-based architecture is critical for performance. Tauri v2's `Channel<T>` provides **guaranteed message ordering** via an index-based system, with automatic optimization for payload size (direct eval for small payloads, fetch-based IPC for larger ones). This is fundamentally superior to Tauri events for streaming: events are unordered fire-and-forget with JSON-only payloads, while Channels are ordered, typed, and handle binary data efficiently. Each executing agent sends structured progress events through a Channel:

```rust
#[derive(Clone, Serialize)]
enum AgentEvent {
    Started { node_id: NodeId, agent_name: String },
    TokensGenerated { node_id: NodeId, delta: String },  // streaming tokens
    ToolInvoked { node_id: NodeId, tool: String, args: Value },
    Completed { node_id: NodeId, tokens_used: u64, cost_cents: f64 },
    Failed { node_id: NodeId, error: String },
    BudgetWarning { node_id: NodeId, remaining_pct: f32 },
}
```

The frontend receives these events and maintains a reactive tree model that drives the visualization. Users can click any node to inspect its full context (input prompt, output, tool calls), pause execution at any point (triggering a Gate), or cancel a subtree (which propagates cancellation through structured concurrency).

### Cost overlays answer "where did my money go?"

Every span carries cost metadata. The Observatory aggregates this into a cost overlay on the execution tree: each node shows its subtree's total cost, with color coding (green = under budget, yellow = approaching budget, red = budget-constrained). A **cost breakdown view** shows: cost by model tier, cost by agent role, cost by loop iteration (revealing whether the first or fifth attempt was most expensive), and cost by tool category. This granularity lets users identify optimization opportunities—"the security scanner added $0.40 per iteration but never found anything; disable it for this project."

### Deterministic replay from any checkpoint

Every Gate node and every loop iteration boundary creates an implicit **checkpoint**: the full shared state, the execution tree position, and the budget state are serialized. Users can select any checkpoint in the Observatory and **replay execution from that point** with modified parameters—a different model, a revised prompt, an increased budget, or a manually edited code state. This is enabled by the symbolic variable management in shared state: all agent inputs are derived from named state keys, so modifying a state key and re-executing the subtree produces a deterministic (modulo LLM stochasticity) re-run. Deterministic replay transforms debugging from "start over" to "adjust and continue."

---

## MCP as the extensibility spine and the Ripple Engine

The Model Context Protocol is Agentique Console's **primary extension mechanism**. Rather than building a proprietary plugin API, the system treats every external capability—file system access, git operations, build tools, test runners, linters, language servers, CI/CD pipelines, documentation databases—as an MCP server. Agents are MCP clients that discover and invoke tools through the standardized protocol.

### Server lifecycle management

Agentique Console manages MCP servers through a registry that handles discovery, startup, health monitoring, and graceful shutdown. Local MCP servers (bundled with the application or installed by the user) communicate via stdio transport using Tokio's `TokioChildProcess`. Remote MCP servers communicate via Streamable HTTP with SSE for server-pushed events. The November 2025 MCP spec additions are architecturally significant:

**The Tasks primitive** enables long-running operations. When an agent invokes a test suite that takes 30 seconds, the MCP server returns a task handle immediately. The agent (or its parent Loop node) polls for completion or subscribes to progress events. This prevents the agent's context window from blocking on slow tool execution and enables the Parallel primitive to fan out multiple long-running tool invocations concurrently.

**Sampling with Tools** enables compositional agent architectures where MCP servers can themselves perform multi-step LLM reasoning using the client's model. This means a specialized code analysis MCP server can run its own agent loop—inspecting code, forming hypotheses, checking hypotheses against the AST—without the orchestrating agent managing every step. The MCP server becomes an "agent-as-a-tool" that encapsulates both reasoning and tool use behind a clean interface.

### The Ripple Engine: change propagation analysis

The Ripple Engine is a specialized subsystem—not the system's sole core, but a critical capability for multi-file coding tasks. When an agent modifies a file, the Ripple Engine uses the code intelligence subsystem's dependency graph to determine the **blast radius**: which other files import the modified symbols, which tests cover the modified code, and which downstream modules may need updating. This analysis is entirely deterministic (tree-sitter AST + dependency graph traversal, no LLM needed) and completes in milliseconds.

The Ripple Engine feeds into the execution tree by dynamically expanding the task scope. If a code change to `auth.rs` affects `session.rs` and `middleware.rs`, the Ripple Engine injects additional Leaf nodes into the current Sequence to review and potentially update the affected files. This automatic scope expansion prevents the common failure mode where an agent modifies one file in isolation, breaking dependent code that it never examined.

### Extension model for third-party capabilities

Third-party extensions are MCP servers that register with Agentique Console's server registry. The registry supports three discovery mechanisms: **local configuration** (a `.agentique/mcp.json` file in the project root listing server commands and arguments), **well-known URLs** (the MCP spec's `.well-known` discovery protocol for remote servers), and **the MCP Registry** (the open catalog launched in 2025 for discovering public servers). Users can install MCP servers from the registry directly through the Console's UI, similar to installing VS Code extensions but using a standard protocol.

The security model follows Tauri v2's **default-deny capability system**. Each MCP server declares the permissions it requires (file system paths, network access, environment variables). The user explicitly grants capabilities, which are stored in the project's capability configuration. MCP servers cannot access resources beyond their granted scope.

---

## Implementation roadmap in three phases

### Phase 1: Foundation (months 1-4)

Build the execution tree runtime and the core agent loop. Implement the six primitives (Leaf, Sequence, Parallel, Loop, Route, Gate) with structured concurrency via `JoinSet` and `CancellationToken`. Integrate the `rmcp` crate for MCP client functionality with stdio transport. Build a minimal Tauri shell with a conversation interface and basic execution tree visualization using Channels. Implement the working memory tier with observation masking for context compaction. Ship a single-agent coding experience (Leaf nodes only) that can generate code, run tests via MCP, and report results—a functional vertical slice that validates the IPC architecture, the MCP integration, and the Tauri streaming performance.

**Key deliverable**: A working desktop application where a user can describe a coding task, the system generates code, runs tests through an MCP server, and streams results back in real-time with token/cost visibility.

### Phase 2: Multi-agent orchestration (months 5-8)

Enable multi-agent workflows by implementing the full execution tree composition. Build the generate→test→fix Loop with budget governance and stagnation detection. Add the Parallel primitive for concurrent code review (security + style + performance). Implement the Route primitive with a lightweight classifier for intent routing. Build the code intelligence pipeline (tree-sitter indexing + Tantivy search) and integrate it as the context retrieval system for agents. Implement session memory (Tier 2) with typed state keys and the episodic-to-semantic consolidation process for project memory (Tier 3). Expand the Agent Execution Observatory to show the full tree with cost overlays and clickable node inspection.

**Key deliverable**: Multi-agent coding workflows that handle iterative refinement, with visible execution trees, budget enforcement, and cross-session project memory.

### Phase 3: Intelligence and evolution (months 9-12)

Add deterministic replay from checkpoints. Implement the Ripple Engine for change propagation analysis. Build the LLM Router with depth-aware model selection and cascading cost optimization. Add the Gate primitive with full state persistence for human-in-the-loop approval workflows. Implement user memory (Tier 4) for cross-project personalization. Add MCP server lifecycle management with the Tasks primitive for async tool execution. Optimize context compaction with the hopping window mechanism. Build the MCP extension marketplace UI for server discovery and installation.

**Key deliverable**: A full-featured agentic coding system with sophisticated cost optimization, deterministic replay, change propagation awareness, and a rich extension ecosystem.

---

## Conclusion

The Adaptive Execution Tree is not merely an orchestration framework—it is a **structural unification** of three concerns that other systems treat as separate: concurrency management, cost governance, and runtime observability. By making these concerns isomorphic to a single recursive tree, Agentique Console eliminates the impedance mismatches that plague production agent systems. An agent that exceeds its budget is cancelled through the same mechanism that cancels orphaned tasks. An observer viewing the execution trace sees the exact same tree that the scheduler is executing. A checkpoint for deterministic replay captures the exact state that structured concurrency is managing. One data structure. Three interpretations. Zero translation layers.

The six composable primitives—Leaf, Sequence, Parallel, Loop, Route, Gate—are deliberately minimal. They are to multi-agent orchestration what `map`, `filter`, and `reduce` are to data processing: a small, orthogonal set from which arbitrarily complex behaviors emerge through composition. This compositional power means the system does not need to anticipate every possible workflow. When a new coding pattern emerges (say, AI-assisted code migration across language boundaries), it can be expressed as a new composition of existing primitives without modifying the framework.

The most important architectural bet is on **deterministic orchestration by default**. The industry trend toward LLM-driven "everything agents" that reason about every step is elegant in demos but catastrophic in production: it multiplies costs, introduces non-determinism at every control-flow decision, and makes debugging nearly impossible. Agentique Console reserves LLM reasoning for where it genuinely adds value—understanding ambiguous intent, generating creative code, diagnosing novel errors—and relies on Rust code for everything else. The result is a system that is fast, predictable, cost-bounded, and debuggable, while retaining the full power of frontier language models for the tasks that require intelligence.