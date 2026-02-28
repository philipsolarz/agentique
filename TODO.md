# Agentique — Implementation TODO

_Based on `instructions.md` architecture spec vs. current codebase state (branch: `rust`)._
_Last updated: 2026-02-27_

---

## Current State Summary

**Done (P0 + P1 + P2):**
- Agent loop with explicit state machine: `AgentState`/`AgentOp`/`AgentEvent` enums, channel-based `run_with_channels()`, backward-compat `process_streaming()` wrapper (`agentique-core`)
- Tool approval flow: `should_auto_approve()` (read-only auto, destructive needs approval), `session_approved_tools` HashSet, `ToolApprovalRequired` event, `ExecApproval` op (`agentique-core`)
- OpenAI streaming provider with retry logic (`llm-provider`)
- Anthropic provider adapter: SSE streaming, `tool_use` content blocks, correct pricing (`llm-provider/anthropic.rs`)
- Model router: depth-based tiers, cost routing, fallback chain (`llm-provider/router.rs`)
- REPL session with symbolic variable handling (`ripple-engine/repl.rs`)
- Recursive Ripple executor: `sub_RLM` tool, budget pools, loop detection, convergence detection (`ripple-engine/executor.rs`, `recursion.rs`)
- REPL tools: set, get, load_file, slice, search, chunks, len
- File tools: read, write, list
- Cost tracking / budget enforcement (`observability`)
- Session persistence to JSONL (`agentique-core/session.rs`)
- Tauri v2 desktop shell with multi-session management + channel-based agent
- React streaming chat UI with Markdown rendering (react-markdown + remark-gfm + react-syntax-highlighter)
- Tool approval dialog in frontend (Allow once / Allow for session / Deny)
- MCP manager: rmcp 0.16, stdio transport, namespacing, 3-tier permissions (`mcp-manager`)
- MCP wired into ToolRouter: `McpToolAdapter`, `register_mcp_tools()`, `is_mcp_tool()`, MCP permissions in agent loop (`agentique-core`)
- Data layer: tree-sitter AST extraction (Rust/Python/TypeScript/JavaScript), tantivy full-text search, petgraph dependency graph (`data-layer`)
- Data layer wired into agent tools: `CodeSearchTool` (full-text + symbol search), `FindRelatedTool` (dependency graph traversal), registered in ToolRouter (`agentique-core/tools/`)
- Session compressor: 85% threshold, keeps last 4 exchanges, LLM-powered summarization, `Compaction` event logging (`agentique-core/session.rs`)
- Recursion tree visualizer: `RecursionTree.tsx` with collapsible step nodes, color-coded by state
- Cost dashboard: progress bar (spend vs budget), expandable per-step breakdown table, warning colors at 80%/100%
- Artifact viewer + diff display: `ArtifactViewer.tsx` with line-based diff, syntax highlighting, collapse/expand, edit vs new badges, `ArtifactCreated` backend event with before/after content capture on `file_write`
- Session sidebar improvements: session name from first user message, creation timestamp, delete session support, improved visual hierarchy
- Async sub-agent concurrency: wave-based dependency scheduling, concurrent dispatch via `futures::future::join_all`, `Arc<Mutex<LoopDetector>>` for thread-safe loop detection, `SubRlmCall`/`ExecutionWave` types, configurable `max_concurrent` (default 8)

**80 tests across workspace (all passing).**

---

## Completed Milestones

### ~~P0 — Core Agent Runtime~~ DONE

- ~~P0.1 · Anthropic Provider Adapter~~ — SSE streaming, tool_use blocks, pricing
- ~~P0.2 · Model Router~~ — Depth/cost/fallback policies, `RouterProvider` wrapper
- ~~P0.3 · Recursive Ripple Executor~~ — Full REPL loop, sub_RLM, budget pools, loop detection, convergence
- ~~P0.4 · Formal Agent State Machine~~ — `AgentState`/`AgentOp`/`AgentEvent`, `run_with_channels()`, tool approval flow

### ~~P1 — Integration Layer~~ DONE

- ~~P1.1 · MCP Manager~~ — rmcp 0.16, stdio transport, namespacing, 3-tier permissions
- ~~P1.2 · Wire MCP into ToolRouter~~ — `McpToolAdapter`, `register_mcp_tools()`, `is_mcp_tool()`, MCP permissions in agent loop
- ~~P1.3 · Data Layer~~ — Tree-sitter (Rust/Python/TypeScript/JavaScript), tantivy full-text search, petgraph dependency graph (12 tests)
- ~~P1.4 · Session Compressor~~ — 85% threshold, keeps system + last 4 exchanges, LLM summarization, `Compaction` event
- ~~P1.5 · Wire Data-Layer into Agent Tools~~ — `CodeSearchTool`, `FindRelatedTool` registered in ToolRouter, indexed on session startup

### ~~P2 — UI Polish & Observability~~ DONE

- ~~P2.1 · Markdown + Syntax Highlighting~~ — react-markdown, remark-gfm, react-syntax-highlighter
- ~~P2.2 · Tool Approval Dialog~~ — Allow once / Allow for session / Deny
- ~~P2.3 · Recursion Tree Visualizer~~ — Collapsible step nodes, color-coded by state, model/token/cost display
- ~~P2.4 · Artifact Viewer + Diff Display~~ — `ArtifactViewer.tsx`, line diff, syntax highlighting, ArtifactCreated events
- ~~P2.5 · Cost Dashboard~~ — Progress bar, per-step breakdown, warning colors at 80%/100%
- ~~P2.6 · Session Sidebar Improvements~~ — Session names, timestamps, delete, improved layout

---

## P3 — Beta Milestone (Async Recursion + Full Observability)

### ~~P3.1 · Async Sub-Agent Concurrency~~ DONE
**Files:** `crates/ripple-engine/src/recursion.rs`, `crates/ripple-engine/src/executor.rs`

- ~~When `sub_RLM()` appears multiple times in same iteration, dispatch concurrently via `futures::future::join_all()`~~
- ~~Dependency DAG analysis: `schedule_sub_agents()` groups independent calls into concurrent waves, serializes dependent calls~~
- ~~Budget pool atomic CAS wired for concurrent reservation~~
- ~~`SubRlmCall` / `ExecutionWave` types, configurable `max_concurrent` (default 8), chunked dispatch~~
- ~~`LoopDetector` wrapped in `Arc<Mutex<>>` for thread-safe concurrent access~~
- ~~8 new tests: wave scheduling (empty, single, independent, chain, 3-level chain, mixed deps, fan-out/fan-in), concurrent budget pool reservation (20-thread stress test)~~

### P3.2 · Full Tracing Spans
**Files:** `crates/observability/src/tracer.rs`, `crates/agentique-core/src/agent_loop.rs`

- Replace the basic `init_tracing()` with `StepTrace` emission at every agent step
- Spans: `agent.turn` → `ripple.loop_iteration` → `llm.completion` | `tool.execution` | `ripple.sub_agent`
- Forward `StepTrace` events to frontend via Tauri event channel
- Export spans to local session JSONL + optional OTLP endpoint
- Structured span data: step_id, parent_step_id, depth, model, token counts, cost, variables changed

### P3.3 · Provenance Citations
**Files:** `crates/observability/src/provenance.rs` (new), `crates/agentique-core/src/agent_loop.rs`

- `ProvenanceRecord { content_hash, step_id, source_refs, model_id, prompt_hash }`
- `ArtifactStore`: immutable content-addressed storage (SHA-256 keyed)
- Attach provenance to every REPL `$FINAL` output and every file write
- `trace_lineage(variable_name) -> Vec<ProvenanceRecord>`
- Surface provenance info in the ArtifactViewer component

### P3.4 · wasmtime Sandbox for Code Execution
**Files:** `crates/ripple-engine/src/executor.rs`, `crates/ripple-engine/Cargo.toml`

- Add `wasmtime` + `wasmtime-wasi` to ripple-engine Cargo.toml
- Sandboxed execution for model-generated code (currently code is expressed as tool calls)
- Configure: epoch-based interruption + fuel metering
- WASI: allow REPL variable reads/writes but no network, no filesystem outside project root
- Fuel limit: configurable, default 10M instructions per code block
- Add adversarial tests: infinite loop (fuel exhaustion), memory bomb, path traversal attempts

### P3.5 · Security Hardening
**Files:** `src-tauri/tauri.conf.json`, `src-tauri/capabilities/`, multiple crate files

- Tauri Isolation Pattern: configure `isolation` in `tauri.conf.json`
- CSP headers restricting WebView script sources
- MCP tool descriptions sanitized before inclusion in LLM prompt (strip injection attempts)
- Path traversal validation on all file tool inputs (reject `..` components, symlink checking)
- `cargo-vet` setup for supply chain audit
- Secrets: migrate to keyring-only via `keyring` crate (no plaintext env vars for API keys)
- SQLCipher encryption for data-layer storage (add `storage.rs` to data-layer crate)

---

## P4 — Production Milestone

### P4.1 · Session Resume & Fork
**Files:** `crates/agentique-core/src/session.rs`, `src-tauri/src/lib.rs`, `src/components/SessionSidebar.tsx`

- "Resume session" — load persisted JSONL, replay conversation history into agent loop
- "Fork session" — clone current session state into a new session, diverge from a point
- Backend: `resume_session` and `fork_session` Tauri commands
- Frontend: resume/fork buttons in session sidebar
- List persisted sessions (from disk) alongside active sessions

### P4.2 · Streamable HTTP MCP Transport
**Files:** `crates/mcp-manager/src/manager.rs`, `crates/mcp-manager/src/config.rs`

- Add HTTP transport support for remote MCP servers (currently stdio only)
- SSE-based streaming for server-to-client notifications
- Auth header support for remote MCP endpoints
- Reconnection logic with exponential backoff

### P4.3 · Google Gemini Provider
**Files:** `crates/llm-provider/src/google.rs` (new), `crates/llm-provider/src/lib.rs`

- Implement `CompletionProvider` for Google Gemini API
- SSE streaming, tool calling format differences, structured output
- Register in model router with capabilities and pricing

### P4.4 · OpenAI-Compatible Provider (Ollama/vLLM)
**Files:** `crates/llm-provider/src/openai_compat.rs` (new)

- Adapter for OpenAI-compatible endpoints (Ollama, vLLM, LM Studio)
- Custom base URL, configurable capabilities
- Auto-detection of supported features

### P4.6 · Plugin Architecture
**Files:** `src-tauri/`, plugin system design

- Tauri v2 plugin surfaces for user-installable extensions
- Custom MCP server configurations as plugins
- Model provider credential management per-plugin
- UI theme plugins with scoped permissions

### P4.7 · Auto-Updater & Cross-Platform Packaging
**Files:** `src-tauri/tauri.conf.json`

- Tauri updater plugin for auto-updates
- Package for Windows (.msi), macOS (.dmg), Linux (.AppImage/.deb)
- Code signing for distribution
- CI/CD pipeline for cross-platform builds

### P4.8 · Performance Optimization
- P50/P95/P99 latency benchmarks per Ripple iteration
- CI regression gates: P95 cost must not exceed 10x P50 for benchmarks
- Incremental tree-sitter parsing (re-parse only changed files)
- Connection pooling for LLM provider HTTP clients
- Streaming response buffer tuning

---

## Immediate Next Steps

Ordered by impact and unblocking subsequent work:

1. **P3.2** — Full tracing spans (OTLP export, structured step hierarchy)
2. **P3.5** — Security hardening (path traversal, CSP, Tauri isolation)
3. **P4.1** — Session resume & fork (critical UX for long workflows)
4. **P3.3** — Provenance citations (audit trail for agent outputs)
5. **P3.4** — wasmtime sandbox for code execution
