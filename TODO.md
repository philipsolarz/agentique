# Agentique — Implementation TODO

_Based on `instructions.md` architecture spec vs. current codebase state (branch: `rust`)._
_Last updated: 2026-02-28_

---

## Current State Summary

**Done (P0 + P1 + P2 + P3.1):**
- Agent loop with state machine, channel-based `run_with_channels()`, tool approval flow
- OpenAI + Anthropic streaming providers with retry logic
- Model router with depth-based tiers, cost routing, fallback chain
- Recursive Ripple executor with sub_RLM, budget pools, loop detection, convergence
- REPL + File + Code Search + Find Related tools
- MCP manager (rmcp, stdio, namespacing, 3-tier permissions)
- Data layer (tree-sitter, tantivy, petgraph)
- Session persistence (JSONL), session compressor (85% threshold, LLM summarization)
- Tauri v2 desktop shell, React streaming chat UI, tool approval dialog
- Recursion tree visualizer, cost dashboard, artifact viewer + diff display
- Async sub-agent concurrency (wave scheduling, concurrent dispatch)

**81 tests across workspace (all passing).**

---

## Completed Milestones

### ~~P0 — Core Agent Runtime~~ DONE
### ~~P1 — Integration Layer~~ DONE
### ~~P2 — UI Polish~~ DONE
### ~~P3.1 · Async Sub-Agent Concurrency~~ DONE
### ~~P3.2 · Fix Known Bugs~~ DONE
### ~~P3.3 · Refactor: Split agent_loop.rs~~ DONE
### ~~P3.4 · Refactor: Extract SSE Parser~~ DONE
### ~~P3.5 · Refactor: Session Factory~~ DONE
### ~~P4.1 · Session Resume & Fork~~ DONE
### ~~P4.2 · OpenAI-Compatible Provider~~ DONE
### ~~P4.5 · Settings UI~~ DONE
### ~~P5.1 · Streaming Tool Calls Display~~ DONE

---

## ~~P3 — Bug Fixes & Critical Refactors~~ DONE

These fix real bugs and reduce tech debt before adding more features.

### P3.2 · Fix Known Bugs
**Priority: Immediate — these cause incorrect behavior.**

1. **Deadlock in `send_message`** (`src-tauri/src/lib.rs:274-288`)
   - `state.sessions.lock().await` held while re-locking at line 288
   - Fix: drop first lock before acquiring second, or restructure to avoid double-lock

2. **MCP tool results use Rust debug format** (`mcp-manager/src/manager.rs:175`)
   - `format!("{:?}", c)` sends `Text { text: "..." }` to LLM instead of actual content
   - Fix: pattern-match `Content` variants and extract text properly

3. **`try_lock()` on MCP permissions** (`agent_loop.rs:277`)
   - Silent denial when mutex is contended — tool incorrectly denied without logging
   - Fix: restructure to avoid holding lock during approval, or use async lock

4. **`notify` crate uses macOS-only feature** (`data-layer/Cargo.toml`)
   - `features = ["macos_fsevent"]` — file watching broken on Linux/WSL
   - Fix: use platform-conditional features or default backend

5. **`SessionCompressor` hardcodes `gpt-4o-mini`** (`session.rs`)
   - Breaks for Anthropic-only users — silently degrades to mechanical summary
   - Fix: accept model from caller, use whatever provider is configured

6. **`AgentState` enum defined but never used** (`agent_loop.rs:36-49`)
   - Dead code — remove it

7. **`get_session_cost` always returns 0.0** (`src-tauri/src/lib.rs`)
   - Dead stub command — remove it

### P3.3 · Refactor: Split agent_loop.rs (1143 lines)
**Files:** `crates/agentique-core/src/`

The agent loop file has grown too large with 3 duplicated loop bodies.

- Extract state machine types (`AgentState`, `AgentOp`, `AgentEvent`, etc.) → `types.rs`
- Remove legacy `process()` and `process_streaming()` methods (~500 lines of duplication)
  - All callers should use `run_with_channels()` instead
  - Add thin convenience wrapper if needed for tests
- Extract `ToolCallAccumulator` and streaming accumulation logic → shared helper
- Extract artifact tracking (`capture_old_content_if_file_write`, `maybe_emit_artifact`) into `Tool` trait annotation or middleware
- Extract `compute_cost_microdollars()` to `observability` crate (also used in `executor.rs` with duplicated formula)

### P3.4 · Refactor: Extract SSE Parser
**Files:** `crates/llm-provider/src/`

- SSE buffer/parse logic duplicated between `anthropic.rs` and `openai.rs`
- Extract to `sse.rs` with generic `SseParser<T: DeserializeOwned>` stream adapter
- Both providers call `SseParser::new(byte_stream).events()` → `Stream<Item = T>`

### P3.5 · Refactor: Session Factory
**Files:** `src-tauri/src/lib.rs`

- `create_session` is 160 lines of imperative wiring
- Extract `SessionBuilder` that encapsulates provider selection, tool registration, MCP loading, index creation
- Tauri `create_session` becomes: `SessionBuilder::new(model, api_key).build()?`
- Moves provider detection heuristic out of Tauri layer

---

## P4 — Features: Session Management & Provider Expansion

### P4.1 · Session Resume & Fork
**Files:** `crates/agentique-core/src/session.rs`, `src-tauri/src/lib.rs`, `src/components/SessionSidebar.tsx`

- **Resume session**: load persisted JSONL, replay conversation history into agent loop
- **Fork session**: clone current session state into a new session, diverge from a point
- Backend: `resume_session` and `fork_session` Tauri commands
- Frontend: resume/fork buttons in session sidebar
- List persisted sessions (from disk) alongside active sessions

### P4.2 · OpenAI-Compatible Provider (Ollama/vLLM/LM Studio)
**Files:** `crates/llm-provider/src/openai_compat.rs` (new)

- Adapter for OpenAI-compatible local endpoints
- Custom base URL, configurable capabilities (context window, pricing)
- Model list auto-discovery from `/v1/models` endpoint
- Wire into Tauri: user can set base URL + model in settings

### P4.3 · Google Gemini Provider
**Files:** `crates/llm-provider/src/google.rs` (new), `crates/llm-provider/src/lib.rs`

- Implement `CompletionProvider` for Gemini API
- SSE streaming, Gemini-specific tool calling format
- Register in model router with capabilities and pricing

### P4.4 · Streamable HTTP MCP Transport
**Files:** `crates/mcp-manager/src/manager.rs`, `crates/mcp-manager/src/config.rs`

- HTTP transport support for remote MCP servers (currently stdio only)
- SSE-based streaming for server-to-client notifications
- Auth header support for remote MCP endpoints
- Reconnection logic with exponential backoff

### P4.5 · Settings UI
**Files:** `src/components/Settings.tsx` (new), `src-tauri/src/lib.rs`

- Settings panel: API keys, default model, budget limits, MCP server configuration
- Persist settings to `~/.agentique/config.toml`
- Provider capabilities display (show pricing, context window for selected model)
- Base URL configuration for OpenAI-compatible providers

---

## P5 — Features: Agent Intelligence & UX

### P5.1 · Streaming Tool Calls Display
**Files:** `src/components/ChatPanel.tsx`

- Show tool name + arguments as they stream in (currently only shown after completion)
- Animated "thinking" indicator during LLM calls
- Collapsible tool call/result sections in chat history

### P5.2 · System Prompt Customization
**Files:** `crates/agentique-core/src/agent_loop.rs`, `src-tauri/src/lib.rs`, UI

- User-editable system prompt per session
- Default system prompt templates (coding assistant, research, general)
- Persist custom prompts with session

### P5.3 · File Watcher & Auto-Index
**Files:** `crates/data-layer/src/`, `crates/agentique-core/src/`

- Watch project directory for file changes (fix `notify` crate config first)
- Incremental re-index on file save (tree-sitter re-parse, tantivy update)
- Notify agent of external file changes during session

### P5.4 · Conversation Export
**Files:** `crates/agentique-core/src/session.rs`, UI

- Export conversation as Markdown
- Export as JSON (for reimport/sharing)
- Copy individual messages to clipboard

### P5.5 · Multi-Model Conversations
**Files:** `crates/llm-provider/src/router.rs`, UI

- Switch models mid-conversation (e.g., use cheap model for exploration, expensive for final output)
- Model selector dropdown in chat UI
- Per-message model indicator in chat history

---

## Immediate Next Steps

Ordered by impact and unblocking subsequent work:

1. ~~**P3.2** — Fix bugs~~ DONE
2. ~~**P3.3** — Split agent_loop.rs~~ DONE
3. ~~**P3.4** — Extract SSE parser~~ DONE
4. ~~**P3.5** — Session factory~~ DONE
5. ~~**P4.1** — Session resume & fork~~ DONE
6. ~~**P4.2** — OpenAI-compatible provider (local LLMs)~~ DONE
7. ~~**P4.5** — Settings UI (needed to configure providers/keys)~~ DONE
