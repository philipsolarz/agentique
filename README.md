# Agentique

A modular Python agent framework. It is a **single distribution** (`agentique`)
built around a dependency-free **core** of contracts and an engine, plus
submodules that provide concrete implementations. Everything imports under the
`agentique.*` namespace.

`agentique` pulls **zero third-party dependencies**. The only submodule that needs
one — the Anthropic provider — keeps its SDK behind an optional extra, so you opt
into it explicitly:

```sh
pip install agentique               # core + tools + memory + testing; no third-party deps
pip install "agentique[anthropic]"  # adds the official `anthropic` SDK for agentique.anthropic
```

## Submodules

| Import | What it provides | Third-party deps |
|---|---|---|
| `agentique.core` | Seam Protocols (`Model`, `Tool`, `Memory`), value types, the `Result` union, and the `Runtime` engine | — |
| `agentique.tools` | Real `Tool`s: `ReadFile`, `Delegate` (multi-agent), `AskHuman` (pause for input) | — |
| `agentique.memory` | Durable `Memory`: `InMemoryStore`, `FileStore` | — |
| `agentique.testing` | Offline test doubles: `StubModel`, `EchoTool` | — |
| `agentique.anthropic` | `AnthropicModel` over the Anthropic Messages API | `anthropic` (via the `anthropic` extra) |

Two further subpackages express the layering the framework is built around —
`console → code → core` (the application talks to a human, the harness coordinates
agents, the core is the generic framework):

| Import | What it provides | Third-party deps |
|---|---|---|
| `agentique.code` | Generic, domain-agnostic harness: `Coordinator`, `Session`, `Artifact`, shared `Store` | — |
| `agentique.console` | The human-facing application: `Console` + the `agentique` CLI REPL | — |

`agentique.code` is deliberately generic and carries **no** domain specifics (no
repos, git, diffs, or PRs); what the artifacts it coordinates *mean* is the
application's concern. `agentique.console` is the only layer that talks to a human.

## Public surface

### `agentique.core`

The application-agnostic contracts and engine. **Zero third-party dependencies.**

- **Seams (Protocols):** `Model`, `Tool`, `Memory` — structural interfaces;
  implement by shape, no inheritance.
- **Value types (frozen):** `Message`, `TextBlock`, `ToolUseBlock`,
  `ToolResultBlock`, `ModelResponse`, `Context`, `Agent`, `Permissions`,
  `ToolSpec`, `ToolResult`.
- **Result union:** `Result = Completed | NeedsHuman | Blocked` — branch with an
  exhaustive `match`. A run pauses as `NeedsHuman` carrying a resumable `Paused`
  snapshot; `Runtime.resume(agent, paused, answer)` continues that exact run.
- **Engine:** `Runtime` — drives one declarative `Agent` to a `Result` (assemble
  context → call model → dispatch permitted tools → fold results → finish).
  Run-control (`max_turns`) lives here, not on the `Agent`.

### `agentique.tools`

Real `Tool`s — actions that cross an external boundary and are therefore
permission-gated by the `Runtime`.

- **`ReadFile()`** — read the UTF-8 text contents of a file (read-only; failures
  returned as error results, not raised).
- **`Delegate(child, *, name, description, runtime=None)`** — expose a child
  `Agent` as a tool the parent may invoke. Pure mechanism for multi-agent
  topologies: it carries no orchestration logic, and is gated by the parent's
  permissions like any other tool.
- **`AskHuman()`** — pause the run for human input. Raises the `PauseRequested`
  control signal, which the `Runtime` turns into a `NeedsHuman` outcome carrying a
  resumable snapshot. Call it alone in a turn.

### `agentique.memory`

Durable, cross-run `Memory` implementations.

- **`InMemoryStore()`** — a process-lifetime dict; ideal for tests.
- **`FileStore(path)`** — durable storage backed by a single JSON file, written
  atomically; file I/O runs off the event loop via `asyncio.to_thread`.

### `agentique.testing`

Deterministic, offline doubles for exercising the agent loop without a network or
API key.

- **`StubModel(responses)`** — replays a scripted sequence of `ModelResponse`s,
  one per `complete` call, and records every call as a `StubCall` for assertions.
  Raises `StubModelExhausted` if called more times than scripted.
- **`EchoTool(name="echo", *, is_error=False)`** — a `Tool` that echoes its
  `value` argument back as the result and records every call.

### `agentique.anthropic`

The Anthropic provider — requires the `anthropic` extra.

```python
from agentique.anthropic import AnthropicModel

model = AnthropicModel("<current-model-id>")  # id is environment-specific
```

`AnthropicModel(model, *, max_tokens=4096, client=None)` satisfies
`agentique.core.Model`. The model id has **no default** — supply a current one
(confirm it in the Anthropic console; do not hardcode a guess). Pass a custom
`AsyncAnthropic` client to control auth, base URL, or retries.

### `agentique.console`

The application layer — a conversational console you can run.

- **`Console`** / **`build_console(model)`** — wires a conversational orchestrator
  agent (holding `ask_human` and a `plan_file` tool) over a `Coordinator`. `send()`
  threads the conversation across turns via the pause/resume spine; `approve` /
  `reject` drive an artifact's lifecycle.
- **CLI:** `agentique` (or `uv run agentique`) — a REPL over `Console`. It reads
  `ANTHROPIC_API_KEY` from the environment or a local `.env` (parsed with the
  standard library, no third-party loader), and supports `/artifacts`,
  `/approve <id>`, `/reject <id>`, `/quit`.

## Layout

Standard src-layout single package:

```
pyproject.toml
src/agentique/
  core/  tools/  memory/  testing/  anthropic/   # framework
  code/                                          # harness (generic)
  console/                                       # application (CLI)
tests/
  core/  tools/  memory/  testing/  anthropic/  test_import_discipline.py
```

Two boundaries are enforced structurally by `tests/test_import_discipline.py`.
The **zero-third-party** boundary: no module under `src/agentique` **except**
`agentique.anthropic` imports anything outside the standard library and
`agentique` itself. The **three-layer** boundary: the dependency arrow runs one
way, `console → code → core`, so `core` must not import `code`/`console` and
`code` must not import `console`.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync       # create the env and install agentique (with the anthropic extra) + dev tools
make check    # ruff (lint + format), ty (types), pytest — the full gate
```

Individual gates: `make lint`, `make type`, `make test`.

### Observing the console while testing

The shipped `agentique` CLI stays free of the dev-only `observability` layer. To
talk to the console *with full run capture* while testing, use the observed dev
REPL:

```sh
make console   # same Console, instrumented; writes runs/<ts>-console/
```

It wraps the orchestrator and the planner sub-agent with one shared recorder (by
composition — no shipped-code dependency on `observability`) and writes a
per-session trace — `events.jsonl`, `manifest.json`, and an anomaly-forward
`digest.md` — refreshed after each turn and printed on exit.
