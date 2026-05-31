# Agentique

A modular Python agent framework. It is a **single distribution** (`agentique`)
built around a **core** execution + topology substrate, plus submodules that
provide concrete implementations. Everything imports under the `agentique.*`
namespace.

Its only base third-party dependency is **Pydantic** (the neutral IR, typed I/O,
durable pause/resume, and the event vocabulary are all Pydantic-backed). Provider
SDKs and telemetry exporters stay behind optional extras, so a base install never
pulls a model vendor's SDK or a telemetry stack:

```sh
pip install agentique               # core + tools + memory + testing (+ pydantic)
pip install "agentique[anthropic]"  # adds the official `anthropic` SDK for agentique.anthropic
```

## Submodules

| Import | What it provides | Extra deps |
|---|---|---|
| `agentique.core` | Seam Protocols (`Model`, `Tool`, `Memory`), the neutral-IR value types, the `Result` union, the middleware onion, typed-I/O validation, the `Engine` (drives one agent) and the `Scheduler` (coordinates many) | — |
| `agentique.tools` | Real `Tool`s: `ReadFile`, `ListDir`, `WriteFile`, `EditFile`, `RunCommand` (confined to a `Workspace`), `Delegate` (multi-agent), `AskHuman` (pause for input) | — |
| `agentique.memory` | Durable `Memory`: `InMemoryStore`, `FileStore` | — |
| `agentique.testing` | Offline doubles: `StubModel`, `EchoTool`, `CollectingSink` | — |
| `agentique.anthropic` | `AnthropicModel` over the Anthropic Messages API | `anthropic` (extra) |

Two further layers express the structure the framework is built around —
`console → harness → core` (the application talks to a human, the harness
coordinates work-products, the core is the execution substrate). The **harness is
the `agentique` root itself**:

| Import | What it provides | Extra deps |
|---|---|---|
| `agentique` (root) | Generic, domain-agnostic harness: `Coordinator` (thin operator layer over the Scheduler), `Role`, `Session`, typed `Artifact` (+ provenance DAG), shared `Store` — `from agentique import …` | — |
| `agentique.console` | The human-facing application: `Console` + the `agentique` CLI REPL | — |

The harness carries **no** domain specifics; what the artifacts it coordinates
*mean* is the application's concern. `agentique.console` is the only layer that
talks to a human.

## Public surface

### `agentique.core`

The execution + topology substrate.

- **Seams (Protocols):** `Model`, `Tool`, `Memory` — structural interfaces;
  implement by shape, no inheritance. `Tool.__call__(ctx, arguments)` receives a
  `RunContext` (run id, event-emit handle, and a `dispatch` handle when a Scheduler
  drives it).
- **Neutral IR (frozen, Pydantic):** `Message`, `TextBlock`, `ToolUseBlock`,
  `ToolResultBlock`, `OpaqueBlock` (carries vendor blocks the core does not model —
  thinking/citations/etc. — through verbatim), `ModelResponse`, neutral
  `StopReason` (`kind` + raw vendor string), `Usage`, `Context`, `Agent`,
  `Permissions`, `ToolSpec`, `ToolResult`. Provider adapters live in the provider
  package; the IR is vendor-neutral and round-trippable.
- **Result union:** `Result = Completed | NeedsHuman | Blocked`. A run pauses as
  `NeedsHuman` carrying a resumable `Paused` snapshot; an unhandled stop reason
  surfaces as `Blocked` rather than folding into a quiet success.
- **Engine:** `Engine` — drives one declarative `Agent` to a `Result`. `max_turns`
  lives here. Every step runs through a **middleware onion** (`Middleware`) at a
  fixed set of points (turn / pre-model / model-call / tool-call); the default
  chain is empty (the trivial path is unchanged). Built-ins: `TracingMiddleware`
  (emits the event vocabulary), `PermissionMiddleware`, `CompactionMiddleware`.
- **Scheduler:** `Scheduler` — registers agents by id and dispatches runs between
  them (`register` / `dispatch` / `resume`), synchronously and deterministically;
  tracks a lightweight `Run` (state + `Paused` snapshot + `parent_id` lineage) and
  owns the single event stream.
- **Permissions:** ordered `Rule`s resolved `deny > ask > allow`. A `deny` ends the
  run; an `ask` joins the human-pause spine (`NeedsHuman`).
- **Typed I/O:** an `Agent.output_type` and a `ToolSpec.args_model` (Pydantic
  models) are validated by the Engine; a mismatch becomes a self-correctable error.
- **Events:** a typed `Event` vocabulary + the `EventSink` seam (`NullSink`
  default). Concrete exporters (e.g. OpenTelemetry) belong in a satellite extra.
- **Compaction:** the `Compactor` seam + `EvictOldestToolResults` rung at the
  pre-model point.

### `agentique.tools`

Real `Tool`s — actions that cross an external boundary, permission-gated by the
Engine.

- **`ReadFile()`**, **`ListDir(workspace)`**, **`WriteFile(workspace)`**,
  **`EditFile(workspace)`**, **`RunCommand(workspace, …)`** — file/command tools
  confined to a **`Workspace(root)`** (rejects absolute paths and `..` escapes).
- **`Delegate(child_id, *, name, description)`** — expose a Scheduler-registered
  child agent as a tool; it dispatches via `ctx.dispatch`, so the child runs as its
  own (lineage-tracked) run and its real `Result` flows back — a child pause becomes
  a real pause of the parent. Pure mechanism, permission-gated like any tool.
- **`AskHuman()`** — pause the run for human input (raises `PauseRequested`, which
  the Engine turns into `NeedsHuman`). Call it alone in a turn.

### `agentique.memory`

- **`InMemoryStore()`** — a process-lifetime dict; ideal for tests.
- **`FileStore(path)`** — durable JSON-file storage, written atomically off the
  event loop.

### `agentique.testing`

- **`StubModel(responses)`** — replays scripted `ModelResponse`s and records calls;
  factories `StubModel.text(...)` / `StubModel.tool_call(...)`.
- **`EchoTool(...)`** — a `Tool` that echoes its `value` argument and records calls.
- **`CollectingSink()`** — an `EventSink` that collects emitted events in order.

### `agentique.anthropic`

The Anthropic provider (requires the `anthropic` extra). `AnthropicModel(model, *,
max_tokens=4096, client=None)` satisfies `agentique.core.Model` and is the only
boundary that maps the neutral IR to/from a vendor SDK. The model id has **no
default** — supply a current one.

### `agentique` (harness root)

- **`Artifact`** — a durable unit of work with a **typed Pydantic `payload`**,
  a provenance DAG (`derived_from`), and an application-defined `status` lifecycle.
  `TextPayload` is the default carrier; `payload_text(payload)` renders a preview.
- **`Coordinator`** — the operator layer over the `Scheduler`: `dispatch` /
  `dispatch_role` launch runs and project the outcome into a proposed `Artifact`;
  `resume(session_id, …)` continues a paused run; `approve` / `reject` /
  `promote_artifact` drive the lifecycle. It holds **no** run-state copy.
- **`Store`** — persists artifacts and session records over the `Memory` seam, and
  persists paused runs (`load_paused_runs` re-pairs them after a restart). It takes
  an app-supplied `payload_models` (kind→model) registry to rehydrate typed
  payloads; unregistered kinds fall back to `TextPayload`.
- **`Role`**, **`Session`** — a named specialist (agent + artifact kind) and the
  harness's projection of one tracked run.

### `agentique.console`

The application layer — a conversational console you can run.

- **`build_console(model, *, fleet_model=None, workspace_root="workspace")`** —
  wires a conversational orchestrator over a `Coordinator` and a fleet
  (planner/explorer/builder/reviewer), each a `Role` confined to a shared
  `Workspace`. The orchestrator holds a `Dispatch` tool (`coordinator.dispatch_role`)
  that lands a specialist's output as a *proposed* artifact; `approve` / `reject`
  are the operator's promotions.
- **CLI:** `agentique` (or `uv run agentique`) — a REPL over `Console`; reads
  `ANTHROPIC_API_KEY` from the environment or a local `.env`; supports
  `/artifacts`, `/approve <id>`, `/reject <id>`, `/quit`.

## Layout

```
pyproject.toml
src/agentique/
  __init__.py                                    # harness public surface (Coordinator, …)
  coordinator.py  session.py  artifact.py        # harness (generic), at the root
  store.py  role.py
  core/  tools/  memory/  testing/  anthropic/   # framework + satellites
  console/                                       # application (CLI)
tests/
```

Two boundaries are enforced structurally by `tests/test_import_discipline.py`.
The **dependency boundary**: no module under `src/agentique` **except**
`agentique.anthropic` imports a third-party root beyond the standard library,
`agentique` itself, and the approved base deps (`pydantic`). The **three-layer**
boundary: the dependency arrow runs one way, `console → harness → core`.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync       # create the env and install agentique (+ the anthropic extra) + dev tools
make check    # ruff (lint + format), ty (types), pytest — the full gate
```

Individual gates: `make lint`, `make type`, `make test`. The dev-only
`observability/` and `scenarios/` packages (run-capture wrappers and dev REPLs)
live at the repo root, are never shipped, and are never imported by `src/`.
`make console` runs the instrumented dev REPL (writes `runs/<ts>-console/`).

## Headroom (extension points, not built this round)

The architecture is intentionally modular: each subsystem is a seam plus the one or
two rungs in use, with room to grow without a rewrite. Reachable additively:
concurrency / pub-sub / mailboxes on the Scheduler bus; durable step-checkpointing
beyond the persist-paused-runs rung; an OpenTelemetry exporter `EventSink` (in a
satellite extra); model-summarization `Compactor`s; a permission rule DSL; a generic
`RunContext[Deps]`; event-stream-based observability for delegation.
