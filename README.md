# Agentique

A modular Python agent framework. It is a **single distribution** (`agentique`)
built around a dependency-free **core** of contracts and an engine, plus
submodules that provide concrete implementations. Everything imports under the
`agentique.*` namespace.

`agentique` pulls **zero third-party dependencies**. The only submodule that needs
one — the Anthropic provider — keeps its SDK behind an optional extra, so you opt
into it explicitly:

```sh
pip install agentique               # core + skills + tools + memory + testing; no third-party deps
pip install "agentique[anthropic]"  # adds the official `anthropic` SDK for agentique.anthropic
```

## Submodules

| Import | What it provides | Third-party deps |
|---|---|---|
| `agentique.core` | Seam Protocols (`Model`, `Tool`, `Skill`, `Memory`), value types, the `Result` union, and the `Runtime` engine | — |
| `agentique.skills` | Pure `Skill`s (e.g. `ExtractText`) | — |
| `agentique.tools` | Real `Tool`s: `ReadFile`, `Delegate` (multi-agent) | — |
| `agentique.memory` | Durable `Memory`: `InMemoryStore`, `FileStore` | — |
| `agentique.testing` | Offline test doubles: `StubModel`, `EchoTool` | — |
| `agentique.anthropic` | `AnthropicModel` over the Anthropic Messages API | `anthropic` (via the `anthropic` extra) |

The application layer (`agentique-console`, *later*) will be built on top of
these, adding orchestration, lifecycle, and gates using core primitives.

## Public surface

### `agentique.core`

The application-agnostic contracts and engine. **Zero third-party dependencies.**

- **Seams (Protocols):** `Model`, `Tool`, `Skill`, `Memory` — structural
  interfaces; implement by shape, no inheritance.
- **Value types (frozen):** `Message`, `TextBlock`, `ToolUseBlock`,
  `ToolResultBlock`, `ModelResponse`, `Context`, `Agent`, `Permissions`,
  `ToolSpec`, `ToolResult`.
- **Result union:** `Result = Completed | NeedsHuman | Blocked` — branch with an
  exhaustive `match`.
- **Engine:** `Runtime` — drives one declarative `Agent` to a `Result` (assemble
  context → call model → dispatch permitted tools → fold results → finish).
  Run-control (`max_turns`) lives here, not on the `Agent`.

### `agentique.skills`

Pure, deterministic `Skill`s — each unit-testable in isolation, touching neither
the world nor the model.

- **`ExtractText(separator="")`** — a `Skill[Message, str]` that joins a message's
  text blocks, ignoring tool-use/tool-result blocks.

### `agentique.tools`

Real `Tool`s — actions that cross an external boundary and are therefore
permission-gated by the `Runtime`.

- **`ReadFile()`** — read the UTF-8 text contents of a file (read-only; failures
  returned as error results, not raised).
- **`Delegate(child, *, name, description, runtime=None)`** — expose a child
  `Agent` as a tool the parent may invoke. Pure mechanism for multi-agent
  topologies: it carries no orchestration logic, and is gated by the parent's
  permissions like any other tool.

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

## Layout

Standard src-layout single package:

```
pyproject.toml
src/agentique/
  core/  skills/  tools/  memory/  testing/  anthropic/
tests/
  core/  skills/  tools/  memory/  testing/  anthropic/  test_import_discipline.py
```

The zero-third-party boundary is enforced structurally:
`tests/test_import_discipline.py` asserts that no module under `src/agentique`
**except** `agentique.anthropic` imports anything outside the standard library and
`agentique` itself.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync       # create the env and install agentique (with the anthropic extra) + dev tools
make check    # ruff (lint + format), ty (types), pytest — the full gate
```

Individual gates: `make lint`, `make type`, `make test`.
