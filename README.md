# Agentique

A modular Python agent framework. The framework is split into a dependency-free
**core** of contracts and an engine, plus satellite packages that provide
concrete implementations — so you install only what you use, and new providers
or tools are just new sibling packages.

All packages share the `agentique.*` import namespace (PEP 420).

## Packages

| Distribution | Import | What it provides | Depends on |
|---|---|---|---|
| `agentique-core` | `agentique.core` | Seam Protocols (`Model`, `Tool`, `Skill`, `Memory`), value types, `Result` union, and the `Runtime` engine | — (zero deps) |
| `agentique-anthropic` | `agentique.anthropic` | `AnthropicModel` over the Anthropic Messages API | core, `anthropic` |
| `agentique-skills` | `agentique.skills` | Pure `Skill`s (e.g. `ExtractText`) | core |
| `agentique-tools` | `agentique.tools` | Real `Tool`s: `ReadFile`, `Delegate` (multi-agent) | core |
| `agentique-memory` | `agentique.memory` | Durable `Memory`: `InMemoryStore`, `FileStore` | core |
| `agentique-testing` | `agentique.testing` | `StubModel` for offline, deterministic tests | core |

The application layer (`agentique-console`, *later*) will be built on top of
these, adding orchestration, lifecycle, and gates using core primitives.

## Layout

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/);
the installable packages live under `packages/`, each with its own `README.md`.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync       # create the env and install all workspace packages + dev tools
make check    # ruff (lint + format), ty (types), pytest — the full gate
```

Individual gates: `make lint`, `make type`, `make test`.
