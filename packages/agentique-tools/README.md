# agentique-tools

Real `Tool` implementations for Agentique — actions that cross an external
boundary and are therefore permission-gated by the `Runtime`.

```python
from agentique.tools import ReadFile, Delegate
```

- **`ReadFile()`** — read the UTF-8 text contents of a file (read-only;
  failures returned as error results, not raised).
- **`Delegate(child, *, name, description, runtime=None)`** — expose a child
  `Agent` as a tool the parent may invoke. Pure mechanism for multi-agent
  topologies: it carries no orchestration logic, and is gated by the parent's
  permissions like any other tool.

Depends only on `agentique-core`.
