# agentique-core

The application-agnostic contracts and engine of the Agentique framework. **Zero
third-party dependencies.**

```python
import agentique.core
```

## Public surface

- **Seams (Protocols):** `Model`, `Tool`, `Skill`, `Memory` — structural
  interfaces; implement by shape, no inheritance.
- **Value types (frozen):** `Message`, `TextBlock`, `ToolUseBlock`,
  `ToolResultBlock`, `ModelResponse`, `Context`, `Agent`, `Permissions`,
  `ToolSpec`, `ToolResult`.
- **Result union:** `Result = Completed | NeedsHuman | Blocked` — branch with an
  exhaustive `match`.
- **Engine:** `Runtime` — drives one declarative `Agent` to a `Result`
  (assemble context → call model → dispatch permitted tools → fold results →
  finish). Run-control (`max_turns`) lives here, not on the `Agent`.

Concrete implementations live in sibling packages (`agentique-anthropic`,
`agentique-skills`, `agentique-tools`, `agentique-memory`, `agentique-testing`)
that depend on this one.
