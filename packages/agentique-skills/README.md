# agentique-skills

Pure, deterministic `Skill` implementations for Agentique. Each is unit-testable
in complete isolation and touches neither the world nor the model.

```python
from agentique.skills import ExtractText
```

- **`ExtractText(separator="")`** — a `Skill[Message, str]` that joins a
  message's text blocks, ignoring tool-use/tool-result blocks.

Depends only on `agentique-core`.
