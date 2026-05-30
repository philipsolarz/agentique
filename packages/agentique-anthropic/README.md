# agentique-anthropic

The Anthropic provider for Agentique: a `Model` implementation over the Anthropic
Messages API.

```python
from agentique.anthropic import AnthropicModel

model = AnthropicModel("<current-model-id>")  # id is environment-specific
```

`AnthropicModel(model, *, max_tokens=4096, client=None)` satisfies
`agentique.core.Model`. The model id has **no default** — supply a current one
(confirm it in the Anthropic console; do not hardcode a guess). Pass a custom
`AsyncAnthropic` client to control auth, base URL, or retries.

Depends on `agentique-core` and the official `anthropic` SDK.
