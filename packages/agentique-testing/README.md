# agentique-testing

Testing utilities for building on Agentique: a deterministic, offline `Model` for
exercising the agent loop without a network or API key.

```python
from agentique.testing import StubModel, StubCall, StubModelExhausted
```

- **`StubModel(responses)`** — replays a scripted sequence of `ModelResponse`s,
  one per `complete` call, and records every call as a `StubCall` for assertions.
  Raises `StubModelExhausted` if called more times than scripted.

Depends only on `agentique-core`.
