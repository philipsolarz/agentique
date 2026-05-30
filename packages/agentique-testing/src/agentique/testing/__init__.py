"""Agentique testing utilities: a deterministic, offline ``Model`` for exercising
the agent loop without a network or API key, plus supporting types. Depends only
on ``agentique-core`` contracts.
"""

from agentique.testing.stub_model import StubCall, StubModel, StubModelExhausted

__all__ = ["StubCall", "StubModel", "StubModelExhausted"]
