"""Agentique testing utilities: deterministic, offline doubles for exercising the
agent loop without a network or API key — a ``Model`` (:class:`StubModel`) and a
``Tool`` (:class:`EchoTool`) — plus supporting types. Depends only on
``agentique-core`` contracts.
"""

from agentique.testing.collecting_sink import CollectingSink
from agentique.testing.echo_tool import EchoTool
from agentique.testing.stub_model import StubCall, StubModel, StubModelExhausted

__all__ = [
    "CollectingSink",
    "EchoTool",
    "StubCall",
    "StubModel",
    "StubModelExhausted",
]
