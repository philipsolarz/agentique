"""Agentique Anthropic provider: a :class:`~agentique.core.model.Model`
implementation backed by the Anthropic Messages API. Depends on
``agentique-core`` contracts and the official ``anthropic`` SDK.
"""

from agentique.anthropic.model import AnthropicModel

__all__ = ["AnthropicModel"]
