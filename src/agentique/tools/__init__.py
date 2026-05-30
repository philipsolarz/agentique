"""Agentique tools: real :class:`~agentique.core.tool.Tool` implementations.

Unlike skills (pure), tools cross external boundaries — the filesystem, the
network, the process. They depend only on ``agentique-core`` contracts.
"""

from agentique.tools.delegate import Delegate
from agentique.tools.read_file import ReadFile

__all__ = ["Delegate", "ReadFile"]
