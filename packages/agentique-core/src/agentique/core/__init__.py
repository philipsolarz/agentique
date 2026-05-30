"""Agentique core: the application-agnostic agent framework contracts.

This package holds **only** the generic contracts an agent is built from — the
seam Protocols (Model, Tool, Skill, Memory), the value types (messages, Context,
Result, Agent, Permissions), and (from A3) the Runtime that drives them. It has
**no third-party dependencies**: concrete seam implementations live in satellite
packages (``agentique.anthropic``, ``agentique.skills``, ``agentique.testing``,
…) that depend on this one.

This module re-exports the public surface; submodules hold one primitive each.
"""

from agentique.core.agent import Agent, Permissions
from agentique.core.context import Context
from agentique.core.memory import Memory
from agentique.core.messages import (
    ContentBlock,
    Message,
    ModelResponse,
    Role,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentique.core.model import Model
from agentique.core.result import Blocked, Completed, NeedsHuman, Result
from agentique.core.runtime import Runtime
from agentique.core.skill import Skill
from agentique.core.tool import Tool, ToolResult, ToolSpec

__version__ = "0.0.1"

__all__ = [
    "Agent",
    "Blocked",
    "Completed",
    "ContentBlock",
    "Context",
    "Memory",
    "Message",
    "Model",
    "ModelResponse",
    "NeedsHuman",
    "Permissions",
    "Result",
    "Role",
    "Runtime",
    "Skill",
    "StopReason",
    "TextBlock",
    "Tool",
    "ToolResult",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "__version__",
]
