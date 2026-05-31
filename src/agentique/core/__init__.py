"""Agentique core: the application-agnostic agent framework contracts.

This package holds the execution + topology substrate an agent is built from — the
seam Protocols (Model, Tool, Memory), the neutral-IR value types (messages, Context,
Result, Agent, Permissions, events), the middleware onion, typed-I/O validation, the
``Engine`` that drives a single agent, and the ``Scheduler`` that coordinates many.
Its only third-party dependency is the approved base dep Pydantic; provider SDKs and
telemetry exporters live in satellite packages behind extras.

This module re-exports the public surface; submodules hold one primitive each.
"""

from agentique.core.agent import Agent, Effect, Permissions, Rule
from agentique.core.compaction import (
    CompactionMiddleware,
    Compactor,
    EvictOldestToolResults,
    estimate_size,
)
from agentique.core.context import Context
from agentique.core.control import PauseRequested, PermissionDenied
from agentique.core.events import (
    Compaction,
    Dispatch,
    Event,
    EventSink,
    ModelCallFinished,
    ModelCallStarted,
    NullSink,
    ToolCalled,
    TurnBoundary,
)
from agentique.core.memory import Memory
from agentique.core.messages import (
    ContentBlock,
    Message,
    ModelResponse,
    OpaqueBlock,
    Role,
    StopKind,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from agentique.core.middleware import (
    Middleware,
    ModelPoint,
    PermissionMiddleware,
    Point,
    PreModelPoint,
    ToolPoint,
    TracingMiddleware,
    TurnPoint,
)
from agentique.core.model import Model
from agentique.core.result import Blocked, Completed, NeedsHuman, Paused, Result
from agentique.core.run_context import Dispatcher, RunContext
from agentique.core.runtime import Engine
from agentique.core.scheduler import Run, RunState, Scheduler
from agentique.core.tool import Tool, ToolResult, ToolSpec
from agentique.core.validation import PydanticValidator, Validator

__version__ = "0.0.1"

__all__ = [
    "Agent",
    "Blocked",
    "Compaction",
    "CompactionMiddleware",
    "Compactor",
    "Completed",
    "ContentBlock",
    "Context",
    "Dispatch",
    "Dispatcher",
    "Effect",
    "Engine",
    "Event",
    "EventSink",
    "EvictOldestToolResults",
    "Memory",
    "Message",
    "Middleware",
    "Model",
    "ModelCallFinished",
    "ModelCallStarted",
    "ModelPoint",
    "ModelResponse",
    "NeedsHuman",
    "NullSink",
    "OpaqueBlock",
    "PauseRequested",
    "Paused",
    "PermissionDenied",
    "PermissionMiddleware",
    "Permissions",
    "Point",
    "PreModelPoint",
    "PydanticValidator",
    "Result",
    "Role",
    "Rule",
    "Run",
    "RunContext",
    "RunState",
    "Scheduler",
    "StopKind",
    "StopReason",
    "TextBlock",
    "Tool",
    "ToolCalled",
    "ToolPoint",
    "ToolResult",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "TracingMiddleware",
    "TurnBoundary",
    "TurnPoint",
    "Usage",
    "Validator",
    "__version__",
    "estimate_size",
]
