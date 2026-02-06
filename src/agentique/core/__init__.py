"""Core protocols, types, config, and errors for agentique."""

from .config import AdapterConfig, AgentiqueConfig
from .errors import (
    AdapterError,
    AgentNotFoundError,
    AgentUnavailableError,
    AgentiqueError,
    InputRequiredError,
    TaskNotFoundError,
    TranslationError,
)
from .events import AsyncEventEmitter, EventHook
from .protocols import AgentAdapter, BridgeMiddleware, ToolMapper
from .types import (
    AgentEvent,
    AgentHierarchy,
    AgentInfo,
    AgentResponse,
    BridgeContext,
    StreamChunk,
    SubAgentInfo,
    TaskState,
    TaskTracker,
)

__all__ = [
    "AdapterConfig",
    "AgentiqueConfig",
    "AdapterError",
    "AgentNotFoundError",
    "AgentUnavailableError",
    "AgentiqueError",
    "InputRequiredError",
    "TaskNotFoundError",
    "TranslationError",
    "AsyncEventEmitter",
    "EventHook",
    "AgentAdapter",
    "BridgeMiddleware",
    "ToolMapper",
    "AgentEvent",
    "AgentHierarchy",
    "AgentInfo",
    "AgentResponse",
    "BridgeContext",
    "StreamChunk",
    "SubAgentInfo",
    "TaskState",
    "TaskTracker",
]
