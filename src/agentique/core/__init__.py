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
from .protocols import AdapterFactory, AgentAdapter, BridgeMiddleware, ToolMapper
from .registry import create_adapter, discover_adapters, list_protocols, register_adapter
from .tool_mapper import DefaultToolMapper, FlatHierarchyToolMapper, PerSkillToolMapper
from .types import (
    AgentEvent,
    AgentHierarchy,
    AgentInfo,
    AgentResponse,
    BridgeContext,
    ContextMapping,
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
    "AdapterFactory",
    "AgentAdapter",
    "BridgeMiddleware",
    "ToolMapper",
    "create_adapter",
    "discover_adapters",
    "list_protocols",
    "register_adapter",
    "DefaultToolMapper",
    "FlatHierarchyToolMapper",
    "PerSkillToolMapper",
    "AgentEvent",
    "AgentHierarchy",
    "AgentInfo",
    "AgentResponse",
    "BridgeContext",
    "ContextMapping",
    "StreamChunk",
    "SubAgentInfo",
    "TaskState",
    "TaskTracker",
]
