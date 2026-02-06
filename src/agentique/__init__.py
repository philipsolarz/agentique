"""Agentique — bridge any agent ecosystem to MCP.

Agentique is a Python framework that exposes any agent ecosystem as a
first-class MCP server. It translates MCP's tool/resource/prompt
primitives into agent protocol operations, letting any MCP client —
Claude, Cursor, ChatGPT, or custom hosts — seamlessly interact with
remote agents regardless of their underlying protocol.

Quick start::

    from agentique import AgentInfo, create_server

    server = create_server(agents=[
        AgentInfo(name="my-agent", base_url="http://localhost:9000"),
    ])
    server.run(transport="stdio")
"""

from __future__ import annotations

# Core types (protocol-agnostic)
from .core.types import (
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
from .core.config import AgentiqueConfig, AdapterConfig
from .core.errors import (
    AgentiqueError,
    AdapterError,
    AgentNotFoundError,
    AgentUnavailableError,
    InputRequiredError,
    TaskNotFoundError,
    TranslationError,
)
from .core.events import AsyncEventEmitter, EventHook
from .core.protocols import AdapterFactory, AgentAdapter, BridgeMiddleware, ToolMapper
from .core.registry import create_adapter, discover_adapters, list_protocols, register_adapter
from .core.telemetry import get_tracer, set_span_attribute, trace_agent_call
from .core.tool_mapper import DefaultToolMapper, FlatHierarchyToolMapper, PerSkillToolMapper

# Bridge layer
from .bridge.router import (
    AgentRouter,
    DirectRouter,
    KeywordRouter,
    LLMRouter,
    WeightedKeywordRouter,
)
from .bridge.provider import AgentProvider
from .bridge.task_manager import TaskManager
from .bridge.middleware import (
    ErrorMappingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareChain,
    RateLimitMiddleware,
)
from .bridge.context_manager import ContextManager
from .bridge.dependencies import (
    clear as clear_dependencies,
    configure as configure_dependencies,
    get_adapter,
    get_config,
    get_context_manager,
    get_emitter,
    get_router,
    get_task_manager,
)
from .bridge.fastmcp_middleware import AgentiqueMiddleware
from .bridge.storage import InMemoryTaskStore, TaskStore

# Server factory
from .server import create_server, mount_bridge

__all__ = [
    # Core types
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
    # Config
    "AdapterConfig",
    "AgentiqueConfig",
    # Errors
    "AdapterError",
    "AgentNotFoundError",
    "AgentUnavailableError",
    "AgentiqueError",
    "InputRequiredError",
    "TaskNotFoundError",
    "TranslationError",
    # Events
    "AsyncEventEmitter",
    "EventHook",
    # Protocols
    "AdapterFactory",
    "AgentAdapter",
    "BridgeMiddleware",
    "ToolMapper",
    # Registry
    "create_adapter",
    "discover_adapters",
    "list_protocols",
    "register_adapter",
    # Telemetry
    "get_tracer",
    "set_span_attribute",
    "trace_agent_call",
    # Tool mappers
    "DefaultToolMapper",
    "FlatHierarchyToolMapper",
    "PerSkillToolMapper",
    # Bridge
    "AgentProvider",
    "AgentRouter",
    "AgentiqueMiddleware",
    "ContextManager",
    "DirectRouter",
    "InMemoryTaskStore",
    "KeywordRouter",
    "LLMRouter",
    "TaskStore",
    "WeightedKeywordRouter",
    "TaskManager",
    # Middleware
    "ErrorMappingMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "MiddlewareChain",
    "RateLimitMiddleware",
    # Dependencies
    "clear_dependencies",
    "configure_dependencies",
    "get_adapter",
    "get_config",
    "get_context_manager",
    "get_emitter",
    "get_router",
    "get_task_manager",
    # Server
    "create_server",
    "mount_bridge",
]
