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
from .core.protocols import AgentAdapter, BridgeMiddleware, ToolMapper

# Bridge layer
from .bridge.router import AgentRouter, KeywordRouter, DirectRouter
from .bridge.provider import AgentProvider
from .bridge.task_manager import TaskManager

# Server factory
from .server import create_server

__all__ = [
    # Core types
    "AgentEvent",
    "AgentHierarchy",
    "AgentInfo",
    "AgentResponse",
    "BridgeContext",
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
    "AgentAdapter",
    "BridgeMiddleware",
    "ToolMapper",
    # Bridge
    "AgentProvider",
    "AgentRouter",
    "DirectRouter",
    "KeywordRouter",
    "TaskManager",
    # Server
    "create_server",
]
