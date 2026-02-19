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
    ContentTypeNotSupportedError,
    InputRequiredError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
    TranslationError,
    UnsupportedOperationError,
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
    LLMRouter,
    RoutingStrategy,
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
from .bridge.auth import (
    ApiKeyCredentials,
    BearerCredentials,
    OAuthCodeCredentials,
    SecuritySchemeInfo,
    parse_security_scheme,
    select_auth_elicitation,
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
    get_session_router,
    get_session_adapter,
    get_session_config,
    set_session_override,
    clear_session_overrides,
)
from .bridge.fastmcp_middleware import AgentiqueMiddleware
from .bridge.lifespans import (
    compose_lifespans,
    make_cleanup_lifespan,
    make_health_monitor_lifespan,
)
from .bridge.health import AgentHealth, HealthMonitor
from .bridge.output_models import (
    AgentHealthOutput,
    AgentInspectOutput,
    AgentListOutput,
    AgentMessageOutput,
    AgentSummary,
    ErrorOutput,
    HealthCheckOutput,
    SubAgentSummary,
    TaskListOutput,
    TaskStatusOutput,
    TaskSummary,
    WebhookNotificationOutput,
)
from .bridge.persistent_stores import DynamoDBTaskStore, RedisTaskStore
from .bridge.storage import InMemoryTaskStore, TaskStore
from .bridge.visibility import AgentVisibility, TenantVisibilityMiddleware, VisibilityPolicy
from .bridge.webhook import PushNotification, WebhookReceiver

# Extensions passthrough
from .extensions import (
    ALL_EXTENSION_URIS,
    MCP_SESSION_URI,
    POLICY_CONTEXT_URI,
    ROUTING_METADATA_URI,
    TRACE_CONTEXT_URI,
    build_gateway_metadata,
    current_trace_context,
    pack_mcp_session,
    pack_policy_context,
    pack_routing_metadata,
    pack_trace_context,
    unpack_mcp_session,
    unpack_policy_context,
    unpack_routing_metadata,
    unpack_trace_context,
)

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
    # Auth schemes
    "ApiKeyCredentials",
    "BearerCredentials",
    "OAuthCodeCredentials",
    "SecuritySchemeInfo",
    "parse_security_scheme",
    "select_auth_elicitation",
    # Bridge
    "AgentProvider",
    "AgentRouter",
    "AgentiqueMiddleware",
    "ContextManager",
    "DirectRouter",
    "InMemoryTaskStore",
    "LLMRouter",
    "RoutingStrategy",
    "TaskStore",
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
    "get_session_router",
    "get_session_adapter",
    "get_session_config",
    "set_session_override",
    "clear_session_overrides",
    # Health monitoring
    "AgentHealth",
    "HealthMonitor",
    # Output models
    "AgentHealthOutput",
    "AgentInspectOutput",
    "AgentListOutput",
    "AgentMessageOutput",
    "AgentSummary",
    "ErrorOutput",
    "HealthCheckOutput",
    "SubAgentSummary",
    "TaskListOutput",
    "TaskStatusOutput",
    "TaskSummary",
    "WebhookNotificationOutput",
    # Persistent stores
    "DynamoDBTaskStore",
    "RedisTaskStore",
    # Visibility
    "AgentVisibility",
    "TenantVisibilityMiddleware",
    "VisibilityPolicy",
    # Webhooks
    "PushNotification",
    "WebhookReceiver",
    # Lifespans
    "compose_lifespans",
    "make_cleanup_lifespan",
    "make_health_monitor_lifespan",
    # Extensions
    "ALL_EXTENSION_URIS",
    "MCP_SESSION_URI",
    "POLICY_CONTEXT_URI",
    "ROUTING_METADATA_URI",
    "TRACE_CONTEXT_URI",
    "build_gateway_metadata",
    "current_trace_context",
    "pack_mcp_session",
    "pack_policy_context",
    "pack_routing_metadata",
    "pack_trace_context",
    "unpack_mcp_session",
    "unpack_policy_context",
    "unpack_routing_metadata",
    "unpack_trace_context",
    # Errors (new)
    "ContentTypeNotSupportedError",
    "PushNotificationNotSupportedError",
    "TaskNotCancelableError",
    "UnsupportedOperationError",
    # Server
    "create_server",
    "mount_bridge",
]
