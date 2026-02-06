"""Bridge layer components for routing, provider mapping, and task state."""

from .context_manager import ContextManager
from .dependencies import (
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
from .fastmcp_middleware import AgentiqueMiddleware
from .health import AgentHealth, HealthMonitor
from .middleware import (
    ErrorMappingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareChain,
    RateLimitMiddleware,
)
from .output_models import (
    AgentHealthOutput,
    AgentInspectOutput,
    AgentListOutput,
    AgentMessageOutput,
    AgentSummary,
    ErrorOutput,
    HealthCheckOutput,
    TaskStatusOutput,
    WebhookNotificationOutput,
)
from .persistent_stores import DynamoDBTaskStore, RedisTaskStore
from .provider import AgentProvider
from .router import (
    AgentRouter,
    DirectRouter,
    KeywordRouter,
    LLMRouter,
    RoutingStrategy,
    WeightedKeywordRouter,
)
from .storage import InMemoryTaskStore, TaskStore
from .task_manager import TaskManager
from .visibility import AgentVisibility
from .webhook import PushNotification, WebhookReceiver

__all__ = [
    "AgentiqueMiddleware",
    "ContextManager",
    "ErrorMappingMiddleware",
    "InMemoryTaskStore",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "MiddlewareChain",
    "RateLimitMiddleware",
    "AgentProvider",
    "AgentRouter",
    "DirectRouter",
    "KeywordRouter",
    "LLMRouter",
    "RoutingStrategy",
    "TaskStore",
    "WeightedKeywordRouter",
    "TaskManager",
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
    "TaskStatusOutput",
    "WebhookNotificationOutput",
    # Persistent stores
    "DynamoDBTaskStore",
    "RedisTaskStore",
    # Visibility
    "AgentVisibility",
    # Webhooks
    "PushNotification",
    "WebhookReceiver",
]
