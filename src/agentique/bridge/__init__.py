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
)
from .fastmcp_middleware import AgentiqueMiddleware
from .middleware import (
    ErrorMappingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareChain,
    RateLimitMiddleware,
)
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
]
