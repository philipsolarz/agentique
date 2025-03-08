"""
Agentique: Agentic AI Library

A library for creating AI agents that can interact with environments through
function calls and communicate with other agents.

This library provides a robust framework for building AI agents that can:
- Connect to language models (OpenAI, Anthropic)
- Execute dynamic tools/functions
- Maintain conversation history
- Communicate with other agents
- Produce structured outputs

The main components are:
- Agentique: Main interface for creating and managing agents
- Agent: Core class that handles interactions with AI models
- ToolRegistry: Registry for managing available tools
- MessageModel: Representation of conversation messages
- StructuredResult: Base model for structured agent outputs
"""

from .agent_core import Agent, MAX_TOOL_ITERATIONS, DEFAULT_MAX_RECURSION_DEPTH
from .models import MessageModel, StructuredResult, AgentConfig, ToolParameters, MessageAgentParameters
from .tool_registry import ToolRegistry
from .client import OpenAIClientWrapper, AnthropicClientWrapper
from .agentique import Agentique
from .agent_registry import AgentRegistry
from .agent_communication import message_agent, message_agent_sync
from .logging_config import configure_logging, get_logger
from .exceptions import (
    AgenticError, ToolExecutionError, ToolNotFoundError,
    AgentNotFoundError, APIError, MaxRecursionError, 
    InvalidArgumentError, MessageHistoryError
)

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "Agent",
    "Agentique",
    "ToolRegistry",
    "AgentRegistry",
    
    # Models
    "MessageModel", 
    "StructuredResult", 
    "AgentConfig", 
    "ToolParameters",
    "MessageAgentParameters",
    
    # API wrappers
    "OpenAIClientWrapper",
    "AnthropicClientWrapper",
    
    # Constants
    "MAX_TOOL_ITERATIONS", 
    "DEFAULT_MAX_RECURSION_DEPTH",
    
    # Agent communication
    "message_agent",
    "message_agent_sync",
    
    # Logging
    "configure_logging",
    "get_logger",
    
    # Exceptions
    "AgenticError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "AgentNotFoundError",
    "APIError",
    "MaxRecursionError",
    "InvalidArgumentError",
    "MessageHistoryError"
]