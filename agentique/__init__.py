"""
Agentic Game AI Library

A library for creating AI agents that can interact with game worlds through
function calls and communicate with other agents.

This library provides a robust framework for building AI agents that can:
- Connect to OpenAI's language models
- Execute dynamic tools/functions
- Maintain conversation history
- Communicate with other agents
- Produce structured outputs

The main components are:
- GameAI: Main interface for creating and managing agents
- Agent: Core class that handles interactions with AI models
- ToolRegistry: Registry for managing available tools
- MessageModel: Representation of conversation messages
- FinalAnswer: Structured format for agent outputs
"""

from .agent_core import Agent, MAX_TOOL_ITERATIONS, DEFAULT_MAX_RECURSION_DEPTH
from .models import MessageModel, FinalAnswer, AgentConfig, ToolParameters, MessageAgentParameters
from .tool_registry import ToolRegistry
from .client import OpenAIClientWrapper
from .game_ai import GameAI
from .agent_registry import AgentRegistry
from .agent_communication import message_agent, message_agent_sync
from .logging_config import configure_logging, get_logger
from .exceptions import (
    AgenticGameAIError, ToolExecutionError, ToolNotFoundError,
    AgentNotFoundError, APIError, MaxRecursionError, 
    InvalidArgumentError, MessageHistoryError
)

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "Agent",
    "GameAI",
    "ToolRegistry",
    "AgentRegistry",
    
    # Models
    "MessageModel", 
    "FinalAnswer", 
    "AgentConfig", 
    "ToolParameters",
    "MessageAgentParameters",
    
    # API wrapper
    "OpenAIClientWrapper",
    
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
    "AgenticGameAIError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "AgentNotFoundError",
    "APIError",
    "MaxRecursionError",
    "InvalidArgumentError",
    "MessageHistoryError"
]