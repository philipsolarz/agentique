"""
Agentique: Streamlined Agentic AI Library

A focused library for creating AI agents that can interact with environments through
function calls and produce structured outputs using modern OpenAI models.

Key components:
- Agent: Core class that handles interactions with AI models
- ToolRegistry: Registry for managing available tools
- OpenAIClient: Streamlined client for modern OpenAI models
"""

from .agent import Agent
from .tools import ToolRegistry, Tool
from .models import Message, StructuredOutput, AgentConfig, ToolDefinition
from .openai_client import OpenAIClient
from .exceptions import AgentiqueError, ToolExecutionError, ToolNotFoundError
from .logging import configure_logging, get_logger, print_json, print_code, print_object, console

__version__ = "0.2.0"
__all__ = [
    # Core classes
    "Agent",
    "ToolRegistry",
    "Tool",
    "OpenAIClient",
    
    # Models
    "Message", 
    "StructuredOutput", 
    "AgentConfig",
    "ToolDefinition",
    
    # Exceptions
    "AgentiqueError",
    "ToolExecutionError",
    "ToolNotFoundError",
    
    # Logging
    "configure_logging",
    "get_logger",
    "print_json",
    "print_code",
    "print_object",
    "console",
]

# Configure default logging
default_logger = configure_logging(level="INFO")
default_logger.info(f"🚀 Agentique v{__version__} initialized")