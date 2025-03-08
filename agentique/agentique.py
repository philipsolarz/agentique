"""
Main Agentique class for the Agentique library.

This module provides the central interface for creating and managing
AI agents within various environments.

Design Patterns:
- Facade Pattern: Provides a simplified interface to the complex subsystems
- Factory Pattern: Creates and configures Agent instances
- Registry Pattern: Maintains references to created agents
"""

from typing import Dict, List, Any, Callable, Optional, Union, Type
import logging
from pydantic import BaseModel

from .agent_core import Agent
from .models import StructuredResult, AgentConfig, MessageAgentParameters
from .tool_registry import ToolRegistry
from .client import OpenAIClientWrapper, AnthropicClientWrapper, BaseClientWrapper
from .agent_registry import AgentRegistry
from .agent_communication import message_agent

logger = logging.getLogger(__name__)


class Agentique:
    """
    Main interface for the Agentique library.
    
    This class provides a high-level API for creating and managing agents,
    registering tools, and coordinating interactions.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None, 
        anthropic_api_key: Optional[str] = None,
        default_config: Optional[AgentConfig] = None
    ):
        """
        Initialize the Agentique instance.
        
        Args:
            openai_api_key: OpenAI API key (optional if using Anthropic exclusively)
            anthropic_api_key: Anthropic API key (optional if using OpenAI exclusively)
            default_config: Default configuration for new agents
        """
        if not openai_api_key and not anthropic_api_key:
            raise ValueError("At least one API key (OpenAI or Anthropic) must be provided")
            
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.default_config = default_config or AgentConfig(agent_id="default")
        self.agents: Dict[str, Agent] = {}
        self.tool_registry = ToolRegistry()
        
        # Register common tools
        self._register_common_tools()
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: Optional[str] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
        parameter_model: Optional[Type[BaseModel]] = None
    ) -> None:
        """
        Register a tool that can be used by agents.
        
        Args:
            name: Name of the tool
            function: Function to execute
            description: Description of the tool
            parameters_schema: JSON schema for parameters
            parameter_model: Pydantic model for parameters
        """
        self.tool_registry.register_tool(
            name=name,
            function=function,
            description=description,
            parameters_schema=parameters_schema,
            parameter_model=parameter_model
        )
    
    def create_agent(
        self,
        agent_id: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_history_messages: int = 100,
        structured_output_model: Optional[Type[BaseModel]] = None
    ) -> Agent:
        """
        Create a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            system_prompt: Initial system prompt/persona
            model: LLM model to use (overrides default)
            provider: LLM provider to use (OpenAI or Anthropic, overrides default)
            max_history_messages: Maximum number of messages to keep in history
            structured_output_model: Optional Pydantic model for structured outputs
            
        Returns:
            The new Agent instance
        """
        # Add structured output instruction to system prompt
        structured_output_instruction = (
            "IMPORTANT: When asked to provide a structured or specific format output, "
            "use the structure_output function instead of writing it directly."
        )
        
        full_system_prompt = system_prompt or self.default_config.system_prompt or ""
        if full_system_prompt:
            full_system_prompt = f"{full_system_prompt}\n\n{structured_output_instruction}"
        else:
            full_system_prompt = structured_output_instruction
        
        # Use configured provider/model or defaults
        provider = provider or self.default_config.provider
        model = model or self.default_config.model
        
        # Create configuration
        config = AgentConfig(
            agent_id=agent_id,
            system_prompt=full_system_prompt,
            model=model,
            provider=provider,
            temperature=self.default_config.temperature,
            max_history=max_history_messages
        )
        
        # Create API client based on provider
        client = self._create_client(config.provider, config.model)
        
        # Create and store the agent
        agent = Agent(
            agent_id=agent_id,
            client=client,
            system_prompt=config.system_prompt,
            tool_registry=self.tool_registry,
            max_history_messages=config.max_history,
            structured_output_model=structured_output_model or StructuredResult
        )
        
        # Register the agent in both the local map and the global registry
        self.agents[agent_id] = agent
        AgentRegistry.register(agent_id, agent)
        
        logger.info(f"Created agent: {agent_id}")
        
        return agent
    
    def _create_client(self, provider: str, model: str) -> BaseClientWrapper:
        """
        Create an API client based on the provider.
        
        Args:
            provider: The LLM provider (openai or anthropic)
            model: The model to use
            
        Returns:
            An instance of BaseClientWrapper
            
        Raises:
            ValueError: If an unsupported provider is specified or the API key is missing
        """
        if provider.lower() == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")
            return OpenAIClientWrapper(api_key=self.openai_api_key, model=model)
        elif provider.lower() == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key is required for Anthropic provider")
            return AnthropicClientWrapper(api_key=self.anthropic_api_key, model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'anthropic'.")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an existing agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            The Agent instance or None if not found
        """
        return self.agents.get(agent_id)
    
    def _register_common_tools(self) -> None:
        """Register common built-in tools."""
        # Register structure_output tool
        self.register_tool(
            name="structure_output",
            function=self._structure_output_fn,
            description="Provide a structured output to complete the current task. Use this function to provide your final response in a structured format.",
            parameter_model=StructuredResult
        )
        
        # Register message_agent tool
        self.register_tool(
            name="message_agent",
            function=message_agent,
            description="Send a message to another agent and get a response.",
            parameter_model=MessageAgentParameters
        )
    
    def _structure_output_fn(self, **kwargs) -> Dict[str, Any]:
        """
        Built-in structured output function.
        
        Args:
            **kwargs: Fields for the StructuredResult
            
        Returns:
            The structured output data
        """
        # Validate the structured output using our model
        structured_result = StructuredResult(**kwargs)
        
        # Return the validated data
        return structured_result.model_dump()