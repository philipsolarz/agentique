"""
Main GameAI class for the agentic game AI library.

This module provides the central interface for creating and managing
AI agents within a game environment.

Design Patterns:
- Facade Pattern: Provides a simplified interface to the complex subsystems
- Factory Pattern: Creates and configures Agent instances
- Registry Pattern: Maintains references to created agents
"""

from typing import Dict, List, Any, Callable, Optional, Union, Type
import logging
from pydantic import BaseModel

from .agent_core import Agent
from .models import FinalAnswer, AgentConfig, MessageAgentParameters
from .tool_registry import ToolRegistry
from .client import OpenAIClientWrapper
from .agent_registry import AgentRegistry
from .agent_communication import message_agent

logger = logging.getLogger(__name__)


class GameAI:
    """
    Main interface for the agentic game AI library.
    
    This class provides a high-level API for creating and managing agents,
    registering tools, and coordinating interactions.
    """
    
    def __init__(self, api_key: str, default_config: Optional[AgentConfig] = None):
        """
        Initialize the GameAI instance.
        
        Args:
            api_key: OpenAI API key
            default_config: Default configuration for new agents
        """
        self.api_key = api_key
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
        max_history_messages: int = 100
    ) -> Agent:
        """
        Create a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            system_prompt: Initial system prompt/persona
            model: OpenAI model to use (overrides default)
            max_history_messages: Maximum number of messages to keep in history
            
        Returns:
            The new Agent instance
        """
        # Add final answer instruction to system prompt
        final_answer_instruction = (
            "IMPORTANT: Always use the final_answer function to provide your final response "
            "instead of writing it directly. Structure your answer according to the function parameters."
        )
        
        full_system_prompt = system_prompt or self.default_config.system_prompt or ""
        if full_system_prompt:
            full_system_prompt = f"{full_system_prompt}\n\n{final_answer_instruction}"
        else:
            full_system_prompt = final_answer_instruction
        
        # Create configuration
        config = AgentConfig(
            agent_id=agent_id,
            system_prompt=full_system_prompt,
            model=model or self.default_config.model,
            temperature=self.default_config.temperature,
            max_history=max_history_messages
        )
        
        # Create OpenAI client
        client = OpenAIClientWrapper(
            api_key=self.api_key,
            model=config.model
        )
        
        # Create and store the agent
        agent = Agent(
            agent_id=agent_id,
            openai_client=client,
            system_prompt=config.system_prompt,
            tool_registry=self.tool_registry,
            max_history_messages=config.max_history
        )
        
        # Register the agent in both the local map and the global registry
        self.agents[agent_id] = agent
        AgentRegistry.register(agent_id, agent)
        
        logger.info(f"Created agent: {agent_id}")
        
        return agent
    
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
        # Register final_answer tool
        self.register_tool(
            name="final_answer",
            function=self._final_answer_fn,
            description="Submit a final answer to complete the current task. Always use this function to provide your final response instead of writing it directly.",
            parameter_model=FinalAnswer
        )
        
        # Register message_agent tool
        self.register_tool(
            name="message_agent",
            function=message_agent,
            description="Send a message to another agent and get a response.",
            parameter_model=MessageAgentParameters
        )
    
    def _final_answer_fn(self, **kwargs) -> Dict[str, Any]:
        """
        Built-in final answer function.
        
        Args:
            **kwargs: Fields for the FinalAnswer
            
        Returns:
            The final answer data
        """
        # Validate the final answer using our model
        final_answer = FinalAnswer(**kwargs)
        
        # Return the validated data
        return final_answer.model_dump()