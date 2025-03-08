"""
Agent registry for the agentic game AI library.

This module provides a registry for storing and retrieving agent instances,
enabling agent-to-agent communication.

Design Patterns:
- Registry Pattern: Provides a central store for agent instances
- Singleton Pattern: The registry is a singleton accessed via class methods
"""

from typing import Dict, Any, Optional, TypeVar
import logging
import threading

logger = logging.getLogger(__name__)

# Type variable for Agent for type hints
Agent = TypeVar('Agent')

class AgentRegistry:
    """
    Registry for storing and retrieving agent instances.
    
    The AgentRegistry provides a global store of agent instances that
    can be referenced by ID, enabling agent-to-agent communication.
    """
    
    # Class-level storage for agent instances
    _agents: Dict[str, Agent] = {}
    
    # Lock for thread-safe operations
    _lock = threading.RLock()
    
    @classmethod
    def register(cls, agent_id: str, agent: Agent) -> None:
        """
        Register an agent in the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance to register
        """
        with cls._lock:
            if agent_id in cls._agents:
                logger.warning(f"Overwriting existing agent with ID: {agent_id}")
            cls._agents[agent_id] = agent
            logger.debug(f"Registered agent: {agent_id}")
    
    @classmethod
    def get_agent(cls, agent_id: str) -> Optional[Agent]:
        """
        Get an agent from the registry.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            The agent instance or None if not found
        """
        with cls._lock:
            agent = cls._agents.get(agent_id)
            if agent is None:
                logger.warning(f"Agent not found: {agent_id}")
            return agent
    
    @classmethod
    def unregister(cls, agent_id: str) -> None:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: ID of the agent to remove
        """
        with cls._lock:
            if agent_id in cls._agents:
                del cls._agents[agent_id]
                logger.debug(f"Unregistered agent: {agent_id}")
            else:
                logger.warning(f"Attempted to unregister non-existent agent: {agent_id}")
    
    @classmethod
    def list_agents(cls) -> Dict[str, Agent]:
        """
        Get a dictionary of all registered agents.
        
        Returns:
            Dictionary mapping agent IDs to agent instances
        """
        with cls._lock:
            return cls._agents.copy()