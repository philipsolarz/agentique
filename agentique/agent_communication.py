"""
Agent communication tools for the agentic game AI library.

This module provides tools for agent-to-agent communication, allowing
agents to interact with each other as part of a multi-agent system.

Design Patterns:
- Mediator Pattern: Facilitates communication between agents without them
  being directly aware of each other's implementations
- Chain of Responsibility: Messages can be passed between agents
"""

import logging
import inspect
import asyncio
from typing import Dict, Any, Union, Optional

from .agent_registry import AgentRegistry
from .models import FinalAnswer

logger = logging.getLogger(__name__)

# Maximum recursion depth for agent-to-agent calls
MAX_RECURSION_DEPTH = 5


async def message_agent(
    target_agent_id: str, 
    message: str,
    call_depth: int = 0,
    max_depth: Optional[int] = None,
    maintain_context: bool = False
) -> Union[Dict[str, Any], str]:
    """
    Send a message to another agent and get a response.
    
    This function allows an agent to communicate with another agent,
    enabling collaboration and information sharing.
    
    Args:
        target_agent_id: ID of the agent to message
        message: Message to send to the agent
        call_depth: Current recursion depth (internal use)
        max_depth: Maximum allowed recursion depth
        maintain_context: Whether to include a summary of the current conversation
                         context when messaging the target agent
        
    Returns:
        Response from the target agent
        
    Raises:
        ValueError: If the target agent is not found
        RuntimeError: If maximum recursion depth is exceeded
    """
    # If max_depth not provided, use the default
    from .agent_core import DEFAULT_MAX_RECURSION_DEPTH
    if max_depth is None:
        max_depth = DEFAULT_MAX_RECURSION_DEPTH
    
    # Check recursion depth to prevent infinite loops
    if call_depth >= max_depth:
        error_msg = f"Maximum recursion depth ({max_depth}) exceeded"
        logger.warning(f"{error_msg}. Cannot message agent {target_agent_id}.")
        return {"error": error_msg, "target_agent_id": target_agent_id}
    
    # Log the attempt
    logger.info(f"Messaging agent {target_agent_id} at depth {call_depth}")
    
    # Get the target agent from the registry
    target_agent = AgentRegistry.get_agent(target_agent_id)
    
    if not target_agent:
        error_msg = f"Agent {target_agent_id} not found"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Get the calling agent from the current context, if available
        calling_agent = None
        try:
            current_frame = inspect.currentframe()
            while current_frame:
                if 'self' in current_frame.f_locals:
                    self_obj = current_frame.f_locals['self']
                    # Check if it's an Agent instance
                    if hasattr(self_obj, 'agent_id') and hasattr(self_obj, 'message_history'):
                        calling_agent = self_obj
                        break
                current_frame = current_frame.f_back
        finally:
            # Explicitly delete the frame reference to avoid reference cycles
            del current_frame
        
        # Modify message with context if requested and possible
        actual_message = message
        if maintain_context and calling_agent:
            context_summary = calling_agent.summarize_history(max_length=300)
            actual_message = (
                f"Message from agent {calling_agent.agent_id}:\n\n"
                f"{message}\n\n"
                f"Context from our conversation:\n{context_summary}"
            )
        
        # Run the target agent with the message
        # Pass the increased call_depth to track recursion
        result = await target_agent.run(
            user_input=actual_message,
            call_depth=call_depth + 1,
            max_depth=max_depth
        )
        
        # Format the result based on its type
        if isinstance(result, FinalAnswer):
            # Convert FinalAnswer to dictionary
            return result.model_dump()
        elif isinstance(result, dict):
            # Already a dictionary
            return result
        else:
            # Convert string or other types to a response dict
            return {"response": str(result)}
            
    except Exception as e:
        error_msg = f"Error messaging agent {target_agent_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


# This is a synchronous version of message_agent for cases where async is not possible
def message_agent_sync(
    target_agent_id: str, 
    message: str,
    call_depth: int = 0,
    max_depth: Optional[int] = None
) -> Union[Dict[str, Any], str]:
    """
    Synchronous version of message_agent for non-async contexts.
    
    This function should be used only when async execution is not possible.
    It creates a new event loop to run the async function.
    
    Args:
        target_agent_id: ID of the agent to message
        message: Message to send to the agent
        call_depth: Current recursion depth
        max_depth: Maximum allowed recursion depth
        
    Returns:
        Response from the target agent
    """
    import asyncio
    
    try:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async function in the loop
        result = loop.run_until_complete(
            message_agent(target_agent_id, message, call_depth, max_depth)
        )
        
        # Close the loop
        loop.close()
        
        return result
        
    except Exception as e:
        error_msg = f"Error in synchronous message to agent {target_agent_id}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}