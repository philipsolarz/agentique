"""
Data models for the agentic game AI library.

This module contains Pydantic models for message representation,
configuration, and structured outputs.

Design Patterns:
- Data Transfer Object (DTO): Models represent data structures for transfer
- Validator Pattern: Models include validation logic for their fields
"""

from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, model_validator, ConfigDict


class MessageRole(str, Enum):
    """Enumeration of valid message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """
    Represents a tool call made by the assistant.
    
    Attributes:
        id: Unique identifier for this tool call
        type: Type of tool call (always "function" for current OpenAI API)
        function: Details about the function being called
    """
    id: str
    type: str = "function"  # Currently always "function" in OpenAI API
    function: Dict[str, Any]
    
    model_config = ConfigDict(
        extra="allow"  # Allow additional fields for future API compatibility
    )


class MessageModel(BaseModel):
    """
    Represents a message in the conversation history.
    
    This model matches the OpenAI API message format for Chat Completions.
    
    Attributes:
        role: The role of the message sender (system, user, assistant, or tool)
        content: The text content of the message (can be None for function calls)
        name: Name identifier (used for tool responses)
        tool_calls: List of tool calls initiated by the assistant
        tool_call_id: ID of the tool call this message is responding to
    """
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for future API compatibility
        populate_by_name=True  # Allow populating by field name
    )
    
    @model_validator(mode='after')
    def validate_content_or_tool_calls(self):
        """Ensure that assistant messages have either content or tool_calls."""
        if self.role == MessageRole.ASSISTANT:
            if self.content is None and not self.tool_calls:
                raise ValueError("Assistant messages must have either content or tool_calls")
        elif self.role == MessageRole.TOOL:
            if self.content is None:
                raise ValueError("Tool messages must have content")
            if not self.tool_call_id:
                raise ValueError("Tool messages must have a tool_call_id")
        return self


class ToolParameters(BaseModel):
    """
    Base class for tool parameter definitions.
    
    This class should be extended by specific tool parameter models.
    """
    model_config = ConfigDict(
        extra="forbid",  # Prevent additional fields not defined in the model
        json_schema_extra={
            "examples": []  # Can be populated by subclasses
        }
    )


class GameAction(str, Enum):
    """Common game actions for the FinalAnswer model."""
    MOVE = "move"
    ATTACK = "attack"
    DEFEND = "defend"
    INTERACT = "interact"
    SPEAK = "speak"
    USE_ITEM = "use_item"
    OBSERVE = "observe"
    WAIT = "wait"
    OTHER = "other"


class FinalAnswer(BaseModel):
    """
    Structured format for the agent's final answer in a game context.
    
    Attributes:
        action: The type of action being taken
        message: A textual description or message
        target: Optional target of the action (character, item, location)
        confidence: Confidence level (0-1) in the answer
        reasoning: Optional reasoning behind the decision
        metadata: Optional additional metadata about the response
    """
    action: Union[GameAction, str] = Field(..., 
        description="The type of action being taken")
    message: str = Field(..., 
        description="Description of the action or response")
    target: Optional[str] = Field(None, 
        description="Target of the action (character, item, location)")
    confidence: float = Field(..., ge=0, le=1, 
        description="Confidence level (0-1)")
    reasoning: Optional[str] = Field(None, 
        description="Reasoning behind the decision")
    metadata: Dict[str, Any] = Field(default_factory=dict, 
        description="Additional metadata about the action")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "action": "move",
                    "message": "I move cautiously toward the cave entrance",
                    "target": "cave entrance",
                    "confidence": 0.9,
                    "reasoning": "The cave seems to be the source of the strange noises, and I need to investigate.",
                    "metadata": {"energy_cost": 2, "time_taken": "1 minute"}
                }
            ]
        }
    )


class MessageAgentParameters(ToolParameters):
    """Parameters for messaging another agent."""
    target_agent_id: str = Field(..., description="ID of the agent to message")
    message: str = Field(..., description="Message to send to the agent")
    maintain_context: bool = Field(False, description="Whether to include conversation context")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"target_agent_id": "assistant", "message": "What's the capital of France?", "maintain_context": False},
                {"target_agent_id": "expert", "message": "Can you analyze this data?", "maintain_context": True}
            ]
        }
    )


class MoveParameters(ToolParameters):
    """Parameters for movement actions."""
    direction: str = Field(..., 
        description="Direction to move (north, south, east, west, up, down)")
    distance: Optional[float] = Field(1.0, 
        description="Distance to move in the specified direction")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"direction": "north", "distance": 2.5},
                {"direction": "up", "distance": 1.0}
            ]
        }
    )


class AgentConfig(BaseModel):
    """
    Configuration for an agent.
    
    Attributes:
        agent_id: Unique identifier for the agent
        system_prompt: Base system prompt or persona
        model: OpenAI model name to use
        temperature: Sampling temperature for responses
        max_history: Maximum number of messages to keep in history
    """
    agent_id: str
    system_prompt: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_history: int = 100