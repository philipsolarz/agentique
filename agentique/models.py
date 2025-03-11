"""
Core data models for the Agentique library.

This module contains Pydantic models for message representation,
configuration, and structured outputs.
"""

from typing import Literal, Optional, List, Dict, Any, Type, ClassVar, Union
from enum import Enum
from pydantic import BaseModel, Field, model_validator, ConfigDict

class MessageRole(str, Enum):
    """Enumeration of valid message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class ToolCallFunction(BaseModel):
    """
    Schema for a function call's function field in the API response.
    """
    name: str = Field(
        ...,
        description="The name of the function being called."
    )
    arguments: str = Field(
        ...,
        description="A JSON-encoded string of the function call arguments."
    )

class ToolCall(BaseModel):
    """
    Schema for a function call in the API response.
    """
    id: str = Field(
        ...,
        description="A unique identifier for the function call."
    )
    type: Literal["function"] = Field(
        "function",
        description="The type of tool call (always 'function')."
    )
    function: ToolCallFunction

class Message(BaseModel):
    """
    A message in the conversation history.
    
    Follows the OpenAI chat completions API format.
    
    Attributes:
        role: The role of the message sender (system, user, assistant, or tool)
        content: The text content of the message (can be None for function calls)
        name: Name identifier (used for tool responses)
        tool_calls: List of tool calls initiated by the assistant
        tool_call_id: ID of the tool call this message is responding to
    """
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    refusal: Optional[str] = None
    
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for future compatibility
        populate_by_name=True  # Allow populating by field name
    )
    
    @model_validator(mode='after')
    def validate_message(self):
        """Ensure the message has valid content based on its role."""
        if self.role == MessageRole.ASSISTANT:
            if self.content is None and not self.tool_calls:
                raise ValueError("Assistant messages must have either content or tool_calls")
        elif self.role == MessageRole.TOOL:
            if self.content is None:
                raise ValueError("Tool messages must have content")
            if not self.tool_call_id:
                raise ValueError("Tool messages must have a tool_call_id")
        return self

class StructuredOutput(BaseModel):
    """
    Base model for structured outputs from agents.
    
    This is a generic base class that can be extended to create
    domain-specific structured output formats.
    """
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields in subclasses
    )
    
    @classmethod
    def schema_json(cls) -> str:
        """Get the JSON schema for this model."""
        return cls.model_json_schema()

class AgentConfig(BaseModel):
    """
    Configuration for an agent.
    
    Attributes:
        name: Name identifier for the agent
        model: OpenAI model to use (must be gpt-4o-mini or newer)
        system_prompt: Base system prompt for the agent
        temperature: Sampling temperature for responses (0.0 to 2.0)
        max_history: Maximum number of messages to keep in history
    """
    name: str
    model: str = Field(default="gpt-4o-mini")
    system_prompt: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_history: int = Field(default=100, gt=0)
    
    @model_validator(mode='after')
    def validate_model(self):
        """Ensure the model is a supported version."""
        supported_prefixes = ['gpt-4o', 'gpt-4-', 'gpt-4.', 'o1-']
        if not any(self.model.startswith(prefix) for prefix in supported_prefixes):
            raise ValueError(f"Model {self.model} is not supported. Must be GPT-4o or newer.")
        return self