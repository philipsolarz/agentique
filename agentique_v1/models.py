"""
Data models for the Agentique library.

This module contains Pydantic models for message representation,
configuration, and structured outputs.

Design Patterns:
- Data Transfer Object (DTO): Models represent data structures for transfer
- Validator Pattern: Models include validation logic for their fields
"""

from typing import Optional, List, Dict, Any, Union, Literal, Type, ClassVar, get_type_hints
import json
import logging
from enum import Enum
from pydantic import BaseModel, Field, model_validator, ConfigDict, create_model

logger = logging.getLogger(__name__)

# Schema for individual property definitions in the parameters schema.
class JSONSchemaProperty(BaseModel):
    type: Union[str, List[str]] = Field(
        ...,
        description="The property type, e.g. 'string' or ['string', 'null'] for optional fields."
    )
    description: Optional[str] = Field(
        None,
        description="A description of the property."
    )
    enum: Optional[List[Any]] = Field(
        None,
        description="Optional enumeration of valid values for this property."
    )

# Schema for the JSON parameters which must be an object with defined properties.
class JSONSchemaParameters(BaseModel):
    type: Literal["object"] = Field(
        "object",
        const=True,
        description="Must be the string 'object'."
    )
    properties: Dict[str, JSONSchemaProperty] = Field(
        ...,
        description="A mapping of property names to their schema definitions."
    )
    required: List[str] = Field(
        ...,
        description="A list of required property names."
    )
    additionalProperties: bool = Field(
        ...,
        description="Indicates whether additional properties are allowed (should be False in strict mode)."
    )

# Schema representing a function's definition.
class FunctionDefinition(BaseModel):
    name: str = Field(
        ...,
        description="The name of the function (e.g. 'get_weather')."
    )
    description: str = Field(
        ...,
        description="A detailed description of when and how to use the function."
    )
    parameters: JSONSchemaParameters = Field(
        ...,
        description="The JSON schema defining the input arguments for the function."
    )
    strict: bool = Field(
        ...,
        description="If true, enforces that function calls adhere strictly to the schema."
    )

# Top-level schema for a tool function as used in function calling.
class ToolFunction(BaseModel):
    type: Literal["function"] = Field(
        "function",
        const=True,
        description="Must be the string 'function'."
    )
    function: FunctionDefinition

# Schema for a function call's function field in the response.
class ToolCallFunction(BaseModel):
    name: str = Field(
        ...,
        description="The name of the function being called."
    )
    arguments: str = Field(
        ...,
        description="A JSON-encoded string of the function call arguments."
    )

# Schema for a function call response.
class ToolCall(BaseModel):
    id: str = Field(
        ...,
        description="A unique identifier for the function call."
    )
    type: Literal["function"] = Field(
        "function",
        const=True,
        description="The type of tool call (always 'function')."
    )
    function: ToolCallFunction


class MessageRole(str, Enum):
    """Enumeration of valid message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


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
        refusal: If the model refuses to generate a response, this field will contain the refusal message
    """
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    refusal: Optional[str] = None
    
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for future API compatibility
        populate_by_name=True  # Allow populating by field name
    )
    
    @model_validator(mode='after')
    def validate_content_or_tool_calls(self):
        """Ensure that assistant messages have either content, tool_calls, or refusal."""
        if self.role == MessageRole.ASSISTANT:
            if self.content is None and not self.tool_calls and self.refusal is None:
                raise ValueError("Assistant messages must have either content, tool_calls, or refusal")
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
        extra="forbid",
    )


class StructuredResult(BaseModel):
    """
    Base model for structured outputs from agents.
    
    This is a generic base class that can be extended by users to create
    domain-specific structured output formats. For OpenAI integration,
    this class will be used directly with client.beta.chat.completions.parse.
    """
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields defined in subclasses
    )
    
    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "StructuredResult":
        """
        Create an instance of this class from a dictionary.
        
        Args:
            data: Dictionary of data
            
        Returns:
            Instance of this class
        """
        return cls.model_validate(data)


class MessageAgentParameters(ToolParameters):
    """Parameters for messaging another agent."""
    target_agent_id: str = Field(..., description="ID of the agent to message")
    message: str = Field(..., description="Message to send to the agent")
    maintain_context: bool = Field(False, description="Whether to include conversation context")

class AgentConfig(BaseModel):
    """
    Configuration for an agent.
    
    Attributes:
        agent_id: Unique identifier for the agent
        system_prompt: Base system prompt or persona
        model: Model name to use (defaults to OpenAI's gpt-4o-mini)
        provider: AI provider to use ('openai' or 'anthropic')
        temperature: Sampling temperature for responses
        max_history: Maximum number of messages to keep in history
        structured_output: Whether to use structured output
    """
    agent_id: str
    system_prompt: Optional[str] = None
    model: str = "gpt-4o-mini"
    provider: str = "openai"  # 'openai' or 'anthropic'
    temperature: float = 0.7
    max_history: int = 100
    structured_output: bool = True