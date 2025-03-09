"""
Data models for the Agentique library.

This module contains Pydantic models for message representation,
configuration, and structured outputs.

Design Patterns:
- Data Transfer Object (DTO): Models represent data structures for transfer
- Validator Pattern: Models include validation logic for their fields
"""

from typing import Optional, List, Dict, Any, Union, Literal, Type, ClassVar, get_type_hints
import copy
import json
import logging
from enum import Enum
from pydantic import BaseModel, Field, model_validator, ConfigDict, create_model

logger = logging.getLogger(__name__)


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
        extra="forbid",  # Prevent additional fields not defined in the model
        json_schema_extra={
            "examples": []  # Can be populated by subclasses
        }
    )


class StructuredResult(BaseModel):
    """
    Base model for structured outputs from agents.
    
    This is a generic base class that can be extended by users to create
    domain-specific structured output formats.
    
    For OpenAI integration, this class provides methods to generate JSON schema
    and create response_format parameters.
    """
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields defined in subclasses
    )
    
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """
        Generate a JSON schema for this model compatible with OpenAI's structured outputs.
        
        Returns:
            Dict containing the JSON schema
        """
        # Get the schema from Pydantic
        schema = cls.model_json_schema()
        
        # Process schema to be compatible with OpenAI's requirements
        processed = cls._process_schema_for_openai(schema)
        
        # Ensure additionalProperties is false at the root level
        if processed.get("type") == "object":
            processed["additionalProperties"] = False
            
        return processed
    @classmethod
    def _process_schema_for_openai(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a JSON schema to make it compatible with OpenAI's requirements.
        
        Args:
            schema: The original schema
            
        Returns:
            Processed schema compatible with OpenAI
        """
        # Base case for non-dict schemas
        if not isinstance(schema, dict):
            return schema
            
        # Create a new dict with only allowed properties
        processed = {}
        
        # Whitelist of keywords allowed in OpenAI schemas
        allowed_keywords = [
            "type", "properties", "required", "additionalProperties", 
            "items", "anyOf", "oneOf", "allOf", "$ref", "$defs",
            "description", "enum", "title"
        ]
        
        # Copy allowed fields
        for key in allowed_keywords:
            if key in schema:
                # For certain keys that have nested structure, process them recursively
                if key == "properties" and isinstance(schema[key], dict):
                    # Process each property recursively
                    properties = {}
                    for prop_name, prop_schema in schema[key].items():
                        properties[prop_name] = cls._process_schema_for_openai(prop_schema)
                    processed[key] = properties
                elif key == "items" and isinstance(schema[key], dict):
                    # Process array items recursively
                    processed[key] = cls._process_schema_for_openai(schema[key])
                elif key in ["anyOf", "oneOf", "allOf"] and isinstance(schema[key], list):
                    # Process composition schemas recursively
                    processed[key] = [cls._process_schema_for_openai(item) for item in schema[key]]
                else:
                    # Copy other allowed keys as is
                    processed[key] = schema[key]
        
        # Ensure we have type information
        if "type" in schema:
            processed["type"] = schema["type"]
        
        # Special handling for object types
        if schema.get("type") == "object":
            # Ensure additionalProperties is false
            processed["additionalProperties"] = False
            
            # For objects, make sure all properties are in required
            if "properties" in processed:
                # Get the set of property keys
                property_keys = set(processed["properties"].keys())
                
                # If no required field exists, add it with all properties
                if "required" not in processed or not processed["required"]:
                    processed["required"] = list(property_keys)
                else:
                    # Make sure required only includes keys that exist in properties
                    existing_required = set(processed["required"])
                    valid_required = existing_required.intersection(property_keys)
                    processed["required"] = list(valid_required)
                    
                    # If there are properties not in required, add them
                    missing_required = property_keys - valid_required
                    if missing_required:
                        processed["required"].extend(list(missing_required))
        
        return processed
    
    @classmethod
    def get_response_format(cls, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the response_format parameter for OpenAI API calls.
        
        Args:
            name: Optional name for the schema
            
        Returns:
            response_format dictionary for OpenAI API
        """
        # Get the raw schema from Pydantic
        schema = cls.model_json_schema()
        
        # Process the schema for OpenAI compatibility
        schema = cls._process_schema_for_openai(schema)
        
        # Log the final schema for debugging 
        schema_name = name or cls.__name__.lower()
        logger.debug(f"Final processed schema for {schema_name}: {json.dumps(schema, indent=2)}")
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema
            }
        }
    
    @classmethod
    def _process_schema_for_openai(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a schema to make it compatible with OpenAI's requirements.
        
        Args:
            schema: The JSON schema to process
        
        Returns:
            Processed schema compatible with OpenAI
        """
        # Create a deep copy to avoid modifying the input
        processed = copy.deepcopy(schema)
        
        # Set additionalProperties: false at root level
        if processed.get("type") == "object":
            processed["additionalProperties"] = False
        
        # Process $defs section if present (for enums and nested types)
        if "$defs" in processed:
            for def_name, def_schema in processed["$defs"].items():
                if def_schema.get("type") == "object":
                    # Set additionalProperties for each object in $defs
                    def_schema["additionalProperties"] = False
                    
                    # Make sure all properties are required
                    if "properties" in def_schema:
                        def_schema["required"] = list(def_schema["properties"].keys())
        
        # Handle properties and references
        if "properties" in processed:
            for prop_name, prop_schema in processed["properties"].items():
                # If this property uses $ref, remove any extra fields that are not allowed
                if isinstance(prop_schema, dict) and "$ref" in prop_schema:
                    # OpenAI doesn't allow other fields alongside $ref
                    # Keep only the $ref field
                    ref_value = prop_schema["$ref"]
                    processed["properties"][prop_name] = {"$ref": ref_value}
                # If this is an inline object, ensure it has additionalProperties: false
                elif isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
                    prop_schema["additionalProperties"] = False
                    
                    # Set required fields for the nested object
                    if "properties" in prop_schema:
                        prop_schema["required"] = list(prop_schema["properties"].keys())
                        
            # Make sure all top-level properties are required
            processed["required"] = list(processed["properties"].keys())
                    
        return processed
    
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


class MoveParameters(ToolParameters):
    """Parameters for movement actions."""
    direction: str = Field(..., 
        description="Direction to move (north, south, east, west, up, down)")
    distance: float = Field(...,  # Remove default value as OpenAI requires all properties
        description="Distance to move in the specified direction")


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