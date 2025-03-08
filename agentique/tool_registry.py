"""
Tool registry for the agentic game AI library.

This module provides functionality for registering, managing, and executing
tools that agents can use during interactions.

Design Patterns:
- Command Pattern: Each tool is a command that can be executed by name
- Factory Pattern: Creates tool specifications from function definitions
- Registry Pattern: Maintains a central registry of available tools
"""

from typing import Dict, List, Any, Callable, Optional, Union, Type, get_type_hints
import inspect
import json
import asyncio
import logging
import functools
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Manages available tools and their execution.
    
    The ToolRegistry handles registration, schema generation, and execution
    of tools (functions) that can be called by AI agents.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._schema_cache: Dict[str, Dict[str, Any]] = {}  # Cache for generated schemas
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: Optional[str] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
        parameter_model: Optional[Type[BaseModel]] = None
    ) -> None:
        """
        Register a new tool with metadata.
        
        Args:
            name: Name of the tool (if None, uses function.__name__)
            function: The function to execute
            description: Description of what the tool does (if None, uses docstring)
            parameters_schema: JSON Schema for the tool's parameters
            parameter_model: Pydantic model for parameters (alternative to schema)
        """
        # Use function name if not provided
        tool_name = name or function.__name__
        
        # Use docstring for description if not provided
        tool_description = description or inspect.getdoc(function) or f"Function {tool_name}"
        
        # Generate parameters schema if not provided
        tool_schema = None
        if parameters_schema:
            tool_schema = parameters_schema
        elif parameter_model:
            # Use Pydantic model schema
            tool_schema = parameter_model.model_json_schema()
        else:
            # Try to generate from type hints
            tool_schema = self._get_or_generate_schema(function)
        
        # Store the tool information
        self.tools[tool_name] = {
            "function": function,
            "description": tool_description,
            "parameters_schema": tool_schema,
            "is_async": asyncio.iscoroutinefunction(function)
        }
        
        logger.info(f"Registered tool: {tool_name}")
    
    def _get_or_generate_schema(self, function: Callable) -> Dict[str, Any]:
        """
        Get cached schema or generate a new one for a function.
        
        Args:
            function: The function to generate a schema for
            
        Returns:
            JSON schema for the function parameters
        """
        # Use function's qualified name as cache key
        cache_key = f"{function.__module__}.{function.__qualname__}"
        
        # Check if schema is already in cache
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]
        
        # Generate new schema
        schema = self._generate_schema_from_type_hints(function)
        
        # Store in cache
        self._schema_cache[cache_key] = schema
        
        return schema
    
    def get_tool_definitions(
        self, 
        tool_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions for specified tools.
        
        Args:
            tool_names: List of tool names to include (None for all)
            
        Returns:
            List of tool definitions in OpenAI format
        """
        tool_definitions = []
        
        # Determine which tools to include
        names_to_include = tool_names if tool_names is not None else self.tools.keys()
        
        # Build tool definitions
        for name in names_to_include:
            if name not in self.tools:
                logger.warning(f"Tool not found: {name}")
                continue
            
            tool_info = self.tools[name]
            
            # Create OpenAI-compatible tool definition
            tool_def = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters_schema"]
                }
            }
            
            tool_definitions.append(tool_def)
        
        return tool_definitions
    
    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool with given arguments.
        
        Supports both synchronous and asynchronous tool functions.
        Uses ThreadPoolExecutor for potentially blocking synchronous functions.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            KeyError: If the tool is not found
            TypeError: If arguments don't match function signature
            Exception: Any exception raised by the tool function
        """
        if tool_name not in self.tools:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        tool_info = self.tools[tool_name]
        function = tool_info["function"]
        is_async = tool_info["is_async"]
        
        try:
            logger.debug(f"Executing tool '{tool_name}' with arguments: {arguments}")
            
            # Execute the function with the provided arguments
            if is_async:
                # For async functions, await the result
                result = await function(**arguments)
            else:
                # For sync functions that might block, use ThreadPoolExecutor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # Use default executor
                    functools.partial(function, **arguments)
                )
            
            # Format the result if needed
            formatted_result = self._format_result(result)
            logger.debug(f"Tool '{tool_name}' execution successful: {formatted_result}")
            return formatted_result
            
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def _generate_schema_from_type_hints(self, function: Callable) -> Dict[str, Any]:
        """
        Generate a JSON schema from function type hints.
        
        Args:
            function: The function to generate a schema for
            
        Returns:
            JSON schema for the function parameters
        """
        # Get function signature and type hints
        sig = inspect.signature(function)
        type_hints = get_type_hints(function)
        
        # Create properties for each parameter
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # Skip self parameter for methods
            if param_name == 'self':
                continue
            
            # Get type hint for this parameter
            param_type = type_hints.get(param_name, Any)
            
            # Create property definition
            prop_def = self._type_to_schema(param_type)
            
            # Add description from parameter's default docstring if available
            if param.default is not inspect.Parameter.empty:
                if hasattr(param.default, '__doc__') and param.default.__doc__:
                    prop_def['description'] = param.default.__doc__
            
            properties[param_name] = prop_def
            
            # Add to required list if no default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        
        # Construct final schema
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
        
        return schema
    
    def _type_to_schema(self, type_hint: Any) -> Dict[str, Any]:
        """
        Convert a Python type hint to a JSON schema type.
        
        Args:
            type_hint: Python type or type hint
            
        Returns:
            Corresponding JSON schema type definition
        """
        # Basic type mappings
        if type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif type_hint is list or type_hint == List:
            return {"type": "array", "items": {}}
        elif type_hint is dict or type_hint == Dict:
            return {"type": "object"}
        elif hasattr(type_hint, '__origin__'):
            # Handle typing generics like List[str], Dict[str, int], etc.
            origin = type_hint.__origin__
            args = type_hint.__args__
            
            if origin is list or origin is List:
                return {
                    "type": "array",
                    "items": self._type_to_schema(args[0]) if args else {}
                }
            elif origin is dict or origin is Dict:
                return {"type": "object"}
            elif origin is Union:
                # Handle Optional[T] (Union[T, None])
                if type(None) in args:
                    non_none_args = [arg for arg in args if arg is not type(None)]
                    if len(non_none_args) == 1:
                        schema = self._type_to_schema(non_none_args[0])
                        schema["nullable"] = True
                        return schema
                # Regular Union
                return {"anyOf": [self._type_to_schema(arg) for arg in args]}
        
        # Default to any type if we can't determine
        return {}
    
    def _format_result(self, result: Any) -> Any:
        """
        Format a tool execution result for inclusion in the message history.
        
        Args:
            result: Raw result from tool execution
            
        Returns:
            Formatted result suitable for OpenAI API
        """
        # If already a string, return as is
        if isinstance(result, str):
            return result
        
        # If a Pydantic model, convert to dict
        if isinstance(result, BaseModel):
            result = result.model_dump()
        
        # For dicts, lists, or other JSON-serializable objects, convert to JSON string
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # If not JSON serializable, convert to string representation
            return str(result)