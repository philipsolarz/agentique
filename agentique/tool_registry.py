"""
Tool registry for the Agentique library.

This module provides functionality for registering, managing, and executing
tools that agents can use during interactions.

Design Patterns:
- Command Pattern: Each tool is a command that can be executed by name
- Factory Pattern: Creates tool specifications from function definitions
- Registry Pattern: Maintains a central registry of available tools
"""

from typing import Dict, List, Any, Callable, Optional, Union, Type
import inspect
import json
import asyncio
import logging
import functools
from pydantic import BaseModel

# Import OpenAI's pydantic_function_tool helper
from openai.lib._tools import pydantic_function_tool
    
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
        
        # Generate parameters schema using the most appropriate method
        tool_schema = None
        
        if parameters_schema:
            # Direct schema has highest priority
            tool_schema = parameters_schema
        elif parameter_model:
            # Use OpenAI's helper to generate the schema from the Pydantic model
            schema_object = pydantic_function_tool(
                parameter_model, 
                name=tool_name, 
                description=tool_description
            )
            # Extract the parameters schema from the tool object
            tool_schema = schema_object["function"]["parameters"]
        else:
            # If no schema or model provided, create a minimal schema
            tool_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        # Store the tool information
        self.tools[tool_name] = {
            "function": function,
            "description": tool_description,
            "parameters_schema": tool_schema,
            "is_async": asyncio.iscoroutinefunction(function)
        }
        
        logger.info(f"Registered tool: {tool_name}")
    
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
            
            # Create OpenAI-compatible tool definition with strict mode
            tool_def = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters_schema"],
                    "strict": True
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
    
    def _format_result(self, result: Any) -> Any:
        """
        Format a tool execution result for inclusion in the message history.
        
        Args:
            result: Raw result from tool execution
            
        Returns:
            Formatted result suitable for API
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