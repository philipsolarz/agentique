"""
Tool registry and execution for the Agentique library.

Provides functionality for registering, managing and executing
tools that agents can use during interactions.
"""

import inspect
import json
import asyncio
import functools
from typing import Dict, List, Any, Callable, Optional, Union, Type, Protocol, TypeVar, runtime_checkable
from concurrent.futures import ThreadPoolExecutor
import time

from pydantic import BaseModel

from .utils import generate_function_schema
from .exceptions import ToolNotFoundError, ToolExecutionError
from .logging import get_logger, print_json, console

logger = get_logger("tools")

# Thread pool for executing synchronous functions
_THREAD_POOL = ThreadPoolExecutor(max_workers=10)

@runtime_checkable
class Tool(Protocol):
    """Protocol defining what constitutes a tool function."""
    __name__: str  # All callables have a __name__

T = TypeVar('T', bound=BaseModel)

class ToolRegistry:
    """
    Registry for managing available tools and their execution.
    
    Handles registration of tools, generating their schemas,
    and executing them when requested.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        logger.info("🛠️ Tool registry initialized")
    
    def register(
        self,
        func: Callable,
        parameter_model: Type[BaseModel],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Register a function as a tool.
        
        Args:
            func: The function to register
            parameter_model: Pydantic model for parameters (required)
            name: Custom name for the tool (defaults to function name)
            description: Description of what the tool does (defaults to docstring)
        """
        # Determine tool name
        tool_name = name or func.__name__
        
        # Get description from docstring if not provided
        tool_description = description or inspect.getdoc(func) or f"Function {tool_name}"
        
        # Log the parameter model
        logger.debug(f"Using parameter model: [cyan]{parameter_model.__name__}[/cyan]")
        
        # Generate optimized function schema using our utility
        function_def = generate_function_schema(func, parameter_model)
        
        # Extract schema
        schema = function_def["parameters"]
        
        # Log the parameter schema
        logger.debug(f"Parameter schema for tool [bold]{tool_name}[/bold]:")
        print_json(schema, title=f"{tool_name} Parameters")
        
        # Log tool docstring
        if inspect.getdoc(func):
            logger.debug(f"Tool docstring for [bold]{tool_name}[/bold]:")
            console.print(f"[dim italic]{inspect.getdoc(func)}[/dim italic]")
        
        # Store the tool info
        self.tools[tool_name] = {
            "function": func,
            "description": tool_description,
            "parameters_schema": schema,
            "parameter_model": parameter_model,
            "is_async": asyncio.iscoroutinefunction(func)
        }
        
        log_msg = (
            f"✅ Registered tool: [bold green]{tool_name}[/bold green] | "
            f"Parameters: [cyan]{len(schema.get('properties', {})) if 'properties' in schema else 0}[/cyan] | "
            f"{'[blue]async[/blue]' if asyncio.iscoroutinefunction(func) else '[yellow]sync[/yellow]'}"
        )
        logger.info(log_msg)
    
    def get_definitions(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions.
        
        Args:
            tool_names: List of tool names to include (None for all)
            
        Returns:
            List of tool definitions in OpenAI format
        """
        definitions = []
        
        # Determine which tools to include
        names = tool_names if tool_names is not None else self.tools.keys()
        
        if not names:
            logger.warning("No tool names provided and no tools registered")
            return []
            
        logger.info(f"Getting tool definitions for: [bold]{', '.join(names)}[/bold]")
        
        # Build the definitions
        for name in names:
            if name not in self.tools:
                logger.warning(f"⚠️ Tool not found: [bold red]{name}[/bold red]")
                continue
            
            tool_info = self.tools[name]
            
            definition = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters_schema"]
                }
            }
            
            logger.debug(f"Created definition for tool: [bold]{name}[/bold]")
            definitions.append(definition)
        
        logger.info(f"✅ Generated {len(definitions)} tool definitions")
        return definitions
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool with the given arguments.
        
        Supports both synchronous and asynchronous tool functions.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            ToolNotFoundError: If the tool is not found
            ToolExecutionError: If the tool execution fails
        """
        if tool_name not in self.tools:
            logger.error(f"❌ Tool not found: [bold red]{tool_name}[/bold red]")
            raise ToolNotFoundError(tool_name)
        
        tool_info = self.tools[tool_name]
        function = tool_info["function"]
        is_async = tool_info["is_async"]
        
        logger.info(f"🔧 Executing tool: [bold]{tool_name}[/bold]")
        logger.debug(f"Arguments for [bold]{tool_name}[/bold]:")
        print_json(arguments, title="Tool Arguments")
        
        try:
            # Validate arguments against parameter model
            try:
                param_model = tool_info["parameter_model"]
                logger.debug(f"Validating arguments against [cyan]{param_model.__name__}[/cyan]")
                validated_args = param_model(**arguments).model_dump()
                logger.debug("✅ Arguments validated successfully")
            except Exception as e:
                logger.error(f"❌ Invalid arguments for tool [bold]{tool_name}[/bold]: {str(e)}")
                raise ToolExecutionError(
                    tool_name=tool_name,
                    message=f"Invalid arguments: {str(e)}",
                    original_error=e
                )
            
            # Execute the function
            logger.debug(f"Executing {'async' if is_async else 'sync'} function for tool [bold]{tool_name}[/bold]")
            start_time = time.time()
            
            if is_async:
                # For async functions, await the result
                logger.debug(f"[bold green]Executing {tool_name}...[/bold green]")
                result = await function(**validated_args)
            else:
                # For sync functions, run in thread pool
                loop = asyncio.get_event_loop()
                logger.debug(f"[bold green]Executing {tool_name} (sync)...[/bold green]")
                result = await loop.run_in_executor(
                    _THREAD_POOL,
                    functools.partial(function, **validated_args)
                )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Tool [bold]{tool_name}[/bold] executed in [bold green]{elapsed_time:.4f}s[/bold green]")
            
            # Format the result
            formatted_result = self._format_result(result)
            
            # Log the result
            if isinstance(result, (dict, list)):
                print_json(result, title=f"{tool_name} Result")
            elif isinstance(result, str) and len(result) > 100:
                logger.debug(f"Result preview: {result[:100]}...")
            else:
                logger.debug(f"Result: {result}")
                
            return formatted_result
            
        except Exception as e:
            if isinstance(e, ToolExecutionError):
                raise
            
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            raise ToolExecutionError(
                tool_name=tool_name,
                message=str(e),
                original_error=e
            )
    
    def _format_result(self, result: Any) -> Any:
        """Format a tool execution result for inclusion in messages."""
        # If already a string, return as is
        if isinstance(result, str):
            return result
        
        # If a Pydantic model, convert to dict
        if isinstance(result, BaseModel):
            result = result.model_dump()
            logger.debug("Converted Pydantic model to dict")
        
        # Try to convert to JSON string for serializable objects
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Fall back to string representation
            logger.warning("Could not JSON serialize result, falling back to string representation")
            return str(result)