"""
Core Agent implementation for the Agentique library.

Provides a streamlined Agent class focused on message history management
and interaction with modern OpenAI models.
"""

import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Type, Union

from pydantic import BaseModel

from .models import Message, MessageRole, StructuredOutput, AgentConfig
from .openai_client import OpenAIClient
from .tools import ToolRegistry
from .exceptions import ToolExecutionError, ToolNotFoundError
from .logging import get_logger, print_json, console

logger = get_logger("agent")

class Agent:
    """
    Core agent class that manages state and handles interactions.
    
    Focused on maintaining message history and coordinating with
    the OpenAI API for completions and tool execution.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        client: OpenAIClient,
        tool_registry: Optional[ToolRegistry] = None
    ):
        """
        Initialize an Agent instance.
        
        Args:
            config: Configuration for the agent
            client: OpenAI client instance
            tool_registry: Optional registry of available tools
        """
        self.config = config
        self.client = client
        self.tool_registry = tool_registry or ToolRegistry()
        self.messages: List[Message] = []
        self.session_id = str(uuid.uuid4())[:8]  # Create a short session ID for logging
        
        logger.info(
            f"🤖 Agent initialized: [bold green]{config.name}[/bold green] | "
            f"Model: [blue]{config.model}[/blue] | "
            f"Session: [yellow]{self.session_id}[/yellow]"
        )
        
        # Initialize with system message if provided
        if config.system_prompt:
            logger.debug("Adding system prompt:")
            console.print(f"[dim italic]{config.system_prompt[:200]}{'...' if len(config.system_prompt) > 200 else ''}[/dim italic]")
            self.add_message(MessageRole.SYSTEM, config.system_prompt)
    
    def add_message(
        self,
        role: MessageRole,
        content: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None
    ) -> Message:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role (system, user, assistant, or tool)
            content: Message content
            name: Name identifier (for tool responses)
            tool_calls: Tool calls initiated by the assistant
            tool_call_id: ID of the tool call this message is responding to
            
        Returns:
            The created message object
        """
        message = Message(
            role=role,
            content=content,
            name=name,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id
        )
        
        self.messages.append(message)
        
        # Log the message addition
        role_colors = {
            MessageRole.SYSTEM: "magenta",
            MessageRole.USER: "green",
            MessageRole.ASSISTANT: "blue",
            MessageRole.TOOL: "yellow"
        }
        role_color = role_colors.get(role, "white")
        
        # Build message preview
        msg_preview = ""
        if content:
            if len(content) > 50:
                msg_preview = f": \"{content[:47]}...\""
            else:
                msg_preview = f": \"{content}\""
        elif tool_calls:
            tool_names = [tc.get('name', 'unknown') for tc in tool_calls]
            msg_preview = f": Calling {', '.join(tool_names)}"
            
        log_prefix = f"📝 Added [{role_color}]{role}[/{role_color}] message" 
        if name:
            log_prefix += f" (name: {name})"
        if tool_call_id:
            log_prefix += f" (tool_call_id: {tool_call_id[:6]}...)"
            
        logger.debug(f"{log_prefix}{msg_preview}")
        
        # Trim history if exceeds max length
        self._trim_history()
        
        return message
    
    def _trim_history(self) -> None:
        """
        Trim message history if it exceeds the configured maximum length.
        Always preserves the system message if present.
        """
        if len(self.messages) <= self.config.max_history:
            return
            
        # Check if first message is a system message
        has_system = self.messages and self.messages[0].role == MessageRole.SYSTEM
        
        # Calculate how many messages to remove
        to_remove = len(self.messages) - self.config.max_history
        
        logger.info(f"⚠️ Trimming message history: removing {to_remove} oldest messages")
        
        if has_system:
            # Keep system message and remove oldest messages after it
            system_message = self.messages[0]
            self.messages = [system_message] + self.messages[to_remove+1:]
            logger.debug("System message preserved during trimming")
        else:
            # No system message, just remove oldest messages
            self.messages = self.messages[to_remove:]
    
    def get_formatted_messages(self) -> List[Dict[str, Any]]:
        """
        Get messages formatted for OpenAI API.
        
        Returns:
            List of message dictionaries in OpenAI format
        """
        formatted = [msg.model_dump(exclude_none=True) for msg in self.messages]
        logger.debug(f"Formatted {len(formatted)} messages for API request")
        return formatted
    
    def reset(self, keep_system_prompt: bool = True) -> None:
        """
        Reset the conversation history.
        
        Args:
            keep_system_prompt: Whether to keep the system message
        """
        logger.info(f"🔄 Resetting agent conversation history (keep_system_prompt={keep_system_prompt})")
        
        if keep_system_prompt and self.messages and self.messages[0].role == MessageRole.SYSTEM:
            # Keep only the system message
            system_message = self.messages[0]
            self.messages = [system_message]
            logger.debug("Kept system message after reset")
        else:
            # Clear all messages
            self.messages = []
            logger.debug("Cleared all messages")
            
            # Reinitialize with system prompt if available
            if keep_system_prompt and self.config.system_prompt:
                logger.debug("Re-adding system prompt after reset")
                self.add_message(MessageRole.SYSTEM, self.config.system_prompt)
    
    async def run(
        self,
        user_input: str,
        tools: Optional[List[str]] = None,
        max_iterations: int = 5,
        structured_output_model: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None
    ) -> Union[str, BaseModel]:
        """
        Process user input and run the agent interaction.
        
        This streamlined version:
        1. Adds the user input to message history
        2. Calls the OpenAI API
        3. Executes any requested tools
        4. Returns the final text response or structured output
        
        Args:
            user_input: The user's input message
            tools: Optional list of tool names to make available
            max_iterations: Maximum number of tool call iterations
            structured_output_model: Optional model for structured output
            temperature: Temperature override (uses config value if None)
            
        Returns:
            Either a text response or a structured output object
        """
        run_id = str(uuid.uuid4())[:6]  # Short ID for this run
        logger.info(f"🚀 Starting agent run [bold cyan]{run_id}[/bold cyan] | Agent: [bold]{self.config.name}[/bold]")
        
        # Log user input
        if len(user_input) > 100:
            logger.info(f"User input: \"{user_input[:97]}...\"")
        else:
            logger.info(f"User input: \"{user_input}\"")
        
        # Add user message to history
        self.add_message(MessageRole.USER, content=user_input)
        
        # Use temperature from config if not specified
        temperature = temperature if temperature is not None else self.config.temperature
        logger.debug(f"Using temperature: {temperature}")
        
        # Get available tool definitions if tools are specified
        tool_definitions = None
        if tools and self.tool_registry:
            logger.info(f"Getting tool definitions for: [bold]{', '.join(tools)}[/bold]")
            tool_definitions = self.tool_registry.get_definitions(tools)
        
        start_time = time.time()
        
        # Handle structured output if requested
        if structured_output_model:
            logger.info(f"Running with structured output model: [bold blue]{structured_output_model.__name__}[/bold blue]")
            try:
                result = await self._run_with_structured_output(
                    structured_output_model,
                    tool_definitions,
                    max_iterations,
                    temperature
                )
                elapsed_time = time.time() - start_time
                logger.info(
                    f"✅ Run [cyan]{run_id}[/cyan] completed in [bold green]{elapsed_time:.2f}s[/bold green] | "
                    f"Result type: [bold blue]{type(result).__name__}[/bold blue]"
                )
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"❌ Run [cyan]{run_id}[/cyan] failed after [bold red]{elapsed_time:.2f}s[/bold red] | "
                    f"Error: {str(e)}"
                )
                raise
        
        # Regular conversation flow with possible tool calls
        iteration = 0
        logger.info(f"Running regular conversation flow (max {max_iterations} iterations)")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Iteration {iteration}/{max_iterations}")
            
            # Get formatted messages for API
            formatted_messages = self.get_formatted_messages()
            
            # Call OpenAI API
            with console.status("[bold green]Calling OpenAI API...[/bold green]"):
                response = await self.client.chat_completions(
                    messages=formatted_messages,
                    tools=tool_definitions,
                    temperature=temperature
                )
            
            # Extract assistant message
            assistant_message = response.choices[0].message
            
            # Check for tool calls
            if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
                tool_calls = assistant_message.tool_calls
                logger.info(f"🔧 Assistant requested {len(tool_calls)} tool call(s)")
                
                # Add assistant's tool call message to history
                self.add_message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": json.loads(tc.function.arguments)
                            },
                            "strict": "True"
                        }
                        for tc in tool_calls
                    ]
                )
                
                # Process each tool call
                for tool_call in tool_calls:
                    function_call = tool_call.function
                    tool_name = function_call.name
                    tool_id = tool_call.id
                    
                    logger.info(f"🔧 Processing tool call: [bold]{tool_name}[/bold] (ID: {tool_id[:8]}...)")
                    
                    try:
                        # Parse arguments
                        try:
                            arguments = json.loads(function_call.arguments)
                            logger.debug(f"Parsed arguments for {tool_name}:")
                            print_json(arguments, title=f"{tool_name} Arguments")
                        except json.JSONDecodeError as e:
                            error_msg = f"Error parsing arguments: {str(e)}"
                            logger.error(f"❌ {error_msg}")
                            result = error_msg
                            
                        # Execute the tool
                        if self.tool_registry and 'arguments' in locals():
                            logger.debug(f"Executing tool: {tool_name}")
                            result = await self.tool_registry.execute(tool_name, arguments)
                        else:
                            result = f"Tool {tool_name} not available (no tool registry)"
                            logger.warning(f"⚠️ {result}")
                            
                    except (ToolNotFoundError, ToolExecutionError) as e:
                        result = str(e)
                        logger.error(f"❌ Tool execution error: {result}")
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        logger.error(f"❌ Unexpected error during tool execution: {str(e)}", exc_info=True)
                        
                    # Add tool result to history
                    self.add_message(
                        role=MessageRole.TOOL,
                        name=tool_name,
                        content=result,
                        tool_call_id=tool_id
                    )
            else:
                # No tool calls, just a regular response
                content = assistant_message.content
                logger.info("Assistant provided a direct response (no tool calls)")
                
                # Add to history
                self.add_message(role=MessageRole.ASSISTANT, content=content)
                
                # Log completion
                elapsed_time = time.time() - start_time
                logger.info(
                    f"✅ Run [cyan]{run_id}[/cyan] completed in [bold green]{elapsed_time:.2f}s[/bold green] | "
                    f"Iterations: {iteration}/{max_iterations}"
                )
                
                # Log content preview
                if content:
                    preview = content[:100] + ("..." if len(content) > 100 else "")
                    logger.debug(f"Response preview: \"{preview}\"") 
                
                return content
        
        # If we reach here, we hit max iterations without a final response
        elapsed_time = time.time() - start_time
        warning_msg = f"Reached maximum iterations ({max_iterations}) without resolving the conversation."
        logger.warning(f"⚠️ {warning_msg} Run time: {elapsed_time:.2f}s")
        return warning_msg
    
    async def _run_with_structured_output(
        self,
        output_model: Type[BaseModel],
        tool_definitions: Optional[List[Dict[str, Any]]],
        max_iterations: int,
        temperature: float
    ) -> BaseModel:
        """
        Run the agent with structured output support.
        
        Similar to the main run method but returns a structured output
        in the specified format.
        
        Args:
            output_model: Pydantic model for the structured output
            tool_definitions: Available tool definitions
            max_iterations: Maximum number of tool call iterations
            temperature: Temperature for response sampling
            
        Returns:
            An instance of the specified output model
        """
        iteration = 0
        logger.info(f"Running with structured output: [bold blue]{output_model.__name__}[/bold blue]")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Iteration {iteration}/{max_iterations} (structured output mode)")
            
            # Get formatted messages
            formatted_messages = self.get_formatted_messages()
            
            # If we have tools and this isn't the final iteration, handle tool calls first
            if tool_definitions and iteration < max_iterations:
                logger.debug("Checking if tools are needed before generating structured output")
                
                # Call OpenAI API with tools
                with console.status("[bold green]Calling OpenAI API with tools...[/bold green]"):
                    response = await self.client.chat_completions(
                        messages=formatted_messages,
                        tools=tool_definitions,
                        temperature=temperature
                    )
                
                # Extract assistant message
                assistant_message = response.choices[0].message
                
                # Check for tool calls
                if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
                    tool_calls = assistant_message.tool_calls
                    logger.info(f"🔧 Assistant requested {len(tool_calls)} tool call(s) before generating structured output")
                    
                    # Process tool calls (same as in regular run)
                    self.add_message(
                        role=MessageRole.ASSISTANT,
                        tool_calls=[
                            {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": json.loads(tc.function.arguments)
                            }
                            for tc in tool_calls
                        ]
                    )
                    
                    # Process each tool call
                    for tool_call in tool_calls:
                        function_call = tool_call.function
                        tool_name = function_call.name
                        tool_id = tool_call.id
                        
                        logger.info(f"🔧 Processing tool call: [bold]{tool_name}[/bold] (ID: {tool_id[:8]}...)")
                        
                        try:
                            # Parse arguments
                            try:
                                arguments = json.loads(function_call.arguments)
                            except json.JSONDecodeError as e:
                                error_msg = f"Error parsing arguments: {str(e)}"
                                logger.error(f"❌ {error_msg}")
                                result = error_msg
                                
                            # Execute the tool
                            if self.tool_registry and 'arguments' in locals():
                                result = await self.tool_registry.execute(tool_name, arguments)
                            else:
                                result = f"Tool {tool_name} not available (no tool registry)"
                                logger.warning(f"⚠️ {result}")
                                
                        except (ToolNotFoundError, ToolExecutionError) as e:
                            result = str(e)
                            logger.error(f"❌ Tool execution error: {result}")
                        except Exception as e:
                            result = f"Error: {str(e)}"
                            logger.error(f"❌ Unexpected error during tool execution: {str(e)}", exc_info=True)
                            
                        # Add tool result to history
                        self.add_message(
                            role=MessageRole.TOOL,
                            name=tool_name,
                            content=result,
                            tool_call_id=tool_id
                        )
                    
                    # Continue to next iteration
                    continue
                else:
                    logger.debug("No tool calls needed, proceeding to structured output generation")
            
            # Either no tools, or final iteration, or no tool calls were made
            # Get structured output
            logger.info(f"Generating structured output using model: [bold blue]{output_model.__name__}[/bold blue]")
            
            with console.status("[bold green]Generating structured output...[/bold green]"):
                result = await self.client.get_structured_output(
                    messages=formatted_messages,
                    output_class=output_model,
                    temperature=temperature
                )
            
            # Add a simple content message to history for context
            content = json.dumps(result.model_dump(), ensure_ascii=False)
            self.add_message(role=MessageRole.ASSISTANT, content=content)
            
            logger.info(f"✅ Generated structured output: [bold blue]{output_model.__name__}[/bold blue]")
            return result
        
        # If we reach here, max iterations were exceeded without getting structured output
        logger.error(f"❌ Failed to get structured output after {max_iterations} iterations")
        raise RuntimeError(f"Failed to get structured output after {max_iterations} iterations")