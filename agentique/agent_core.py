"""
Core agent functionality for the Agentique library.

This module contains the Agent class which handles message history,
tool selection, and conversation management.

Design Patterns:
- Command Pattern: Tools are commands that can be executed dynamically
- Strategy Pattern: Different tools represent different strategies for solving problems
- Template Method: The run method defines a template for agent execution flow
"""

from typing import List, Dict, Any, Optional, Union, Callable, Type
import json
import copy
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .models import MessageModel, StructuredResult, ToolCall
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Maximum number of tool call iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 10

# Default maximum recursion depth for agent-to-agent calls
DEFAULT_MAX_RECURSION_DEPTH = 5

# Thread pool for executing synchronous functions in async context
_THREAD_POOL = ThreadPoolExecutor(max_workers=10)


class Agent:
    """
    Manages an AI agent, its state, and conversation logic.
    
    The Agent class is the central component for handling interactions with
    the LLM API, managing message history, and coordinating tool usage.
    """
    
    def __init__(
        self,
        agent_id: str,
        client,  # Will be an instance of BaseClientWrapper
        system_prompt: Optional[str] = None,
        tool_registry = None,  # Will be typed properly once implemented
        max_history_messages: int = 100,
        structured_output_model: Optional[Type[BaseModel]] = None,
    ):
        """
        Initialize an Agent instance.
        
        Args:
            agent_id: Unique identifier for this agent
            client: API client wrapper instance
            system_prompt: Base system prompt/persona for the agent
            tool_registry: Registry of tools available to the agent
            max_history_messages: Maximum number of messages to keep in history
            structured_output_model: Optional Pydantic model for structuring output
        """
        self.agent_id = agent_id
        self.client = client
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry
        self.message_history: List[MessageModel] = []
        self.max_history_messages = max_history_messages
        self.structured_output_model = structured_output_model or StructuredResult
        
        # Initialize with system message if provided
        if system_prompt:
            self.add_message("system", system_prompt)
    
    def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Any]] = None,
        tool_call_id: Optional[str] = None
    ) -> None:
        """
        Add a message to the agent's history.
        
        Args:
            role: Message role (system, user, assistant, or tool)
            content: Message content
            name: Name identifier (for tool responses)
            tool_calls: Tool calls initiated by the assistant
            tool_call_id: ID of the tool call this message is responding to
        """
        # Convert OpenAI API tool calls to our format if needed
        converted_tool_calls = None
        if tool_calls:
            converted_tool_calls = []
            for tc in tool_calls:
                # Check if it's already a dict or our model
                if isinstance(tc, dict):
                    converted_tool_calls.append(tc)
                elif isinstance(tc, ToolCall):
                    converted_tool_calls.append(tc)
                # Handle OpenAI API format
                elif hasattr(tc, 'id') and hasattr(tc, 'function'):
                    # Convert to our format
                    converted_tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                else:
                    # Unknown format, try to convert to dict
                    try:
                        if hasattr(tc, "model_dump"):
                            converted_tool_calls.append(tc.model_dump())
                        elif hasattr(tc, "__dict__"):
                            converted_tool_calls.append(tc.__dict__)
                        else:
                            logger.warning(f"Unknown tool call format: {type(tc)}. Trying to use as-is.")
                            converted_tool_calls.append(tc)
                    except Exception as e:
                        logger.error(f"Error converting tool call: {e}")
        
        message = MessageModel(
            role=role,
            content=content,
            name=name,
            tool_calls=converted_tool_calls,
            tool_call_id=tool_call_id
        )
        self.message_history.append(message)
        
        # Trim history if it exceeds the maximum length
        # Always keep the system message if present
        if len(self.message_history) > self.max_history_messages:
            # Check if the first message is a system message
            has_system = self.message_history and self.message_history[0].role == "system"
            
            # Calculate how many messages to remove
            to_remove = len(self.message_history) - self.max_history_messages
            
            if has_system:
                # Keep the system message and remove oldest messages after it
                system_message = self.message_history[0]
                self.message_history = [system_message] + self.message_history[to_remove+1:]
            else:
                # No system message, just remove oldest messages
                self.message_history = self.message_history[to_remove:]
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get formatted message history for API calls.
        
        Returns:
            List of message dictionaries in OpenAI format
        """
        return [msg.model_dump(exclude_none=True) for msg in self.message_history]
    
    def fork_history(self) -> List[MessageModel]:
        """
        Create a copy of the current message history.
        
        Used for recursive/inner calls to maintain context.
        
        Returns:
            A deep copy of the current message history
        """
        return copy.deepcopy(self.message_history)
    
    def reset(self, keep_system_prompt: bool = True) -> None:
        """
        Reset the agent's conversation history.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt message
        """
        if keep_system_prompt and self.message_history and self.message_history[0].role == "system":
            # Keep only the system message
            system_message = self.message_history[0]
            self.message_history = [system_message]
        else:
            # Clear all history
            self.message_history = []
            
            # Reinitialize with system prompt if available
            if keep_system_prompt and self.system_prompt:
                self.add_message("system", self.system_prompt)
    
    def estimate_token_count(self) -> int:
        """
        Estimate the number of tokens in the current message history.
        
        This is a rough estimation based on word count and is not exact.
        For precise counting, use a proper tokenizer.
        
        Returns:
            Estimated token count
        """
        # Very rough estimation: ~1.3 tokens per word
        token_count = 0
        
        for msg in self.message_history:
            # Count tokens in content
            if msg.content:
                # Rough estimate: split by space and count
                words = msg.content.split()
                token_count += int(len(words) * 1.3) + 5  # +5 for message metadata
            
            # Count tokens in tool calls (if any)
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Estimate tokens in function name and arguments
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    
                    token_count += len(function_name.split()) * 1.3
                    
                    # If arguments is a string, count it
                    if isinstance(arguments, str):
                        token_count += len(arguments.split()) * 1.3
                    # If it's a dict, convert to str first
                    elif isinstance(arguments, dict):
                        args_str = str(arguments)
                        token_count += len(args_str.split()) * 1.3
        
        return int(token_count)
    
    def trim_history_if_needed(self, max_tokens: int = 4000) -> None:
        """
        Trim the message history if it exceeds the specified token limit.
        
        This method keeps the system message and most recent messages,
        removing older messages to stay within the token limit.
        
        Args:
            max_tokens: Maximum number of tokens to allow
        """
        # Check current token count
        current_tokens = self.estimate_token_count()
        
        if current_tokens <= max_tokens:
            # Already within limit
            return
        
        # We need to trim messages
        # Always keep system message if present
        has_system = self.message_history and self.message_history[0].role == "system"
        
        # Start with all messages (or all except system)
        if has_system:
            system_message = self.message_history[0]
            messages_to_consider = self.message_history[1:]
        else:
            messages_to_consider = self.message_history.copy()
        
        # Remove oldest messages first until we're under the limit
        while messages_to_consider and self.estimate_token_count() > max_tokens:
            # Remove the oldest message
            oldest = messages_to_consider.pop(0)
            
            # Update history
            if has_system:
                self.message_history = [system_message] + messages_to_consider
            else:
                self.message_history = messages_to_consider
    
    def summarize_history(self, max_length: int = 200) -> str:
        """
        Create a summary of the conversation history.
        
        This can be used to create a condensed version of the history
        for context preservation in long conversations.
        
        Args:
            max_length: Maximum length of the summary
            
        Returns:
            A text summary of the conversation
        """
        if not self.message_history:
            return "No conversation history."
        
        # Start building the summary
        summary_parts = []
        
        # Process each message
        for msg in self.message_history:
            if msg.role == "system":
                # Skip system messages in the summary
                continue
                
            if msg.role == "user":
                prefix = "User asked: "
            elif msg.role == "assistant":
                prefix = "Assistant replied: "
            elif msg.role == "tool":
                prefix = f"Tool '{msg.name}' returned: "
            else:
                prefix = f"{msg.role.capitalize()}: "
            
            # Get content or tool calls
            if msg.content:
                # Truncate long content
                content = msg.content
                if len(content) > 50:
                    content = content[:47] + "..."
                summary_parts.append(f"{prefix}{content}")
            elif msg.tool_calls:
                tool_names = [tc.function.name for tc in msg.tool_calls]
                summary_parts.append(f"{prefix}Called tools: {', '.join(tool_names)}")
        
        # Join all parts and truncate if too long
        summary = "\n".join(summary_parts)
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
            
        return summary
    
    async def _call_api(
        self,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = "auto",
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> Any:
        """
        Call the LLM API with the current message history.
        
        Args:
            tools: Optional list of tool definitions
            tool_choice: Control tool choice behavior ("auto", "required", or None)
            response_format: Optional format specification for the response
            temperature: Sampling temperature for responses
            
        Returns:
            API response
        """
        try:
            # Get the formatted message history
            messages = self.get_messages()
            
            # For debugging, log the response format if present
            if response_format:
                schema = response_format.get("json_schema", {}).get("schema", {})
                logger.debug(f"Using response_format schema: {json.dumps(schema, indent=2)}")
            
            # Call the API with appropriate parameters
            response = await self.client.chat_completions(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                temperature=temperature
            )
            
            return response
        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Error calling LLM API: {str(e)}", exc_info=True)
            raise
    
    async def run(
        self,
        user_input: str,
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        call_depth: int = 0,
        max_depth: Optional[int] = None,
        temperature: float = 0.7
    ) -> Union[str, StructuredResult]:
        """
        Execute a single interaction turn.
        
        This method processes a user input, queries the LLM model, and returns
        the response. It maintains conversation history for context and
        executes tools as requested by the model in a loop until a response
        is provided.
        
        Args:
            user_input: The user's input message
            tools: Optional list of tool names available for this interaction
            system_prompt: Optional override for the system prompt
            call_depth: Current recursion depth (for agent-to-agent calls)
            max_depth: Maximum allowed recursion depth
            temperature: Sampling temperature for responses
            
        Returns:
            Either the assistant's text response or a structured output
        """
        # Check recursion depth to prevent infinite loops
        if max_depth is None:
            max_depth = DEFAULT_MAX_RECURSION_DEPTH
            
        if call_depth > max_depth:
            error_msg = f"Maximum recursion depth ({max_depth}) exceeded"
            logger.warning(error_msg)
            return error_msg
        
        # Add user message to history
        self.add_message("user", content=user_input)
        
        # Handle system prompt override
        original_system_message = None
        original_system_index = None
        
        if system_prompt and (self.system_prompt != system_prompt):
            # Find and temporarily update system message in history
            for i, msg in enumerate(self.message_history):
                if msg.role == "system":
                    original_system_message = msg.content
                    original_system_index = i
                    self.message_history[i] = MessageModel(role="system", content=system_prompt)
                    break
            
            # If no system message was found, add one at the beginning
            if original_system_message is None:
                system_msg = MessageModel(role="system", content=system_prompt)
                self.message_history.insert(0, system_msg)
        
        try:
            # If using OpenAI and structured output is desired, use response_format
            response_format = None
            use_native_structured_output = False
            is_openai = hasattr(self.client, 'model') and 'gpt' in self.client.model.lower()
            
            if self.structured_output_model and is_openai:
                # Check if the model supports structured outputs (gpt-4o and later)
                if 'gpt-4o' in self.client.model or 'gpt-4.5' in self.client.model or 'o1-' in self.client.model or 'o3-' in self.client.model:
                    response_format = self.structured_output_model.get_response_format()
                    use_native_structured_output = True
            
            # Get available tool definitions
            available_tools = []
            if tools and self.tool_registry:
                available_tools = self.tool_registry.get_tool_definitions(tools)
                
                # Make sure all tools have strict mode enabled
                for tool in available_tools:
                    if "function" in tool:
                        tool["function"]["strict"] = True
            
            # For structured output with OpenAI's supported models, use parse method
            if use_native_structured_output and not available_tools:
                try:
                    messages = self.get_messages()
                    result = await self.client.parse_structured_output(
                        messages=messages,
                        output_class=self.structured_output_model,
                        temperature=temperature
                    )
                    
                    # Check for refusal
                    if isinstance(result, dict) and "refusal" in result:
                        # Create assistant message with the refusal
                        self.add_message("assistant", content=result["refusal"])
                        return result["refusal"]
                    
                    # Add the assistant's structured output to history as a text message
                    if hasattr(result, "model_dump"):
                        content = json.dumps(result.model_dump(), indent=2)
                    else:
                        content = str(result)
                    
                    self.add_message("assistant", content=content)
                    return result
                except Exception as e:
                    logger.error(f"Error with structured output parsing: {e}. Falling back to regular flow.")
                    # Fall back to the regular flow
            
            # Start the conversation loop
            iteration_count = 0
            final_response = None
            
            while iteration_count < MAX_TOOL_ITERATIONS:
                iteration_count += 1
                
                # Call LLM API with appropriate format
                response = await self._call_api(
                    tools=available_tools,
                    response_format=response_format,
                    temperature=temperature
                )
                
                # Extract the assistant message from the response
                assistant_message = response.choices[0].message
                
                # Check for a refusal
                if hasattr(assistant_message, 'refusal') and assistant_message.refusal:
                    refusal_content = assistant_message.refusal
                    self.add_message("assistant", content=refusal_content)
                    return refusal_content
                
                # Check for tool calls
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    # Add the assistant's tool call message to history
                    self.add_message(
                        role="assistant",
                        tool_calls=assistant_message.tool_calls
                    )
                    
                    # Process each tool call
                    for tool_call in assistant_message.tool_calls:
                        tool_id = tool_call.id
                        function_call = tool_call.function
                        tool_name = function_call.name
                        
                        # Parse arguments
                        try:
                            arguments = json.loads(function_call.arguments)
                        except json.JSONDecodeError as e:
                            error_msg = f"Error parsing arguments for {tool_name}: {str(e)}"
                            logger.error(error_msg)
                            arguments = {}
                            tool_result = error_msg
                        else:
                            # Execute the tool and get result
                            try:
                                tool_result = await self.tool_registry.execute_tool(tool_name, arguments)
                            except Exception as e:
                                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                tool_result = {"error": error_msg, "details": str(e)}
                            
                            # Log the result for debugging
                            logger.debug(f"Tool {tool_name} result: {tool_result}")
                        
                        # Add tool result message to history
                        self.add_message(
                            role="tool",
                            name=tool_name,
                            content=tool_result if isinstance(tool_result, str) else json.dumps(tool_result),
                            tool_call_id=tool_id
                        )
                
                # Check for content (final answer)
                elif hasattr(assistant_message, 'content') and assistant_message.content:
                    # Add the assistant's response to history
                    self.add_message(
                        role="assistant",
                        content=assistant_message.content
                    )
                    
                    # Check if this is structured output in JSON format
                    content = assistant_message.content
                    if self.structured_output_model and content.strip().startswith('{') and content.strip().endswith('}'):
                        try:
                            # Try to parse as JSON and validate against the structured output model
                            data = json.loads(content)
                            return self.structured_output_model.model_validate(data)
                        except (json.JSONDecodeError, ValueError):
                            # If parsing fails, return the content as a string
                            pass
                    
                    # We have a final answer
                    final_response = content
                    break
                
                else:
                    # Unexpected response format
                    error_msg = "Unexpected response format from LLM API"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Check if we have a final response
            if final_response is not None:
                return final_response
            else:
                # We hit the iteration limit
                warning_msg = f"Reached maximum tool iteration limit ({MAX_TOOL_ITERATIONS})"
                logger.warning(warning_msg)
                return warning_msg
                
        finally:
            # Restore original system message if it was changed
            if original_system_message is not None and original_system_index is not None:
                self.message_history[original_system_index] = MessageModel(
                    role="system", 
                    content=original_system_message
                )
    
    async def continue_conversation(
        self, 
        tools: Optional[List[str]] = None,
        call_depth: int = 0,
        max_depth: Optional[int] = None,
        temperature: float = 0.7
    ) -> Union[str, StructuredResult]:
        """
        Continue the conversation after tool calls.
        
        Used when the agent needs to make multiple tool calls in sequence
        but no new user input is provided.
        
        Args:
            tools: Optional list of tool names available for this continuation
            call_depth: Current recursion depth (for agent-to-agent calls)
            max_depth: Maximum allowed recursion depth
            temperature: Sampling temperature for responses
            
        Returns:
            Next response from the model or a structured output
        """
        # Check recursion depth to prevent infinite loops
        if max_depth is None:
            max_depth = DEFAULT_MAX_RECURSION_DEPTH
            
        if call_depth > max_depth:
            error_msg = f"Maximum recursion depth ({max_depth}) exceeded"
            logger.warning(error_msg)
            return error_msg
        
        # Set up response format for structured output if appropriate
        response_format = None
        is_openai = hasattr(self.client, 'model') and 'gpt' in self.client.model.lower()
        
        if self.structured_output_model and is_openai:
            # Check if the model supports structured outputs (gpt-4o and later)
            if 'gpt-4o' in self.client.model or 'gpt-4.5' in self.client.model or 'o1-' in self.client.model or 'o3-' in self.client.model:
                response_format = self.structured_output_model.get_response_format()
        
        # Get available tool definitions
        available_tools = []
        if tools and self.tool_registry:
            available_tools = self.tool_registry.get_tool_definitions(tools)
            
            # Make sure all tools have strict mode enabled
            for tool in available_tools:
                if "function" in tool:
                    tool["function"]["strict"] = True
        
        # Start the continuation loop
        iteration_count = 0
        
        while iteration_count < MAX_TOOL_ITERATIONS:
            iteration_count += 1
            
            # Call LLM API with current history
            response = await self._call_api(
                tools=available_tools,
                response_format=response_format,
                temperature=temperature
            )
            
            # Extract the assistant message from the response
            assistant_message = response.choices[0].message
            
            # Check for a refusal
            if hasattr(assistant_message, 'refusal') and assistant_message.refusal:
                refusal_content = assistant_message.refusal
                self.add_message("assistant", content=refusal_content)
                return refusal_content
            
            # Check for tool calls
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                # Add the assistant's tool call message to history
                self.add_message(
                    role="assistant",
                    tool_calls=assistant_message.tool_calls
                )
                
                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_id = tool_call.id
                    function_call = tool_call.function
                    tool_name = function_call.name
                    
                    # Parse arguments
                    try:
                        arguments = json.loads(function_call.arguments)
                    except json.JSONDecodeError as e:
                        error_msg = f"Error parsing arguments for {tool_name}: {str(e)}"
                        logger.error(error_msg)
                        arguments = {}
                        tool_result = error_msg
                    else:
                        # Execute the tool and get result
                        try:
                            tool_result = await self.tool_registry.execute_tool(tool_name, arguments)
                        except Exception as e:
                            error_msg = f"Error executing tool {tool_name}: {str(e)}"
                            logger.error(error_msg)
                            tool_result = {"error": error_msg}
                    
                    # Add tool result message to history
                    self.add_message(
                        role="tool",
                        name=tool_name,
                        content=tool_result if isinstance(tool_result, str) else json.dumps(tool_result),
                        tool_call_id=tool_id
                    )
            
            # Check for content (final answer)
            elif hasattr(assistant_message, 'content') and assistant_message.content:
                # Add the assistant's response to history
                self.add_message(
                    role="assistant",
                    content=assistant_message.content
                )
                
                # Check if this is structured output in JSON format
                content = assistant_message.content
                if self.structured_output_model and content.strip().startswith('{') and content.strip().endswith('}'):
                    try:
                        # Try to parse as JSON and validate against the structured output model
                        data = json.loads(content)
                        return self.structured_output_model.model_validate(data)
                    except (json.JSONDecodeError, ValueError):
                        # If parsing fails, return the content as a string
                        pass
                
                # Return the content as the final answer
                return content
            
            else:
                # Unexpected response format
                error_msg = "Unexpected response format from LLM API"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # We hit the iteration limit
        warning_msg = f"Reached maximum tool iteration limit ({MAX_TOOL_ITERATIONS})"
        logger.warning(warning_msg)
        return warning_msg