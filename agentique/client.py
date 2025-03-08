"""
API client wrappers for the Agentique library.

This module provides wrappers around various LLM APIs
to simplify interactions with them.

Design Patterns:
- Adapter Pattern: Adapts the various APIs to our library's needs
- Retry Pattern: Implements retry logic for handling transient errors
"""

from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
import time
from abc import ABC, abstractmethod

# Import OpenAI client
from openai import AsyncClient as OpenAIAsyncClient
from openai.types.chat import ChatCompletion
from openai import APIError as OpenAIAPIError
from openai import APIConnectionError as OpenAIConnectionError
from openai import RateLimitError as OpenAIRateLimitError

# Import Anthropic client
from anthropic import AsyncAnthropic
from anthropic.types import Message
from anthropic import APIError as AnthropicAPIError
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import RateLimitError as AnthropicRateLimitError

logger = logging.getLogger(__name__)


class BaseClientWrapper(ABC):
    """
    Abstract base class for API client wrappers.
    
    Defines the common interface that all client wrappers must implement.
    """
    
    @abstractmethod
    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = "auto",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_base_delay: float = 1.0
    ) -> Any:
        """
        Send a chat completion request to the API.
        
        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            tool_choice: Control tool choice behavior
            temperature: Sampling temperature
            max_retries: Maximum number of retries on transient errors
            retry_base_delay: Base delay for exponential backoff (in seconds)
            
        Returns:
            API response with chat completion
        """
        pass


class OpenAIClientWrapper(BaseClientWrapper):
    """
    Wrapper for the OpenAI API client.
    
    Simplifies interactions with the OpenAI API, handling authentication
    and providing a streamlined interface for chat completions with tools.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI client wrapper.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use for completions (default: gpt-4o-mini)
        """
        self.client = OpenAIAsyncClient(api_key=api_key)
        self.model = model
    
    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = "auto",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_base_delay: float = 1.0
    ) -> ChatCompletion:
        """
        Async wrapper for OpenAI chat completions API with robust error handling.
        
        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            tool_choice: Control tool choice behavior ("auto", "required", or None)
            temperature: Sampling temperature
            max_retries: Maximum number of retries on transient errors
            retry_base_delay: Base delay for exponential backoff (in seconds)
            
        Returns:
            OpenAI API response (ChatCompletion object)
            
        Raises:
            Exception: If API call fails after all retries
        """
        retry_count = 0
        last_exception = None
        
        while retry_count <= max_retries:
            try:
                # Log attempt for debugging
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count}/{max_retries} for OpenAI API call")
                
                # Prepare API call parameters
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                # Add tools if provided
                if tools:
                    params["tools"] = tools
                    
                    if tool_choice:
                        params["tool_choice"] = tool_choice
                
                # Log the request (excluding message content for brevity)
                logger.debug(f"Calling OpenAI API with model={self.model}, "
                           f"message_count={len(messages)}, tool_count={len(tools) if tools else 0}")
                
                # Call the OpenAI API
                start_time = time.time()
                response = await self.client.chat.completions.create(**params)
                elapsed_time = time.time() - start_time
                
                # Log success
                logger.debug(f"OpenAI API call completed in {elapsed_time:.2f}s")
                
                return response
                
            except (OpenAIAPIError, OpenAIConnectionError, OpenAIRateLimitError) as e:
                retry_count += 1
                last_exception = e
                
                # Check if we should retry
                if retry_count <= max_retries and self._is_retryable_error(e):
                    # Calculate delay with exponential backoff and jitter
                    wait_time = retry_base_delay * (2 ** (retry_count - 1))
                    jitter = wait_time * 0.1 * (asyncio.get_event_loop().time() % 1.0)
                    wait_time += jitter
                    
                    logger.warning(f"OpenAI API error: {str(e)}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API error after {retry_count} retries: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {str(e)}", exc_info=True)
                last_exception = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = retry_base_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Unexpected error: {str(e)}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        # If we get here, we've exhausted retries
        raise last_exception or RuntimeError("Failed to get response from OpenAI API")
    
    def _is_retryable_error(self, error) -> bool:
        """
        Determine if an OpenAI error is retryable.
        
        Args:
            error: The OpenAI error
            
        Returns:
            True if the error is retryable, False otherwise
        """
        # Rate limit errors are always retryable
        if isinstance(error, OpenAIRateLimitError):
            return True
        
        # Connection errors may be retryable
        if isinstance(error, OpenAIConnectionError):
            return True
        
        # Some API errors may be retryable (server errors)
        if isinstance(error, OpenAIAPIError):
            # Server errors (5xx) are retryable
            if hasattr(error, 'status_code') and str(error.status_code).startswith('5'):
                return True
        
        # For other kinds of errors, check the error message
        error_message = str(error).lower()
        
        # Common retryable error keywords
        retryable_keywords = [
            "rate limit",
            "timeout",
            "server error",
            "service unavailable",
            "too many requests",
            "capacity"
        ]
        
        for keyword in retryable_keywords:
            if keyword in error_message:
                return True
        
        return False


class AnthropicClientWrapper(BaseClientWrapper):
    """
    Wrapper for the Anthropic API client.
    
    Simplifies interactions with the Anthropic API, handling authentication
    and providing a streamlined interface for chat completions.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize the Anthropic client wrapper.
        
        Args:
            api_key: Anthropic API key
            model: Model name to use for completions (default: claude-3-sonnet-20240229)
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = "auto",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_base_delay: float = 1.0
    ) -> Message:
        """
        Async wrapper for Anthropic chat completions API with robust error handling.
        
        Args:
            messages: List of conversation messages (in OpenAI format)
            tools: Optional list of tool definitions
            tool_choice: Control tool choice behavior (not used in Anthropic)
            temperature: Sampling temperature
            max_retries: Maximum number of retries on transient errors
            retry_base_delay: Base delay for exponential backoff (in seconds)
            
        Returns:
            Anthropic API response (Message object)
            
        Raises:
            Exception: If API call fails after all retries
        """
        retry_count = 0
        last_exception = None
        
        # Convert OpenAI format messages to Anthropic format
        anthropic_messages = self._convert_to_anthropic_format(messages)
        
        while retry_count <= max_retries:
            try:
                # Log attempt for debugging
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count}/{max_retries} for Anthropic API call")
                
                # Prepare API call parameters
                params = {
                    "model": self.model,
                    "messages": anthropic_messages,
                    "temperature": temperature,
                }
                
                # Add tools if provided
                if tools:
                    params["tools"] = self._convert_tools_to_anthropic_format(tools)
                
                # Log the request
                logger.debug(f"Calling Anthropic API with model={self.model}, "
                           f"message_count={len(anthropic_messages)}")
                
                # Call the Anthropic API
                start_time = time.time()
                response = await self.client.messages.create(**params)
                elapsed_time = time.time() - start_time
                
                # Log success
                logger.debug(f"Anthropic API call completed in {elapsed_time:.2f}s")
                
                # Convert response to a format similar to OpenAI for compatibility
                return self._convert_to_openai_like_response(response)
                
            except (AnthropicAPIError, AnthropicConnectionError, AnthropicRateLimitError) as e:
                retry_count += 1
                last_exception = e
                
                # Check if we should retry
                if retry_count <= max_retries and self._is_retryable_error(e):
                    # Calculate delay with exponential backoff and jitter
                    wait_time = retry_base_delay * (2 ** (retry_count - 1))
                    jitter = wait_time * 0.1 * (asyncio.get_event_loop().time() % 1.0)
                    wait_time += jitter
                    
                    logger.warning(f"Anthropic API error: {str(e)}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Anthropic API error after {retry_count} retries: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error calling Anthropic API: {str(e)}", exc_info=True)
                last_exception = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = retry_base_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Unexpected error: {str(e)}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        # If we get here, we've exhausted retries
        raise last_exception or RuntimeError("Failed to get response from Anthropic API")
    
    def _convert_to_anthropic_format(self, openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format messages to Anthropic format.
        
        Args:
            openai_messages: Messages in OpenAI format
            
        Returns:
            Messages in Anthropic format
        """
        anthropic_messages = []
        
        # Extract system message (if any)
        system_content = None
        for msg in openai_messages:
            if msg["role"] == "system":
                system_content = msg["content"]
                break
        
        # Process other messages
        for msg in openai_messages:
            if msg["role"] == "system":
                # System messages are handled separately in Anthropic
                continue
                
            if msg["role"] == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant":
                assistant_msg = {"role": "assistant"}
                
                if "content" in msg and msg["content"]:
                    assistant_msg["content"] = msg["content"]
                
                # Handle tool calls (note: this is simplified, might need expansion)
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Convert to Anthropic's tool_use format if needed
                    # This is a simplified implementation - the actual conversion 
                    # would depend on Anthropic's latest API format
                    assistant_msg["content"] = msg["content"] or "I'll use a tool to help with this."
                    # Note: Anthropic has a different format for tool calls which
                    # would need to be implemented based on their API docs
                
                anthropic_messages.append(assistant_msg)
            elif msg["role"] == "tool":
                # Anthropic handles tool responses differently
                # This is a simplified conversion
                # Add a user message with the tool response for now
                tool_name = msg.get("name", "tool")
                anthropic_messages.append({
                    "role": "user", 
                    "content": f"Tool {tool_name} returned: {msg['content']}"
                })
        
        # Add system message as the first message if present
        if system_content:
            # In Anthropic API, system prompt is a separate parameter
            # For now, we'll add it as a hidden "system" message
            # This would need to be extracted later
            anthropic_messages.insert(0, {
                "role": "system",
                "content": system_content
            })
        
        return anthropic_messages
    
    def _convert_tools_to_anthropic_format(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Anthropic format.
        
        Args:
            openai_tools: Tools in OpenAI format
            
        Returns:
            Tools in Anthropic format
        """
        # Note: This is a placeholder implementation
        # Actual implementation would depend on Anthropic's tool format
        anthropic_tools = []
        
        for tool in openai_tools:
            if tool["type"] == "function":
                anthropic_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"]
                })
        
        return anthropic_tools
    
    def _convert_to_openai_like_response(self, anthropic_response: Any) -> Any:
        """
        Convert Anthropic response to a format similar to OpenAI's.
        
        Args:
            anthropic_response: The response from Anthropic API
            
        Returns:
            Response in a format similar to OpenAI's
        """
        # Create a class to mimic OpenAI's response structure
        class OpenAILikeResponse:
            def __init__(self, anthropic_response):
                self.id = anthropic_response.id
                
                # Create a message object
                class Message:
                    def __init__(self, content):
                        self.content = content
                        self.tool_calls = [] # Anthropic tools would need conversion
                
                # Create a choice object
                class Choice:
                    def __init__(self, message):
                        self.message = message
                        self.index = 0
                
                # Extract the message content
                content = anthropic_response.content[0].text
                
                # Create the choice
                message = Message(content)
                
                # Set up the structure
                self.choices = [Choice(message)]
        
        # Return the converted response
        return OpenAILikeResponse(anthropic_response)
    
    def _is_retryable_error(self, error) -> bool:
        """
        Determine if an Anthropic error is retryable.
        
        Args:
            error: The Anthropic error
            
        Returns:
            True if the error is retryable, False otherwise
        """
        # Rate limit errors are always retryable
        if isinstance(error, AnthropicRateLimitError):
            return True
        
        # Connection errors may be retryable
        if isinstance(error, AnthropicConnectionError):
            return True
        
        # Some API errors may be retryable (server errors)
        if isinstance(error, AnthropicAPIError):
            # Server errors (5xx) are retryable
            if hasattr(error, 'status_code') and str(error.status_code).startswith('5'):
                return True
        
        # For other kinds of errors, check the error message
        error_message = str(error).lower()
        
        # Common retryable error keywords
        retryable_keywords = [
            "rate limit",
            "timeout",
            "server error",
            "service unavailable",
            "too many requests",
            "capacity"
        ]
        
        for keyword in retryable_keywords:
            if keyword in error_message:
                return True
        
        return False