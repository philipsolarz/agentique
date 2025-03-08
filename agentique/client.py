"""
OpenAI API client wrapper for the agentic game AI library.

This module provides a wrapper around the OpenAI Python client
to simplify interactions with the API.

Design Patterns:
- Adapter Pattern: Adapts the OpenAI API to our library's needs
- Retry Pattern: Implements retry logic for handling transient errors
"""

from typing import Dict, List, Any, Optional
import logging
import asyncio
import time
from openai import AsyncClient
from openai.types.chat import ChatCompletion
from openai import APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)


class OpenAIClientWrapper:
    """
    Wrapper for the OpenAI API client.
    
    Simplifies interactions with the OpenAI API, handling authentication
    and providing a streamlined interface for chat completions with tools.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the OpenAI client wrapper.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use for completions (default: gpt-4)
        """
        self.client = AsyncClient(api_key=api_key)
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
                
            except (APIError, APIConnectionError, RateLimitError) as e:
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
        if isinstance(error, RateLimitError):
            return True
        
        # Connection errors may be retryable
        if isinstance(error, APIConnectionError):
            return True
        
        # Some API errors may be retryable (server errors)
        if isinstance(error, APIError):
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