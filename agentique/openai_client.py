"""
OpenAI API client for the Agentique library.

Provides a streamlined interface to OpenAI's API with focus on
modern models (GPT-4o and newer) and structured outputs.
"""

import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Type, Union, TypeVar

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai import APIError, APIConnectionError, RateLimitError
from pydantic import BaseModel

from .models import Message
from .exceptions import APIError as AgentiqueAPIError
from .logging import get_logger, print_json, console

logger = get_logger("openai")

T = TypeVar('T', bound=BaseModel)

class OpenAIClient:
    """
    Streamlined client for OpenAI's API focused on modern models.
    
    Provides both synchronous and asynchronous interfaces for chat completions
    and structured output parsing.
    """
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_base_delay: float = 1.0
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (must be gpt-4o-mini or newer)
            max_retries: Maximum number of retries for transient errors
            retry_base_delay: Base delay for exponential backoff (seconds)
        """
        self.sync_client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        
        logger.info(f"Initialized OpenAIClient with model: [bold blue]{model}[/bold blue]")
        
        # Validate model is a supported version
        supported_prefixes = ['gpt-4o', 'gpt-4-', 'gpt-4.', 'o1-']
        if not any(model.startswith(prefix) for prefix in supported_prefixes):
            raise ValueError(f"Model {model} is not supported. Must be GPT-4o or newer.")
    
    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, Any]] = None
    ) -> ChatCompletion:
        """
        Send a chat completion request to OpenAI with retry logic.
        
        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            temperature: Sampling temperature
            response_format: Optional format for structured response
            
        Returns:
            OpenAI API response
        """
        retry_count = 0
        last_exception = None
        
        while retry_count <= self.max_retries:
            try:
                # Log attempt if retrying
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count}/{self.max_retries}")
                
                # Prepare API call parameters
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                # Add tools if provided
                if tools:
                    params["tools"] = tools
                    # Log tool definitions for debugging
                    logger.debug("Tool schemas for OpenAI request:")
                    print_json(tools, title="Tool Definitions")
                
                # Add response_format if provided
                if response_format:
                    params["response_format"] = response_format
                    logger.debug("Response format:")
                    print_json(response_format, title="Response Format")
                
                # Log request payload summary
                message_count = len(messages)
                logger.info(
                    f"Sending request to OpenAI API: "
                    f"[bold]{self.model}[/bold] | "
                    f"[cyan]{message_count} messages[/cyan] | "
                    f"[yellow]temp={temperature}[/yellow] | "
                    f"[green]tools={len(tools) if tools else 0}[/green]"
                )
                
                # Log the last user message for context
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    last_user_msg = user_messages[-1].get("content", "")
                    if len(last_user_msg) > 100:
                        last_user_msg = last_user_msg[:97] + "..."
                    logger.debug(f"Last user message: [italic]\"{last_user_msg}\"[/italic]")
                
                # Call the OpenAI API with timing
                start_time = time.time()
                logger.debug("⏳ Making API request...")
                logger.debug("⏳ Waiting for OpenAI response...")
                response = await self.async_client.chat.completions.create(**params)
                logger.debug("✅ Response received")
                elapsed_time = time.time() - start_time
                
                # Log success with timing
                logger.info(f"✅ OpenAI API response received in [bold green]{elapsed_time:.2f}s[/bold green]")
                
                # Log response summary
                try:
                    content = response.choices[0].message.content or ""
                    has_tool_calls = hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls
                    
                    if has_tool_calls:
                        tool_names = [tc.function.name for tc in response.choices[0].message.tool_calls]
                        logger.debug(f"Response includes tool calls: [bold yellow]{', '.join(tool_names)}[/bold yellow]")
                    elif content:
                        preview = content[:100] + ("..." if len(content) > 100 else "")
                        logger.debug(f"Response content: [italic]\"{preview}\"[/italic]")
                except Exception as e:
                    logger.warning(f"Could not summarize response: {str(e)}")
                
                return response
                
            except (APIError, APIConnectionError, RateLimitError) as e:
                retry_count += 1
                last_exception = e
                
                # Check if we should retry
                if retry_count <= self.max_retries and self._is_retryable_error(e):
                    # Calculate delay with exponential backoff and jitter
                    wait_time = self.retry_base_delay * (2 ** (retry_count - 1))
                    jitter = wait_time * 0.1 * (asyncio.get_event_loop().time() % 1.0)
                    wait_time += jitter
                    
                    logger.warning(f"⚠️ OpenAI API error: {str(e)}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ OpenAI API error after {retry_count} retries: {str(e)}")
                    raise AgentiqueAPIError(str(e), getattr(e, 'status_code', None))
            except Exception as e:
                logger.error(f"❌ Unexpected error calling OpenAI API: {str(e)}", exc_info=True)
                retry_count += 1
                last_exception = e
                
                if retry_count <= self.max_retries:
                    wait_time = self.retry_base_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Unexpected error: {str(e)}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise AgentiqueAPIError(str(e))
        
        # If we get here, we've exhausted retries
        raise last_exception or AgentiqueAPIError("Failed to get response from OpenAI API")
    
    async def get_structured_output(
        self,
        messages: List[Dict[str, Any]],
        output_class: Type[T],
        temperature: float = 0.7
    ) -> T:
        """
        Get a structured output response based on a Pydantic model.
        
        This method uses OpenAI's native structured output parsing.
        
        Args:
            messages: List of conversation messages
            output_class: Pydantic model class for the output
            temperature: Sampling temperature
            
        Returns:
            Parsed structured output as an instance of output_class
        """
        try:
            # Get the schema for the model
            schema = output_class.model_json_schema()
            logger.info(f"Requesting structured output using model: [bold]{output_class.__name__}[/bold]")
            logger.debug("Structured output schema:")
            print_json(schema, title=f"{output_class.__name__} Schema")
            
            # Create the response format
            response_format = {
                "type": "json_object",
                "schema": schema
            }
            
            # Call the API with JSON response format
            response = await self.chat_completions(
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
            
            # Parse the JSON content
            content = response.choices[0].message.content
            if not content:
                raise AgentiqueAPIError("Empty response from OpenAI API")
            
            logger.debug("Parsing JSON response into structured output")
            data = json.loads(content)
            print_json(data, title="Raw Structured Output")
            
            result = output_class.model_validate(data)
            logger.info(f"✅ Successfully parsed response into [bold]{output_class.__name__}[/bold]")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON response: {str(e)}")
            raise AgentiqueAPIError(f"Invalid JSON response: {str(e)}")
    
    def _is_retryable_error(self, error) -> bool:
        """Determine if an error is retryable."""
        # Rate limit errors are always retryable
        if isinstance(error, RateLimitError):
            return True
        
        # Connection errors are often retryable
        if isinstance(error, APIConnectionError):
            return True
        
        # Some API errors may be retryable (server errors)
        if isinstance(error, APIError):
            # Server errors (5xx) are retryable
            if hasattr(error, 'status_code') and str(error.status_code).startswith('5'):
                return True
        
        # Check for retryable keywords in error message
        error_message = str(error).lower()
        retryable_keywords = [
            "rate limit",
            "timeout",
            "server error",
            "service unavailable",
            "too many requests",
            "capacity"
        ]
        
        return any(keyword in error_message for keyword in retryable_keywords)