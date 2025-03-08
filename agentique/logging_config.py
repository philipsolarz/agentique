"""
Logging configuration for the agentic game AI library.

This module provides functions to set up logging for the library,
making it easier to debug and monitor the library's behavior.
"""

import logging
import sys
from typing import Optional, Union, TextIO


def configure_logging(
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None,
    output: Optional[Union[str, TextIO]] = None
) -> None:
    """
    Configure logging for the agentic game AI library.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
        output: Output file path or file-like object (default: stderr)
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create handler
    if output is None:
        # Default to stderr
        handler = logging.StreamHandler(sys.stderr)
    elif isinstance(output, str):
        # File path provided
        handler = logging.FileHandler(output)
    else:
        # File-like object provided
        handler = logging.StreamHandler(output)
    
    # Configure handler
    handler.setFormatter(formatter)
    
    # Configure the library's logger
    logger = logging.getLogger("agentic_game_ai")
    logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicates
    for existing_handler in logger.handlers[:]:
        logger.removeHandler(existing_handler)
    
    # Add the new handler
    logger.addHandler(handler)
    
    # Set propagation to False to prevent duplicate logs
    logger.propagate = False
    
    logger.debug("Logging configured for agentic_game_ai")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component of the library.
    
    Args:
        name: Name of the component (e.g., "agent", "tools")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"agentic_game_ai.{name}")