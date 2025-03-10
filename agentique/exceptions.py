"""
Custom exceptions for the Agentique library.

Provides specific error types for better error handling.
"""

class AgentiqueError(Exception):
    """Base exception class for all Agentique errors."""
    pass

class ToolExecutionError(AgentiqueError):
    """Raised when a tool execution fails."""
    
    def __init__(self, tool_name: str, message: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Error executing tool '{tool_name}': {message}")

class ToolNotFoundError(AgentiqueError):
    """Raised when a requested tool is not found."""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool not found: {tool_name}")

class APIError(AgentiqueError):
    """Raised when there's an error in the OpenAI API call."""
    
    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        super().__init__(message)

class ConfigurationError(AgentiqueError):
    """Raised when there's an invalid configuration."""
    pass

class ValidationError(AgentiqueError):
    """Raised when there's a validation error with inputs or outputs."""
    pass