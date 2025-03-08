"""
Custom exceptions for the Agentique library.

This module defines custom exception classes to provide more
specific error handling for the library.
"""


class AgenticError(Exception):
    """Base exception class for all library errors."""
    pass


class ToolExecutionError(AgenticError):
    """Raised when a tool execution fails."""
    
    def __init__(self, tool_name: str, message: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Error executing tool '{tool_name}': {message}")


class ToolNotFoundError(AgenticError):
    """Raised when a requested tool is not found."""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool not found: {tool_name}")


class AgentNotFoundError(AgenticError):
    """Raised when a requested agent is not found."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent not found: {agent_id}")


class APIError(AgenticError):
    """Raised when there's an error in the API call."""
    
    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        super().__init__(message)


class MaxRecursionError(AgenticError):
    """Raised when the maximum recursion depth is exceeded."""
    
    def __init__(self, depth: int):
        self.depth = depth
        super().__init__(f"Maximum recursion depth ({depth}) exceeded")


class InvalidArgumentError(AgenticError):
    """Raised when invalid arguments are provided."""
    pass


class MessageHistoryError(AgenticError):
    """Raised when there's an error with the message history."""
    pass