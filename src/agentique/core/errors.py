"""Typed exception hierarchy for agentique.

Each exception carries ``mcp_code`` and optional ``a2a_code`` so that
callers can translate errors into the appropriate protocol error format.
"""

from __future__ import annotations


class AgentiqueError(Exception):
    """Base exception for all agentique errors."""

    mcp_code: int = -32603  # JSON-RPC internal error
    a2a_code: int | None = None


class AdapterError(AgentiqueError):
    """An adapter failed to communicate with its backend."""

    mcp_code: int = -32603


class AgentNotFoundError(AgentiqueError):
    """The requested agent does not exist in the registry."""

    mcp_code: int = -32602  # Invalid params
    a2a_code: int = -32001  # A2A TaskNotFound


class AgentUnavailableError(AgentiqueError):
    """The agent exists but is not currently reachable."""

    mcp_code: int = -32603


class TaskNotFoundError(AgentiqueError):
    """The requested task ID is unknown."""

    mcp_code: int = -32602
    a2a_code: int = -32001


class InputRequiredError(AgentiqueError):
    """The agent needs user input to proceed (A2A input-required state)."""

    mcp_code: int = -32603


class TranslationError(AgentiqueError):
    """Failed to translate between MCP and agent protocol types."""

    mcp_code: int = -32603
