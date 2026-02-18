"""Enhanced multi-agent test system for AgentMCP validation.

This package provides a comprehensive test agent that validates all
AgentMCP features:

1. Provider Architecture - MCP tool definitions in agent card
2. Background Tasks (SEP-1686) - Long-running async tools
3. User Elicitation - Tools requiring user input
4. Sampling - Complex routing scenarios
5. Task State Machine - Tools producing different states
6. Sub-Agent Visibility - Branch tracking metadata
7. Tool Confirmation Flow - Dangerous operations requiring approval

Usage:
    # Run the server
    adk-test-agent

    # Or programmatically
    from adk_test_agent import build_root_agent, build_app
    agent = build_root_agent()
    app = build_app()
"""

from __future__ import annotations

from .agent import (
    build_root_agent,
    TASK_STATE_SUBMITTED,
    TASK_STATE_WORKING,
    TASK_STATE_INPUT_REQUIRED,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
    TASK_STATE_CANCELED,
)
from .cogito_agent import build_cogito_agent
from .server import build_app, main

__all__ = [
    # Agent builders
    "build_root_agent",
    "build_cogito_agent",
    # Server utilities
    "build_app",
    "main",
    # Task state constants (for testing)
    "TASK_STATE_SUBMITTED",
    "TASK_STATE_WORKING",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
]
