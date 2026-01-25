"""A2A Server for the enhanced multi-agent test system.

This module exposes the test agent via the A2A protocol using the
google-adk's to_a2a utility, making it accessible to A2A clients.

Enhanced for testing AgentMCP features:
- Provider Architecture (MCP tool definitions in agent card)
- Background Tasks (SEP-1686)
- User Elicitation (Human-in-the-Loop)
- Sampling (Agentic LLM Workflows)
- Task State Machine Alignment
- Sub-Agent Visibility
- Tool Confirmation Flow
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from .agent import build_root_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default agent card path (relative to this file's package)
DEFAULT_AGENT_CARD = Path(__file__).parent.parent.parent / "agent_card.json"


def build_app() -> object:
    """Build the A2A Starlette application.

    Returns:
        A Starlette application configured for A2A communication.
    """
    # Get configuration from environment
    host = os.getenv("A2A_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_PORT", "9000"))
    protocol = os.getenv("A2A_PROTOCOL", "http")
    base_url = os.getenv("A2A_BASE_URL")

    # Build the agent
    logger.info("Building root agent...")
    root_agent = build_root_agent()
    logger.info(f"Root agent '{root_agent.name}' built successfully")

    # Determine agent card path
    # Priority: environment variable > default path
    agent_card_path = os.getenv("A2A_AGENT_CARD")
    if not agent_card_path and DEFAULT_AGENT_CARD.exists():
        agent_card_path = str(DEFAULT_AGENT_CARD)
        logger.info(f"Using default agent card: {agent_card_path}")

    # If base URL is provided, use it to construct the host and port
    # This is important for Docker environments where internal and external URLs differ
    if base_url:
        logger.info(f"Using base URL from environment: {base_url}")
        # Parse the base URL to extract host/port for agent card generation
        # Example: http://a2a-test-agent:9000 or http://localhost:9000
        import re
        match = re.match(r"(https?)://([^:]+):(\d+)", base_url)
        if match:
            protocol = match.group(1)
            host_for_card = match.group(2)
            port_for_card = int(match.group(3))
        else:
            host_for_card = host
            port_for_card = port
    else:
        host_for_card = host
        port_for_card = port

    # Build the A2A application
    logger.info(f"Creating A2A application at {protocol}://{host_for_card}:{port_for_card}")

    app = to_a2a(
        root_agent,
        host=host_for_card,
        port=port_for_card,
        protocol=protocol,
        agent_card=agent_card_path if agent_card_path else None,
    )

    logger.info("A2A application created successfully")
    return app


def main() -> None:
    """Run the A2A server using uvicorn."""
    host = os.getenv("A2A_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_PORT", "9000"))
    reload_enabled = os.getenv("A2A_RELOAD", "").lower() in {"1", "true", "yes"}

    logger.info(f"Starting A2A server on {host}:{port}")
    logger.info(f"Reload enabled: {reload_enabled}")

    app = build_app()

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload_enabled,
        log_level="info",
    )


if __name__ == "__main__":
    main()
