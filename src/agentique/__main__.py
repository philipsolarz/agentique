"""CLI entry point for agentique.

Parses agent configuration from the ``AGENTIQUE_AGENTS`` environment
variable and starts the FastMCP server.
"""

from __future__ import annotations

import logging

from .core.config import AgentiqueConfig
from .core.types import AgentInfo
from .server import create_server

logger = logging.getLogger(__name__)


def _parse_agents(value: str | None) -> list[AgentInfo]:
    """Parse agent config from env string.

    Format: ``name=url|skill1,skill2;name2=url2|skill3``
    """
    if not value:
        return []
    agents: list[AgentInfo] = []
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name_url, *skill_parts = entry.split("|")
        if "=" not in name_url:
            raise ValueError(
                "Agent entry must be: name=base_url[|skill1,skill2]"
            )
        name, base_url = name_url.split("=", 1)
        skills: tuple[str, ...] = ()
        if skill_parts and skill_parts[0].strip():
            skills = tuple(
                s.strip() for s in skill_parts[0].split(",") if s.strip()
            )
        agents.append(AgentInfo(
            name=name.strip(), base_url=base_url.strip(), skills=skills,
        ))
    return agents


def main() -> None:
    """Parse config and run the server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = AgentiqueConfig()
    agents = _parse_agents(config.agents)

    if not agents:
        logger.warning("No agents configured. Set AGENTIQUE_AGENTS.")

    server = create_server(agents=agents, config=config)

    transport = config.transport
    # Normalize "http" to "streamable-http" for FastMCP
    if transport == "http":
        transport = "streamable-http"

    logger.info(
        "Starting agentique on %s:%d (transport=%s, agents=%d)",
        config.host, config.port, transport, len(agents),
    )

    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=transport, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
