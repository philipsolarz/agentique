from __future__ import annotations

import os
from typing import Iterable

from .models import AgentDescriptor
from .server import create_server


def _parse_agents(value: str | None) -> list[AgentDescriptor]:
    if not value:
        return []
    agents: list[AgentDescriptor] = []
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name_and_url, *skill_parts = entry.split("|")
        if "=" not in name_and_url:
            raise ValueError(
                "Agent entry must be formatted as name=base_url[|skill1,skill2]"
            )
        name, base_url = name_and_url.split("=", 1)
        skills: Iterable[str] = ()
        if skill_parts and skill_parts[0].strip():
            skills = tuple(skill.strip() for skill in skill_parts[0].split(",") if skill.strip())
        agents.append(AgentDescriptor(name=name.strip(), base_url=base_url.strip(), skills=tuple(skills)))
    return agents


def main() -> None:
    agents = _parse_agents(os.getenv("AGENTIQUE_AGENTS"))
    server = create_server(agents=agents)

    transport = os.getenv("AGENTIQUE_TRANSPORT", "stdio")
    host = os.getenv("AGENTIQUE_HOST", "127.0.0.1")
    port_str = os.getenv("AGENTIQUE_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError("AGENTIQUE_PORT must be an integer") from exc

    if transport == "http":
        server.run(transport=transport, host=host, port=port)
    else:
        server.run(transport=transport)


if __name__ == "__main__":
    main()
