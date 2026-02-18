"""Entry point for the Simulator MCP Server: python -m agentique.simulation.mcp_server"""

import argparse
import logging
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env file from project root if it exists."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        env_file = current / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip()
                        if key and key not in os.environ:
                            os.environ[key] = value
            return
        parent = current.parent
        if parent == current:
            break
        current = parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentique Simulator MCP Server — controls simulated conversations via MCP",
    )
    parser.add_argument(
        "--simulation-ui-url",
        default=None,
        help="URL of the Simulation UI (e.g., http://localhost:8080). "
        "If not set, runs in headless mode.",
    )
    parser.add_argument(
        "--mcp-url",
        default="http://localhost:8000/mcp",
        help="MCP server URL for headless mode (default: http://localhost:8000/mcp)",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port for SSE transport (default: 8081)",
    )
    args = parser.parse_args()

    _load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from .server import create_simulator_mcp_server

    server = create_simulator_mcp_server(
        simulation_ui_url=args.simulation_ui_url,
        mcp_url=args.mcp_url,
    )

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
