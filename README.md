# Agentique

Minimal core repository for the Agentique bridge.

## Included

- `src/` core implementation
- `docker-compose.yml` minimal runtime stack
- `docker/mcp/Dockerfile` container build for the MCP server
- `.env` and `.env.example` environment configuration

## Run

```bash
docker compose up --build
```

Configure agent endpoints in `AGENTIQUE_AGENTS` (for example in `.env`).
