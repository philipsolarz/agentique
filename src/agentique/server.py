from __future__ import annotations

from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.lifespan import lifespan

from .bridge import RouterBridge
from .models import AgentDescriptor
from .router import AgentRouter


def create_server(
    *,
    name: str = "Agentique Bridge",
    agents: list[AgentDescriptor] | None = None,
    router: AgentRouter | None = None,
    client_factory: Any | None = None,
) -> FastMCP:
    """Create a FastMCP server that routes requests to A2A agents."""

    active_router = router or AgentRouter(agents or [])
    bridge = RouterBridge(active_router, client_factory=client_factory)

    @lifespan
    async def bridge_lifespan(server: FastMCP):
        yield {"router": active_router}
        await bridge.aclose()

    mcp = FastMCP(name, lifespan=bridge_lifespan)

    @mcp.tool
    async def a2a_send(
        message: str,
        agent: str | None = None,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
        ctx: Context = CurrentContext(),
    ) -> dict[str, Any]:
        """Route a message to an A2A agent and return its response.

        Uses streaming internally to keep the connection alive during long-running
        operations, while still returning the final aggregated result.
        """

        await ctx.info("Routing message to A2A agent.")

        # Use streaming internally to avoid timeouts on long-running operations
        # Collect all chunks and send progress updates
        chunks = []
        async for chunk in bridge.stream(
            message,
            ctx=ctx,
            agent=agent,
            skill=skill,
            metadata=metadata,
            configuration=configuration,
        ):
            chunks.append(chunk)
            # Send progress updates to keep connection alive
            if chunk.kind in {"status", "task"} and chunk.text:
                await ctx.info(f"Agent working: {chunk.text}")

        # Aggregate the final response from all chunks
        if not chunks:
            return {"agent": agent or "unknown", "text": "", "events": []}

        # Get the last chunk's agent name and aggregate all text
        final_agent = chunks[-1].agent if chunks else (agent or "unknown")
        all_text = " ".join(c.text for c in chunks if c.text).strip()

        await ctx.report_progress(100, 100, "A2A response received")
        return {
            "agent": final_agent,
            "text": all_text,
            "events": [c.to_dict() for c in chunks],
        }

    @mcp.tool
    async def a2a_stream(
        message: str,
        agent: str | None = None,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
        ctx: Context = CurrentContext(),
    ):
        """Stream a response from an A2A agent as chunks."""

        await ctx.info("Starting A2A streaming response.")
        async for chunk in bridge.stream(
            message,
            ctx=ctx,
            agent=agent,
            skill=skill,
            metadata=metadata,
            configuration=configuration,
        ):
            if chunk.kind in {"status", "task"} and chunk.text:
                await ctx.info(f"A2A update: {chunk.text}")
            yield chunk.to_dict()
        await ctx.report_progress(100, 100, "A2A streaming completed")

    @mcp.tool
    def a2a_list_agents() -> list[dict[str, Any]]:
        """List known A2A agents."""

        return [agent.to_dict() for agent in active_router.list_agents()]

    @mcp.resource("a2a://agents")
    async def a2a_agents_resource(ctx: Context = CurrentContext()) -> dict[str, Any]:
        """Provide agent catalog data to MCP clients."""

        return {
            "agents": [agent.to_dict() for agent in active_router.list_agents()],
            "session_id": getattr(ctx, "session_id", None),
        }

    @mcp.resource("a2a://agents/{agent}")
    async def a2a_agent_resource(agent: str, ctx: Context = CurrentContext()) -> dict[str, Any]:
        descriptor = active_router.describe(agent)
        return {
            "agent": descriptor.to_dict(),
            "session_id": getattr(ctx, "session_id", None),
        }

    @mcp.resource("a2a://agents/{agent}/card")
    async def a2a_agent_card_resource(agent: str) -> dict[str, Any]:
        card = await bridge.get_agent_card(agent)
        if card is None:
            return {"agent": agent, "card": None}
        if hasattr(card, "model_dump"):
            return {"agent": agent, "card": card.model_dump()}
        if hasattr(card, "dict"):
            return {"agent": agent, "card": card.dict()}
        return {"agent": agent, "card": card}

    @mcp.prompt
    async def a2a_routing_prompt(goal: str, ctx: Context = CurrentContext()) -> str:
        agent_lines = []
        for agent in active_router.list_agents():
            skills = ", ".join(agent.skills) if agent.skills else "(no skills listed)"
            agent_lines.append(f"- {agent.name}: {skills}")
        agent_block = "\n".join(agent_lines) if agent_lines else "- No agents registered"
        session = getattr(ctx, "session_id", None)
        return (
            "You are an MCP client that can route work to A2A agents.\n"
            f"Goal: {goal}\n\n"
            "Available agents:\n"
            f"{agent_block}\n\n"
            "Pick the best agent and call `a2a_send` with the goal as the message.\n"
            f"Session: {session}\n"
        )

    return mcp
