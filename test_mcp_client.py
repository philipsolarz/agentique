#!/usr/bin/env python3
"""
Test MCP client for AgentMCP bridge.
Tests the MCP server by calling tools and checking responses.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_list_agents():
    """Test listing agents through MCP."""
    print("\n🧪 Test 1: List Agents")
    print("=" * 60)

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "agentique"],
        env={
            "AGENTIQUE_AGENTS": "root=http://localhost:9000|calculator,data_processing,text_manipulation,info_retrieval"
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print(f"✓ Available tools: {[t.name for t in tools.tools]}")

            # Call a2a_list_agents
            result = await session.call_tool("a2a_list_agents", {})
            agents = json.loads(result.content[0].text)
            print(f"✓ Found {len(agents)} agent(s)")
            for agent in agents:
                print(f"  - {agent['name']}: skills={agent['skills']}")


async def test_send_message():
    """Test sending a message through MCP."""
    print("\n🧪 Test 2: Send Message (Calculator)")
    print("=" * 60)

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "agentique"],
        env={
            "AGENTIQUE_AGENTS": "root=http://localhost:9000|calculator,data_processing,text_manipulation,info_retrieval"
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Send a calculation request
            result = await session.call_tool("a2a_send", {
                "message": "What is 100 divided by 5?",
                "agent": "root"
            })

            response = json.loads(result.content[0].text)
            print(f"✓ Agent: {response['agent']}")
            print(f"✓ Response: {response['text']}")
            print(f"✓ Events: {len(response['events'])} event(s)")


async def test_send_by_skill():
    """Test sending a message by skill routing."""
    print("\n🧪 Test 3: Send Message by Skill (Text Manipulation)")
    print("=" * 60)

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "agentique"],
        env={
            "AGENTIQUE_AGENTS": "root=http://localhost:9000|calculator,data_processing,text_manipulation,info_retrieval"
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Send a text manipulation request
            result = await session.call_tool("a2a_send", {
                "message": "Convert hello world to title case",
                "skill": "text_manipulation"
            })

            response = json.loads(result.content[0].text)
            print(f"✓ Agent: {response['agent']}")
            print(f"✓ Response: {response['text']}")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  AgentMCP Bridge - MCP Client Tests")
    print("=" * 60)

    try:
        await test_list_agents()
        await test_send_message()
        await test_send_by_skill()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
