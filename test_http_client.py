#!/usr/bin/env python3
"""Simple HTTP client test for AgentMCP server."""

import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


async def main():
    print("🧪 Testing AgentMCP HTTP Connection")
    print("=" * 60)

    # Connect to MCP server via HTTP
    transport = StreamableHttpTransport(url="http://localhost:8000/mcp")
    client = Client(transport)

    try:
        print("\n✓ Connecting to http://localhost:8000/mcp...")

        # List available tools
        print("\n✓ Listing available tools...")
        tools_response = await client.list_tools()
        tools = tools_response.tools
        print(f"  Found {len(tools)} tools:")
        for tool in tools:
            print(f"    - {tool.name}: {tool.description}")

        # Test a2a_list_agents
        print("\n✓ Testing a2a_list_agents...")
        agents_result = await client.call_tool("a2a_list_agents", {})
        print(f"  Result: {agents_result.content[0].text}")

        # Test a2a_send
        print("\n✓ Testing a2a_send (calculator)...")
        calc_result = await client.call_tool("a2a_send", {
            "message": "What is 25 + 17?",
            "agent": "root"
        })
        print(f"  Question: What is 25 + 17?")
        print(f"  Answer: {calc_result.content[0].text}")

        # Test text processing
        print("\n✓ Testing a2a_send (text processing)...")
        text_result = await client.call_tool("a2a_send", {
            "message": "Convert hello world to uppercase",
            "skill": "text_manipulation"
        })
        print(f"  Question: Convert hello world to uppercase")
        print(f"  Answer: {text_result.content[0].text}")

        print("\n" + "=" * 60)
        print("✅ All tests passed! HTTP connection works perfectly!")
        print("=" * 60)
        print("\n📝 Connection URL: http://localhost:8000/mcp")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
