import os
import asyncio
from agentique import GameAI, FinalAnswer, configure_logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
configure_logging(level="INFO")

# Example tool function
async def get_weather(location: str):
    """Get weather information for a location."""
    # In a real implementation, this would call a weather API
    weather_data = {
        "New York": {"temperature": 72, "conditions": "sunny"},
        "London": {"temperature": 60, "conditions": "rainy"},
        "Tokyo": {"temperature": 80, "conditions": "clear"}
    }
    
    if location in weather_data:
        return weather_data[location]
    else:
        return {"error": f"Weather data for {location} not available"}

async def main():
    # Get OpenAI API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create GameAI instance
    game_ai = GameAI(api_key=api_key)
    
    # Register a custom tool
    game_ai.register_tool(
        name="get_weather",
        function=get_weather,
        description="Get current weather information for a location"
    )
    
    # Create an agent
    agent = game_ai.create_agent(
        agent_id="weather_assistant",
        system_prompt="You are a helpful weather assistant. Help users get weather information."
    )
    
    # Run the agent with a query
    result = await agent.run(
        user_input="What's the weather like in New York?",
        tools=["get_weather"]
    )
    
    # Handle the result
    if isinstance(result, FinalAnswer):
        print(f"Action: {result.action}")
        print(f"Message: {result.message}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")
    else:
        print(f"Response: {result}")

if __name__ == "__main__":
    asyncio.run(main())