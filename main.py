import os
import asyncio
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Dict, Any
from agentique import Agentique, StructuredResult, configure_logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
configure_logging(level="INFO")

# Example of a domain-specific event type
class GameEventType(str, Enum):
    """Game-specific event types"""
    MOVE = "move"
    ATTACK = "attack"
    DEFEND = "defend"
    INTERACT = "interact"
    SPEAK = "speak"
    USE_ITEM = "use_item"
    OBSERVE = "observe"
    WAIT = "wait"
    OTHER = "other"

# Example of a domain-specific structured output model
class GameEvent(StructuredResult):
    """
    Structured format for game events.
    
    This extends the base StructuredResult for game-specific functionality.
    """
    event_type: GameEventType = Field(..., 
        description="The type of game event")
    message: str = Field(..., 
        description="Description of the event or response")
    target: Optional[str] = Field(None, 
        description="Target of the action (character, item, location)")
    confidence: float = Field(..., ge=0, le=1, 
        description="Confidence level (0-1)")
    reasoning: Optional[str] = Field(None, 
        description="Reasoning behind the decision")
    metadata: Dict[str, Any] = Field(default_factory=dict, 
        description="Additional metadata about the event")

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
    # Get API keys from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_api_key and not anthropic_api_key:
        raise ValueError("At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables must be set")
    
    # Create Agentique instance
    agentique = Agentique(
        openai_api_key=openai_api_key, 
        anthropic_api_key=anthropic_api_key
    )
    
    # Register a custom tool
    agentique.register_tool(
        name="get_weather",
        function=get_weather,
        description="Get current weather information for a location"
    )
    
    # Create an agent using OpenAI (default)
    openai_agent = agentique.create_agent(
        agent_id="weather_assistant",
        system_prompt="You are a helpful weather assistant. Help users get weather information.",
        structured_output_model=GameEvent  # Use our custom output model
    )
    
    # Run the agent with a query
    result = await openai_agent.run(
        user_input="What's the weather like in New York? Respond with a game event.",
        tools=["get_weather"]
    )
    
    # Handle the result
    if isinstance(result, GameEvent):
        print(f"Event Type: {result.event_type}")
        print(f"Message: {result.message}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")
        print(f"Target: {result.target}")
        print(f"Confidence: {result.confidence}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"Response: {result}")
    
    # Create an agent with Anthropic (if API key is available)
    if anthropic_api_key:
        anthropic_agent = agentique.create_agent(
            agent_id="claude_assistant",
            system_prompt="You are a helpful assistant. Answer questions concisely.",
            provider="anthropic",
            model="claude-3-sonnet-20240229"
        )
        
        # Run the Anthropic agent
        result = await anthropic_agent.run(
            user_input="Tell me a short joke about programming."
        )
        
        print("\nAnthropic Response:")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())