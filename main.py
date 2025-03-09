import os
import asyncio
from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Any, List
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
# Example simple nested model
class GameMetadata(BaseModel):
    """Structured metadata for game events"""
    location: str = Field(..., description="Location where the event occurred")
    time: str = Field(..., description="Time when the event occurred")
    affected_entities: List[str] = Field(..., description="Entities affected by this event")

class GameEvent(StructuredResult):
    """
    Structured format for game events.
    
    This extends the base StructuredResult for game-specific functionality.
    """
    event_type: GameEventType = Field(..., 
        description="The type of game event")
    message: str = Field(..., 
        description="Description of the event or response")
    target: str = Field(..., 
        description="Target of the action (character, item, location)")
    confidence: float = Field(...,
        description="Confidence level (0-1)")
    reasoning: str = Field(..., 
        description="Reasoning behind the decision")
    # Use a properly defined nested model instead of arbitrary Dict
    metadata: GameMetadata = Field(...,
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
        system_prompt="You are a helpful weather assistant. Help users get weather information. When responding, use a friendly and informative tone."
    )
    
    # Run the agent with a query for non-structured output
    print("Running agent with non-structured output:")
    
    try:
        result = await openai_agent.run(
            user_input="What's the weather like in New York?",
            tools=["get_weather"]
        )
        
        # Handle the result
        print(f"Response: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Try with structured output in a separate agent
    print("\nTrying with structured output (with fallback):")
    
    try:
        structured_agent = agentique.create_agent(
            agent_id="structured_weather_assistant",
            system_prompt=(
                "You are a helpful weather assistant. Help users get weather information. "
                "Format your response as a GameEvent with appropriate fields."
            ),
            structured_output_model=GameEvent
        )
        
        result = await structured_agent.run(
            user_input="What's the weather like in New York? Create a game event for this.",
            tools=["get_weather"]
        )
        
        # Handle the result
        if isinstance(result, GameEvent):
            print(f"Event Type: {result.event_type}")
            print(f"Message: {result.message}")
            print(f"Reasoning: {result.reasoning}")
            print(f"Target: {result.target}")
            print(f"Confidence: {result.confidence}")
            print(f"Metadata: {result.metadata}")
        else:
            print(f"Response: {result}")
    except Exception as e:
        print(f"Structured output error: {str(e)}")
        print("Using fallback plain text output instead.")
    
    # Try with Anthropic if API key is available
    if anthropic_api_key:
        print("\nAnthropic Agent Response:")
        try:
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
            
            print(result)
        except Exception as e:
            print(f"Anthropic error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())