"""
Example of using Agentique for game agents.

This example demonstrates how to create custom structured output
models for game-specific events and interactions.
"""

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

# Game-specific event types
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

# Game-specific structured output model
# Example simple nested model
class GameMetadata(BaseModel):
    """Structured metadata for game events"""
    location: str = Field(..., description="Location where the event occurred")
    time: str = Field(..., description="Time when the event occurred")
    affected_entities: List[str] = Field(..., description="Entities affected by this event")

# Example event types as string literals for maximum compatibility
EVENT_TYPES = ["move", "attack", "defend", "interact", "speak", "use_item", "observe", "wait", "other"]

class GameEvent(StructuredResult):
    """
    Structured format for game events.
    
    This extends the base StructuredResult for game-specific functionality.
    """
    event_type: str = Field(..., 
        description="The type of game event (one of: move, attack, defend, interact, speak, use_item, observe, wait, other)")
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

# Game world simulation tools
async def get_world_state():
    """Get the current state of the game world."""
    return {
        "location": "forest",
        "time": "day",
        "weather": "clear",
        "nearby_entities": ["wolf", "tree", "stream"],
        "inventory": ["sword", "health_potion", "map"]
    }

async def get_entity_info(entity_name: str):
    """Get information about a specific entity in the game world."""
    entities = {
        "wolf": {
            "type": "enemy",
            "health": 50,
            "damage": 10,
            "description": "A gray wolf with gleaming yellow eyes",
            "hostile": True
        },
        "tree": {
            "type": "object",
            "description": "A tall oak tree with broad branches",
            "interactive": True,
            "actions": ["climb", "search"]
        },
        "stream": {
            "type": "environment",
            "description": "A clear flowing stream of water",
            "interactive": True,
            "actions": ["drink", "cross"]
        }
    }
    
    if entity_name in entities:
        return entities[entity_name]
    else:
        return {"error": f"Entity '{entity_name}' not found"}

async def main():
    # Get API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create Agentique instance
    agentique = Agentique(openai_api_key=openai_api_key)
    
    # Register game-specific tools
    agentique.register_tool(
        name="get_world_state",
        function=get_world_state,
        description="Get the current state of the game world"
    )
    
    agentique.register_tool(
        name="get_entity_info",
        function=get_entity_info,
        description="Get information about a specific entity in the game world"
    )
    
    # Create a game agent
    game_agent = agentique.create_agent(
        agent_id="game_character",
        system_prompt=(
            "You are an intelligent game character in a fantasy world. "
            "You make decisions based on the game state and player's instructions. "
            "Always respond with a structured GameEvent that describes your action. "
            "Think about your options carefully before choosing your actions."
        ),
        structured_output_model=GameEvent
    )
    
    # Run the agent with player instructions
    prompts = [
        "I want to explore the forest area around me",
        "I see a wolf. What should I do?",
        "Use my sword to attack the wolf"
    ]
    
    for prompt in prompts:
        print(f"\nPlayer: {prompt}")
        
        # For this example, let's use a fallback approach to avoid the strict schema issues
        try:
            result = await game_agent.run(
                user_input=prompt,
                tools=["get_world_state", "get_entity_info"]
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
            print(f"Error: {str(e)}")
            print("Falling back to non-structured output...")
            
            # Try again without structured output
            try:
                fallback_agent = agentique.create_agent(
                    agent_id=f"game_character_fallback_{prompt[:10]}",
                    system_prompt=(
                        "You are an intelligent game character in a fantasy world. "
                        "You make decisions based on the game state and player's instructions. "
                        "Respond as the character would, describing your actions and observations."
                    )
                )
                
                result = await fallback_agent.run(
                    user_input=prompt,
                    tools=["get_world_state", "get_entity_info"]
                )
                
                print(f"Response: {result}")
            except Exception as fallback_error:
                print(f"Fallback also failed: {str(fallback_error)}")

if __name__ == "__main__":
    asyncio.run(main())