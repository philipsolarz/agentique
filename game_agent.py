"""
Example of using Agentique for game agents.

This example demonstrates how to create custom structured output
models for game-specific events and interactions, using the improved
OpenAI structured output parsing support.
"""

import os
import asyncio
from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Any, List, Optional
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
class GameMetadata(BaseModel):
    """Structured metadata for game events"""
    location: str = Field(..., description="Location where the event occurred")
    time: str = Field(..., description="Time when the event occurred")
    affected_entities: List[str] = Field(..., description="Entities affected by this event")

# Tool parameter models
class EntityInfoParams(BaseModel):
    """Parameters for the get_entity_info tool"""
    entity_name: str = Field(..., description="Name of the entity to get information about")

class Test(BaseModel):
    pass


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
import json

# Game world simulation tools
async def get_world_state() -> str:
    """Get the current state of the game world."""
    state = {
        "location": "forest",
        "time": "day",
        "weather": "clear",
        "nearby_entities": ["wolf", "tree", "stream"],
        "inventory": ["sword", "health_potion", "map"]
    }
    return json.dumps(state)

async def get_entity_info(entity_name: str) -> str:
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
        return json.dumps(entities[entity_name])
    else:
        return json.dumps({"error": f"Entity '{entity_name}' not found"})


# async def get_world_state() -> Dict[str, Any]:
#     """Get the current state of the game world."""
#     return {
#         "location": "forest",
#         "time": "day",
#         "weather": "clear",
#         "nearby_entities": ["wolf", "tree", "stream"],
#         "inventory": ["sword", "health_potion", "map"]
#     }

# async def get_entity_info(entity_name: str) -> Dict[str, Any]:
#     """Get information about a specific entity in the game world."""
#     entities = {
#         "wolf": {
#             "type": "enemy",
#             "health": 50,
#             "damage": 10,
#             "description": "A gray wolf with gleaming yellow eyes",
#             "hostile": True
#         },
#         "tree": {
#             "type": "object",
#             "description": "A tall oak tree with broad branches",
#             "interactive": True,
#             "actions": ["climb", "search"]
#         },
#         "stream": {
#             "type": "environment",
#             "description": "A clear flowing stream of water",
#             "interactive": True,
#             "actions": ["drink", "cross"]
#         }
#     }
    
#     if entity_name in entities:
#         return entities[entity_name]
#     else:
#         return {"error": f"Entity '{entity_name}' not found"}

async def main():
    # Get API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create Agentique instance
    agentique = Agentique(openai_api_key=openai_api_key)
    
    # Register game-specific tools with proper parameter models
    agentique.register_tool(
        name="get_world_state",
        function=get_world_state,
        # parameter_model=Test,
        description="Get the current state of the game world"
    )
    
    agentique.register_tool(
        name="get_entity_info",
        function=get_entity_info,
        parameter_model=EntityInfoParams,
        description="Get information about a specific entity in the game world"
    )
    
    # Create a game agent with a model that supports structured outputs
    game_agent = agentique.create_agent(
        agent_id="game_character",
        system_prompt=(
            "You are an intelligent game character in a fantasy world. "
            "You make decisions based on the game state and player's instructions. "
            "Always respond with a structured GameEvent that describes your action. "
            "Think about your options carefully before choosing your actions."
        ),
        model="gpt-4o-mini",  # Use a model that supports structured outputs
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
        
        try:
            # The agent.run method now properly handles structured outputs
            result = await game_agent.run(
                user_input=prompt,
                tools=["get_world_state", "get_entity_info"]
            )
            
            # Handle the result (which could be a GameEvent or string depending on whether tools were used)
            if isinstance(result, GameEvent):
                print(f"Event Type: {result.event_type}")
                print(f"Message: {result.message}")
                print(f"Reasoning: {result.reasoning}")
                print(f"Target: {result.target}")
                print(f"Confidence: {result.confidence}")
                print(f"Metadata: {result.metadata.model_dump()}")
            else:
                print(f"Response: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())