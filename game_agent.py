"""
Example of using Agentique for game agents.

This example demonstrates how to create custom structured output
models for game-specific events and interactions.
"""

import os
import asyncio
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Dict, Any, List
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
        
        result = await game_agent.run(
            user_input=prompt,
            tools=["get_world_state", "get_entity_info"]
        )
        
        # Handle the result
        if isinstance(result, GameEvent):
            print(f"Event Type: {result.event_type}")
            print(f"Message: {result.message}")
            if result.reasoning:
                print(f"Reasoning: {result.reasoning}")
            if result.target:
                print(f"Target: {result.target}")
            print(f"Confidence: {result.confidence}")
            if result.metadata:
                print(f"Metadata: {result.metadata}")
        else:
            print(f"Response: {result}")

if __name__ == "__main__":
    asyncio.run(main())