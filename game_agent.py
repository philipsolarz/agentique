"""
Example of using the refactored Agentique library for game agents with rich debugging.

This example demonstrates how to create custom structured output
models for game-specific events and interactions, with enhanced
logging for better visibility into the agent's operations.
"""

import os
import asyncio
import json
from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from dotenv import load_dotenv

# Import from refactored library
from agentique import (
    Agent, OpenAIClient, ToolRegistry, StructuredOutput, AgentConfig,
    configure_logging, print_json, console
)

load_dotenv()

# Configure logging with rich output
logger = configure_logging(level="DEBUG", show_path=True)
console.rule("[bold green]Game Agent Example[/bold green]")

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

class WorldStateParams(BaseModel):
    pass
# Example event types as string literals for maximum compatibility
EVENT_TYPES = ["move", "attack", "defend", "interact", "speak", "use_item", "observe", "wait", "other"]

class GameEvent(StructuredOutput):
    """
    Structured format for game events.
    
    This extends the base StructuredOutput for game-specific functionality.
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

async def main():
    console.rule("[bold]Game Agent Initialization[/bold]")
    
    # Get API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        console.print("[bold red]ERROR: OPENAI_API_KEY environment variable not set[/bold red]")
        return
    
    # Create OpenAI client
    client = OpenAIClient(api_key=openai_api_key, model="gpt-4o-mini")
    
    # Create tool registry and register tools
    tools = ToolRegistry()
    
    tools.register(
        func=get_world_state,
        name="get_world_state",
        parameter_model=WorldStateParams,
        description="Get the current state of the game world"
    )
    
    tools.register(
        func=get_entity_info,
        name="get_entity_info",
        parameter_model=EntityInfoParams,
        description="Get information about a specific entity in the game world"
    )
    
    # Create agent configuration
    config = AgentConfig(
        name="game_character",
        model="gpt-4o-mini",
        system_prompt=(
            "You are an intelligent game character in a fantasy world. "
            "You make decisions based on the game state and player's instructions. "
            "Always respond with a structured GameEvent that describes your action. "
            "Think about your options carefully before choosing your actions."
        )
    )
    
    # Create game agent
    game_agent = Agent(
        config=config,
        client=client,
        tool_registry=tools
    )
    
    # Run the agent with player instructions
    prompts = [
        "I want to explore the forest area around me",
        "I see a wolf. What should I do?",
        "Use my sword to attack the wolf"
    ]
    
    for i, prompt in enumerate(prompts):
        console.rule(f"[bold yellow]Interaction {i+1}[/bold yellow]")
        console.print(f"[bold green]Player:[/bold green] {prompt}")
        
        try:
            # Use the run method with structured output
            result = await game_agent.run(
                user_input=prompt,
                tools=["get_world_state", "get_entity_info"],
                structured_output_model=GameEvent
            )
            
            # Display the structured output result in a nice format
            console.print("\n[bold blue]Game Event:[/bold blue]")
            console.print(f"[bold]Type:[/bold] {result.event_type}")
            console.print(f"[bold]Message:[/bold] {result.message}")
            console.print(f"[bold]Target:[/bold] {result.target}")
            console.print(f"[bold]Confidence:[/bold] {result.confidence:.2f}")
            console.print("[bold]Reasoning:[/bold]")
            console.print(f"[dim italic]{result.reasoning}[/dim italic]")
            
            # Print metadata in a nice format
            console.print("[bold]Metadata:[/bold]")
            print_json(result.metadata.model_dump(), title="Event Metadata")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}", highlight=True)
            import traceback
            console.print_exception()

if __name__ == "__main__":
    asyncio.run(main())