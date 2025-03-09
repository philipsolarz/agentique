"""
Debug tool for testing OpenAI schema compatibility.

This script tests different schema variations to see what OpenAI accepts.
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, List
from openai import AsyncClient
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Create client
client = AsyncClient(api_key=api_key)

async def test_schema(schema: Dict[str, Any], name: str = "testschema") -> None:
    """
    Test if a schema is accepted by OpenAI.
    
    Args:
        schema: The schema to test
        name: Optional name for the schema
    """
    logger.info(f"Testing schema: {name}")
    logger.debug(f"Schema content: {json.dumps(schema, indent=2)}")
    
    try:
        # Set up chat messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Provide a test response in the required format."}
        ]
        
        # Call OpenAI with the schema
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": schema
                }
            }
        )
        
        # If we get here, the schema was accepted
        logger.info(f"✅ Schema {name} was ACCEPTED")
        content = response.choices[0].message.content
        logger.debug(f"Response content: {content}")
        return True
        
    except Exception as e:
        # If we get an error, the schema was rejected
        logger.error(f"❌ Schema {name} was REJECTED: {str(e)}")
        return False

async def main():
    """Run tests for different schema variations."""
    # Test 1: Simplest valid schema
    simple_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message": {"type": "string"}
        },
        "required": ["message"]
    }
    await test_schema(simple_schema, "simple_schema")
    
    # Test 2: Schema with enum
    enum_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_type": {
                "type": "string",
                "enum": ["move", "attack", "defend"]
            },
            "message": {"type": "string"}
        },
        "required": ["event_type", "message"]
    }
    await test_schema(enum_schema, "enum_schema")
    
    # Test 3: Nested object schema
    nested_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message": {"type": "string"},
            "metadata": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "location": {"type": "string"},
                    "time": {"type": "string"}
                },
                "required": ["location", "time"]
            }
        },
        "required": ["message", "metadata"]
    }
    await test_schema(nested_schema, "nested_schema")
    
    # Test 4: Full GameEvent schema (hardcoded)
    game_event_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_type": {
                "type": "string",
                "enum": ["move", "attack", "defend", "interact", "speak", 
                       "use_item", "observe", "wait", "other"],
                "description": "Type of game event"
            },
            "message": {
                "type": "string", 
                "description": "Description of the event"
            },
            "target": {
                "type": "string",
                "description": "Target of the action"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level"
            },
            "reasoning": {
                "type": "string",
                "description": "Reasoning behind the decision"
            },
            "metadata": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location where the event occurred"
                    },
                    "time": {
                        "type": "string",
                        "description": "Time when the event occurred"
                    },
                    "affected_entities": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Entities affected by this event"
                    }
                },
                "required": ["location", "time", "affected_entities"],
                "description": "Additional metadata about the event"
            }
        },
        "required": ["event_type", "message", "target", "confidence", "reasoning", "metadata"]
    }
    await test_schema(game_event_schema, "game_event_schema")

if __name__ == "__main__":
    asyncio.run(main())