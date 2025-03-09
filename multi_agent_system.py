"""
Example of a multi-agent system built with Agentique.

This example demonstrates how to create multiple agents that can
communicate with each other to solve a complex problem.
"""

import os
import asyncio
from pydantic import BaseModel, Field
from typing import List
from agentique import Agentique, StructuredResult, configure_logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
configure_logging(level="INFO")

# Example of domain-specific output models
class ResearchResult(StructuredResult):
    """Research findings from the researcher agent"""
    topic: str = Field(..., description="The researched topic")
    facts: List[str] = Field(..., description="Key facts discovered")
    sources: List[str] = Field(..., description="Sources of information")
    follow_up_questions: List[str] = Field(..., description="Suggested follow-up questions")

class AnalysisResult(StructuredResult):
    """Analysis from the analyst agent"""
    summary: str = Field(..., description="Summary of the analysis")
    key_points: List[str] = Field(..., description="Key points from the analysis")
    implications: List[str] = Field(..., description="Implications of the findings")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the analysis")

class ArticleOutline(StructuredResult):
    """Article outline from the writer agent"""
    title: str = Field(..., description="Article title")
    introduction: str = Field(..., description="Introduction paragraph")
    sections: List[str] = Field(..., description="Main section topics")
    conclusion: str = Field(..., description="Conclusion paragraph idea")
    target_audience: str = Field(..., description="Intended audience for the article")

async def research_topic(topic: str):
    """Simulate researching a topic by providing some predefined information"""
    topics = {
        "artificial intelligence": {
            "facts": [
                "AI research began in the 1950s",
                "Machine learning is a subset of AI focusing on learning from data",
                "Deep learning has revolutionized AI since the 2010s",
                "Transformer models like GPT have advanced natural language processing"
            ],
            "sources": ["AI: A Modern Approach", "Deep Learning by Goodfellow et al."]
        },
        "climate change": {
            "facts": [
                "Global temperatures have risen by about 1°C since pre-industrial times",
                "CO2 levels are higher than at any point in the last 800,000 years",
                "Sea levels are rising at an accelerating rate",
                "Extreme weather events are becoming more frequent"
            ],
            "sources": ["IPCC Reports", "NASA Climate Research"]
        }
    }
    
    # Default response for topics not in our database
    result = {
        "facts": [f"Information about {topic} would be researched here",
                 "This is a simulation of a research tool"],
        "sources": ["Example Source 1", "Example Source 2"]
    }
    
    # Return predefined info if available
    return topics.get(topic.lower(), result)

async def main():
    # Get API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create Agentique instance
    agentique = Agentique(openai_api_key=openai_api_key)
    
    # Register tools
    agentique.register_tool(
        name="research_topic",
        function=research_topic,
        description="Research a topic and return facts and sources"
    )
    
    # Create coordinator agent without structured output
    coordinator = agentique.create_agent(
        agent_id="coordinator",
        system_prompt=(
            "You are a coordination agent responsible for managing a research and writing project. "
            "You'll help the user create content about their requested topic. "
            "Provide a step-by-step plan for how you would approach the topic."
        )
    )
    
    # Run the multi-agent system
    print("Starting multi-agent workflow...\n")
    
    # Get user input
    user_topic = "artificial intelligence"
    user_request = f"Create an article about {user_topic}."
    print(f"User Request: {user_request}\n")
    
    try:
        # Run the coordinator agent without message_agent tool
        coordinator_response = await coordinator.run(
            user_input=user_request
        )
        
        print(f"Coordinator's Response:\n{coordinator_response}\n")
        
        # Now demonstrate using research_topic tool directly
        print(f"Research on {user_topic}:")
        
        research_agent = agentique.create_agent(
            agent_id="researcher",
            system_prompt="You are a research specialist. Provide detailed information about topics."
        )
        
        research_result = await research_agent.run(
            user_input=f"Research the topic: {user_topic}",
            tools=["research_topic"]
        )
        
        print(f"Research Result:\n{research_result}\n")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())