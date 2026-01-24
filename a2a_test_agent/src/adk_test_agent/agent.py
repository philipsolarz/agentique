"""Enhanced multi-agent test system for AgentMCP validation.

This module implements a sophisticated test agent with multiple specialized subagents
to validate the MCP-A2A bridge functionality, including:
- Multi-agent routing and orchestration
- Various tool types (synchronous, async, data processing)
- Complex workflows and context preservation
- Different response patterns for testing streaming and incremental updates
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from datetime import datetime
from typing import Any


def build_root_agent() -> Any:
    """Build the root agent with multiple specialized subagents.

    Returns:
        A configured root Agent that orchestrates multiple subagents.
    """
    from google.adk import Agent

    model = os.getenv("ADK_MODEL", "gemini-2.0-flash")

    # ========== Calculator Agent ==========
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    def subtract(a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b

    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    def divide(a: float, b: float) -> str:
        """Divide a by b."""
        if b == 0:
            return "Error: Cannot divide by zero"
        return str(a / b)

    def calculate_statistics(numbers: list[float]) -> dict[str, float]:
        """Calculate statistics (mean, median, min, max) for a list of numbers."""
        if not numbers:
            return {"error": "Empty list provided"}

        sorted_nums = sorted(numbers)
        n = len(sorted_nums)

        return {
            "count": n,
            "sum": sum(sorted_nums),
            "mean": sum(sorted_nums) / n,
            "median": sorted_nums[n // 2] if n % 2 == 1 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2,
            "min": sorted_nums[0],
            "max": sorted_nums[-1],
        }

    calculator_agent = Agent(
        name="Calculator",
        description="Performs arithmetic operations and statistical calculations.",
        model=model,
        instruction="""
            You are a precise calculator agent. You can perform basic arithmetic operations
            (add, subtract, multiply, divide) and calculate statistics on lists of numbers.
            Always use the appropriate tool for the requested calculation.
            Present results clearly and accurately.
        """,
        tools=[add, subtract, multiply, divide, calculate_statistics],
    )

    # ========== Data Processing Agent ==========
    def filter_list(items: list[str], pattern: str) -> list[str]:
        """Filter a list of strings by a regex pattern."""
        try:
            regex = re.compile(pattern)
            return [item for item in items if regex.search(item)]
        except re.error:
            return [f"Error: Invalid regex pattern: {pattern}"]

    def sort_list(items: list[str], reverse: bool = False) -> list[str]:
        """Sort a list of strings alphabetically."""
        return sorted(items, reverse=reverse)

    def count_occurrences(items: list[str], target: str) -> dict[str, int]:
        """Count how many times each unique item appears in the list."""
        from collections import Counter
        counter = Counter(items)
        return {
            "target_count": counter.get(target, 0),
            "total_items": len(items),
            "unique_items": len(counter),
            "all_counts": dict(counter),
        }

    async def process_batch(items: list[str]) -> dict[str, Any]:
        """Process a batch of items with simulated async operations."""
        # Simulate async processing
        await asyncio.sleep(0.1)

        return {
            "processed": len(items),
            "timestamp": datetime.now().isoformat(),
            "preview": items[:3] if len(items) > 3 else items,
            "status": "completed",
        }

    data_agent = Agent(
        name="DataProcessor",
        description="Processes lists and data structures with filtering, sorting, and analysis.",
        model=model,
        instruction="""
            You are a data processing specialist. You can filter lists by patterns,
            sort data, count occurrences, and process batches of items.
            When users ask about data manipulation, use the appropriate tools
            to transform and analyze their data.
        """,
        tools=[filter_list, sort_list, count_occurrences, process_batch],
    )

    # ========== Text Manipulation Agent ==========
    def transform_case(text: str, style: str) -> str:
        """Transform text case. Styles: 'upper', 'lower', 'title', 'sentence'."""
        styles = {
            "upper": text.upper(),
            "lower": text.lower(),
            "title": text.title(),
            "sentence": text.capitalize(),
        }
        return styles.get(style.lower(), f"Unknown style: {style}")

    def count_words(text: str) -> dict[str, int]:
        """Count words, characters, and sentences in text."""
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])

        return {
            "words": len(words),
            "characters": len(text),
            "characters_no_spaces": len(text.replace(" ", "")),
            "sentences": sentences,
        }

    def extract_keywords(text: str, top_n: int = 5) -> list[str]:
        """Extract the most common words from text (simple keyword extraction)."""
        # Simple keyword extraction - remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
        words = [w.lower().strip('.,!?;:') for w in text.split()]
        words = [w for w in words if w and w not in common_words and len(w) > 2]

        from collections import Counter
        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]

    def reverse_text(text: str) -> str:
        """Reverse the text string."""
        return text[::-1]

    text_agent = Agent(
        name="TextProcessor",
        description="Manipulates and analyzes text with various transformations and analyses.",
        model=model,
        instruction="""
            You are a text processing specialist. You can transform text case,
            count words and characters, extract keywords, and reverse text.
            Help users analyze and transform their text in various ways.
        """,
        tools=[transform_case, count_words, extract_keywords, reverse_text],
    )

    # ========== Knowledge/Info Agent (simulated) ==========
    def search_info(query: str) -> dict[str, Any]:
        """Simulate searching for information (mock knowledge base)."""
        # Mock knowledge base
        knowledge_base = {
            "python": {
                "type": "programming_language",
                "description": "Python is a high-level, interpreted programming language.",
                "created": "1991",
                "creator": "Guido van Rossum",
            },
            "agents": {
                "type": "concept",
                "description": "AI agents are autonomous entities that can perceive and act on their environment.",
                "applications": ["automation", "decision-making", "assistance"],
            },
            "mcp": {
                "type": "protocol",
                "description": "Model Context Protocol enables communication between AI models and tools.",
                "features": ["tool_calling", "resource_access", "prompt_management"],
            },
        }

        query_lower = query.lower()
        for key, value in knowledge_base.items():
            if key in query_lower:
                return {
                    "query": query,
                    "found": True,
                    "result": value,
                }

        return {
            "query": query,
            "found": False,
            "message": f"No information found for '{query}'",
            "suggestions": list(knowledge_base.keys()),
        }

    def get_random_fact() -> str:
        """Get a random interesting fact."""
        facts = [
            "The first computer programmer was Ada Lovelace in the 1840s.",
            "Python was named after Monty Python, not the snake.",
            "The first computer bug was an actual moth found in a computer in 1947.",
            "AI agents can coordinate in multi-agent systems to solve complex problems.",
            "The MCP protocol enables seamless integration between AI models and external tools.",
        ]
        return random.choice(facts)

    async def fetch_data(source: str) -> dict[str, Any]:
        """Simulate fetching data from an external source."""
        # Simulate network delay
        await asyncio.sleep(0.2)

        return {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "status": "success",
                "records": random.randint(10, 100),
                "sample": f"Sample data from {source}",
            },
        }

    info_agent = Agent(
        name="InfoRetriever",
        description="Retrieves information, facts, and simulated data from various sources.",
        model=model,
        instruction="""
            You are an information retrieval specialist. You can search a knowledge base,
            provide random facts, and fetch data from sources.
            When users ask for information, use the appropriate tool to find and present it clearly.
        """,
        tools=[search_info, get_random_fact, fetch_data],
    )

    # ========== Root Orchestrator Agent ==========
    root_agent = Agent(
        name="TestAgentRoot",
        description="Multi-capability test agent that orchestrates specialized subagents for calculations, data processing, text manipulation, and information retrieval.",
        model=model,
        instruction="""
            You are a versatile orchestrator agent that coordinates multiple specialized subagents.

            Your subagents are:
            - Calculator: For arithmetic operations and statistical calculations
            - DataProcessor: For list/data manipulation, filtering, sorting, and batch processing
            - TextProcessor: For text transformations, analysis, and manipulation
            - InfoRetriever: For information lookup, facts, and data retrieval

            When a user makes a request:
            1. Analyze the request to understand what capability is needed
            2. Delegate to the appropriate subagent(s)
            3. If a request requires multiple capabilities, coordinate between agents
            4. Present results clearly to the user

            You can handle complex multi-step workflows by sequencing subagent calls.
            Always be helpful, accurate, and clear in your responses.
        """,
        sub_agents=[calculator_agent, data_agent, text_agent, info_agent],
    )

    return root_agent
