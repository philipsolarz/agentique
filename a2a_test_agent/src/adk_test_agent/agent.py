"""Enhanced multi-agent test system for AgentMCP validation.

This module implements a sophisticated test agent with multiple specialized subagents
to validate the MCP-A2A bridge functionality, including:
- Multi-agent routing and orchestration
- Various tool types (synchronous, async, data processing)
- Complex workflows and context preservation
- Different response patterns for testing streaming and incremental updates

NEW: Enhanced for testing all 7 AgentMCP features:
1. Provider Architecture - MCP tool definitions in agent card
2. Background Tasks (SEP-1686) - Long-running async tools
3. User Elicitation - Tools requiring user input
4. Sampling - Complex routing scenarios
5. Task State Machine - Tools producing different states
6. Sub-Agent Visibility - Branch tracking metadata
7. Tool Confirmation Flow - Dangerous operations requiring approval
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from datetime import datetime
from typing import Any


# =============================================================================
# Task State Constants (for state machine testing)
# =============================================================================

TASK_STATE_SUBMITTED = "submitted"
TASK_STATE_WORKING = "working"
TASK_STATE_INPUT_REQUIRED = "input-required"
TASK_STATE_COMPLETED = "completed"
TASK_STATE_FAILED = "failed"
TASK_STATE_CANCELED = "canceled"


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

    # ========== Interactive Agent (for Elicitation & Confirmation testing) ==========
    def request_user_preference(
        question: str,
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        """Request a preference from the user. This simulates elicitation.

        The MCP client should intercept this and prompt the user for input.

        Args:
            question: The question to ask the user
            options: Optional list of choices to present

        Returns:
            A response indicating user input is required
        """
        return {
            "status": TASK_STATE_INPUT_REQUIRED,
            "requires_input": True,
            "question": question,
            "options": options or [],
            "message": f"User input required: {question}",
        }

    def confirm_action(
        action: str,
        details: str | None = None,
    ) -> dict[str, Any]:
        """Request confirmation before proceeding with an action.

        This simulates the tool confirmation flow where the MCP client
        should prompt the user to approve the action before execution.

        Args:
            action: The action that requires confirmation
            details: Additional details about what will happen

        Returns:
            A response indicating confirmation is required
        """
        return {
            "status": TASK_STATE_INPUT_REQUIRED,
            "requires_confirmation": True,
            "action": action,
            "details": details,
            "message": f"Confirmation required for: {action}",
        }

    def dangerous_operation(
        operation: str,
        target: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Perform a dangerous operation that SHOULD require confirmation.

        Operations like 'delete', 'modify', 'execute' on sensitive targets
        should trigger the tool confirmation flow in the MCP client.

        Args:
            operation: The operation type (delete, modify, execute, etc.)
            target: The target of the operation
            force: Skip confirmation (for testing bypass scenarios)

        Returns:
            Result of the operation
        """
        # This metadata signals to the bridge that confirmation is needed
        result = {
            "operation": operation,
            "target": target,
            "requires_confirmation": not force,
            "dangerous": True,
        }

        if force:
            result["status"] = "executed"
            result["message"] = f"Force-executed {operation} on {target}"
        else:
            result["status"] = TASK_STATE_INPUT_REQUIRED
            result["message"] = f"Awaiting confirmation to {operation} on {target}"

        return result

    def interactive_wizard(
        task: str,
        step: int = 1,
    ) -> dict[str, Any]:
        """Run an interactive wizard that requires multiple user inputs.

        This tests the multi-step elicitation flow where the agent
        needs to gather several pieces of information from the user.

        Args:
            task: The task to configure
            step: Current step in the wizard (1-3)

        Returns:
            Current step info or completion status
        """
        steps = {
            1: {
                "status": TASK_STATE_INPUT_REQUIRED,
                "step": 1,
                "total_steps": 3,
                "question": f"What type of {task} do you want to create?",
                "options": ["simple", "advanced", "custom"],
                "next_action": "Call interactive_wizard with step=2",
            },
            2: {
                "status": TASK_STATE_INPUT_REQUIRED,
                "step": 2,
                "total_steps": 3,
                "question": f"Enter a name for your {task}:",
                "options": None,
                "next_action": "Call interactive_wizard with step=3",
            },
            3: {
                "status": TASK_STATE_INPUT_REQUIRED,
                "step": 3,
                "total_steps": 3,
                "question": f"Confirm creation of {task}?",
                "options": ["yes", "no"],
                "requires_confirmation": True,
                "next_action": "Complete wizard",
            },
        }

        if step > 3:
            return {
                "status": TASK_STATE_COMPLETED,
                "message": f"Wizard completed! {task} has been created.",
                "step": "complete",
            }

        return steps.get(step, steps[1])

    interactive_agent = Agent(
        name="Interactive",
        description="Handles user interactions, confirmations, and multi-step wizards. Tests elicitation and tool confirmation flows.",
        model=model,
        instruction="""
            You are an interactive agent that handles user-facing workflows.
            You specialize in:
            - Gathering user preferences through questions
            - Confirming dangerous or important actions before execution
            - Running multi-step wizards that require sequential user input

            When a user wants to do something that requires their input or confirmation,
            use the appropriate tool to request it. Be clear about what you're asking.

            IMPORTANT: For dangerous operations (delete, modify system settings, etc.),
            ALWAYS use confirm_action or dangerous_operation to get user approval first.
        """,
        tools=[
            request_user_preference,
            confirm_action,
            dangerous_operation,
            interactive_wizard,
        ],
    )

    # ========== Workflow Agent (for Task State Machine & Background Tasks) ==========
    async def long_running_task(
        duration_seconds: float = 5.0,
        steps: int = 10,
    ) -> dict[str, Any]:
        """Execute a long-running task with progress updates.

        This tests background task handling and progress tracking.
        The task runs for the specified duration, emitting progress updates.

        Args:
            duration_seconds: Total duration of the task
            steps: Number of progress steps to report

        Returns:
            Final result with execution statistics
        """
        start_time = time.time()
        results = []
        step_duration = duration_seconds / steps

        for i in range(steps):
            await asyncio.sleep(step_duration)
            progress = ((i + 1) / steps) * 100
            results.append({
                "step": i + 1,
                "progress": progress,
                "timestamp": datetime.now().isoformat(),
                "status": TASK_STATE_WORKING,
            })

        elapsed = time.time() - start_time

        return {
            "status": TASK_STATE_COMPLETED,
            "total_steps": steps,
            "duration_seconds": elapsed,
            "results": results,
            "message": f"Long-running task completed in {elapsed:.2f}s",
        }

    async def batch_processor(
        items: list[str],
        batch_size: int = 5,
        delay_per_batch: float = 0.5,
    ) -> dict[str, Any]:
        """Process items in batches with progress updates.

        Tests streaming progress updates during batch processing.

        Args:
            items: List of items to process
            batch_size: Number of items per batch
            delay_per_batch: Simulated processing time per batch

        Returns:
            Processing results with batch-level details
        """
        batches = []
        total = len(items)
        processed = 0

        for i in range(0, total, batch_size):
            batch = items[i:i + batch_size]
            await asyncio.sleep(delay_per_batch)
            processed += len(batch)

            batches.append({
                "batch_index": len(batches),
                "items": batch,
                "processed_count": processed,
                "progress": (processed / total) * 100,
                "status": TASK_STATE_WORKING,
            })

        return {
            "status": TASK_STATE_COMPLETED,
            "total_items": total,
            "total_batches": len(batches),
            "batches": batches,
            "message": f"Processed {total} items in {len(batches)} batches",
        }

    def simulate_failure(
        failure_type: str = "error",
        message: str | None = None,
    ) -> dict[str, Any]:
        """Simulate various failure scenarios for testing error handling.

        Args:
            failure_type: Type of failure (error, timeout, canceled, rejected)
            message: Custom error message

        Returns:
            Failure response with appropriate state
        """
        failure_states = {
            "error": TASK_STATE_FAILED,
            "timeout": TASK_STATE_FAILED,
            "canceled": TASK_STATE_CANCELED,
            "rejected": "rejected",
            "auth_required": "auth-required",
        }

        state = failure_states.get(failure_type, TASK_STATE_FAILED)

        return {
            "status": state,
            "failure_type": failure_type,
            "message": message or f"Simulated {failure_type} failure",
            "recoverable": failure_type in {"timeout", "canceled"},
        }

    def state_machine_demo(
        target_state: str = "working",
    ) -> dict[str, Any]:
        """Demonstrate task state machine transitions.

        This tool explicitly sets the task to a specific state,
        useful for testing state machine alignment.

        Args:
            target_state: Target state (submitted, working, completed, failed, etc.)

        Returns:
            Response with the target state
        """
        valid_states = {
            "submitted": TASK_STATE_SUBMITTED,
            "working": TASK_STATE_WORKING,
            "input_required": TASK_STATE_INPUT_REQUIRED,
            "completed": TASK_STATE_COMPLETED,
            "failed": TASK_STATE_FAILED,
            "canceled": TASK_STATE_CANCELED,
        }

        state = valid_states.get(target_state.lower(), TASK_STATE_WORKING)

        return {
            "status": state,
            "target_state": target_state,
            "message": f"Task state set to: {state}",
            "is_terminal": state in {TASK_STATE_COMPLETED, TASK_STATE_FAILED, TASK_STATE_CANCELED},
        }

    async def progressive_task(
        phases: int = 3,
        phase_duration: float = 1.0,
    ) -> dict[str, Any]:
        """Execute a task with distinct phases and state transitions.

        Tests the full task lifecycle: submitted -> working -> completed

        Args:
            phases: Number of work phases
            phase_duration: Duration of each phase in seconds

        Returns:
            Final result with phase history
        """
        history = [{"state": TASK_STATE_SUBMITTED, "timestamp": datetime.now().isoformat()}]

        # Transition to working
        history.append({"state": TASK_STATE_WORKING, "timestamp": datetime.now().isoformat()})

        for i in range(phases):
            await asyncio.sleep(phase_duration)
            history.append({
                "state": TASK_STATE_WORKING,
                "phase": i + 1,
                "progress": ((i + 1) / phases) * 100,
                "timestamp": datetime.now().isoformat(),
            })

        # Transition to completed
        history.append({"state": TASK_STATE_COMPLETED, "timestamp": datetime.now().isoformat()})

        return {
            "status": TASK_STATE_COMPLETED,
            "phases_completed": phases,
            "state_history": history,
            "message": f"Progressive task completed {phases} phases",
        }

    workflow_agent = Agent(
        name="Workflow",
        description="Handles long-running tasks, batch processing, and state machine demonstrations. Tests background tasks and task state alignment.",
        model=model,
        instruction="""
            You are a workflow orchestration agent that handles:
            - Long-running background tasks with progress tracking
            - Batch processing with incremental updates
            - Task state machine demonstrations
            - Failure simulation for testing error handling

            When executing long-running operations, provide clear progress updates.
            Use the appropriate tools to demonstrate different task states and transitions.
        """,
        tools=[
            long_running_task,
            batch_processor,
            simulate_failure,
            state_machine_demo,
            progressive_task,
        ],
    )

    # ========== SubAgent Visibility Demo Agent ==========
    def announce_branch(
        action: str,
        details: str | None = None,
    ) -> dict[str, str]:
        """Announce the current agent's branch in the hierarchy.

        This helps test sub-agent visibility by explicitly including
        branch information in the response.

        Args:
            action: The action being performed
            details: Additional details

        Returns:
            Response with branch metadata
        """
        return {
            "action": action,
            "details": details or "",
            "agent": "BranchDemo",
            "branch_hint": "root.BranchDemo",
            "message": f"BranchDemo agent executing: {action}",
        }

    def delegate_to_sub(
        task: str,
        sub_agent: str = "level2",
    ) -> dict[str, Any]:
        """Simulate delegation to a deeper sub-agent.

        Tests the visibility of nested agent hierarchies.

        Args:
            task: The task to delegate
            sub_agent: Which sub-agent to delegate to

        Returns:
            Response simulating sub-agent execution
        """
        return {
            "delegated_task": task,
            "delegated_to": sub_agent,
            "branch_hint": f"root.BranchDemo.{sub_agent}",
            "simulated_response": f"Sub-agent {sub_agent} processed: {task}",
            "hierarchy": ["root", "BranchDemo", sub_agent],
        }

    branch_demo_agent = Agent(
        name="BranchDemo",
        description="Demonstrates sub-agent visibility and branch tracking in multi-agent hierarchies.",
        model=model,
        instruction="""
            You are a demonstration agent for sub-agent visibility features.
            Your responses should help test the branch tracking system that shows
            which agent in a hierarchy is currently processing.

            Use announce_branch to show where you are in the hierarchy.
            Use delegate_to_sub to simulate deeper delegation chains.
        """,
        tools=[announce_branch, delegate_to_sub],
    )

    # ========== Root Orchestrator Agent ==========
    root_agent = Agent(
        name="TestAgentRoot",
        description="Multi-capability test agent that orchestrates specialized subagents for calculations, data processing, text manipulation, information retrieval, interactive workflows, background tasks, and sub-agent visibility demonstrations.",
        model=model,
        instruction="""
            You are a versatile orchestrator agent that coordinates multiple specialized subagents.

            Your subagents are:
            - Calculator: For arithmetic operations and statistical calculations
            - DataProcessor: For list/data manipulation, filtering, sorting, and batch processing
            - TextProcessor: For text transformations, analysis, and manipulation
            - InfoRetriever: For information lookup, facts, and data retrieval
            - Interactive: For user confirmations, preferences, and multi-step wizards
            - Workflow: For long-running tasks, batch processing, and state machine demos
            - BranchDemo: For demonstrating sub-agent visibility and hierarchy tracking

            When a user makes a request:
            1. Analyze the request to understand what capability is needed
            2. Delegate to the appropriate subagent(s)
            3. If a request requires multiple capabilities, coordinate between agents
            4. Present results clearly to the user

            IMPORTANT ROUTING GUIDELINES:
            - For dangerous operations (delete, modify, execute) -> Interactive agent
            - For anything requiring user input or confirmation -> Interactive agent
            - For long-running or background tasks -> Workflow agent
            - For testing states/failures/progress -> Workflow agent
            - For sub-agent hierarchy demonstrations -> BranchDemo agent

            You can handle complex multi-step workflows by sequencing subagent calls.
            Always be helpful, accurate, and clear in your responses.
        """,
        sub_agents=[
            calculator_agent,
            data_agent,
            text_agent,
            info_agent,
            interactive_agent,
            workflow_agent,
            branch_demo_agent,
        ],
    )

    return root_agent
