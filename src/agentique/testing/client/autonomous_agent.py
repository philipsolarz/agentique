"""LLM-Powered Autonomous Test Agent.

This agent uses an LLM (Gemini, Claude, etc.) to intelligently explore and test
MCP servers. It discovers capabilities, generates test queries, analyzes responses,
and adapts its testing strategy based on results.

This creates a self-testing loop: AI agents testing AI infrastructure.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .mcp_client import MCPTestClient

logger = logging.getLogger(__name__)


class TestObjective(BaseModel):
    """A high-level testing objective for the autonomous agent."""

    goal: str
    focus_areas: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class TestAction(BaseModel):
    """A single action the agent decides to take."""

    action_type: str  # "send_message", "call_tool", "analyze", "report"
    reasoning: str  # Why the agent chose this action
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str = ""


class TestInsight(BaseModel):
    """An insight discovered during testing."""

    category: str  # "bug", "feature", "limitation", "edge_case"
    description: str
    severity: str = "info"  # "critical", "high", "medium", "low", "info"
    evidence: list[str] = Field(default_factory=list)


class AutonomousTestAgent:
    """An LLM-powered agent that autonomously tests MCP servers.

    The agent:
    1. Discovers MCP server capabilities (tools, resources, prompts)
    2. Generates intelligent test queries using an LLM
    3. Analyzes responses and adapts strategy
    4. Explores edge cases and generates insights
    5. Reports findings and coverage
    """

    def __init__(
        self,
        client: MCPTestClient,
        llm_provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        simulate: bool = True,
    ) -> None:
        self.client = client
        self.llm_provider = llm_provider
        self.model = model
        self.simulate = simulate

        # Testing state
        self.discovered_tools: list[dict[str, Any]] = []
        self.test_history: list[dict[str, Any]] = []
        self.insights: list[TestInsight] = []

        # LLM client
        self._llm_client = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize the LLM client based on provider."""
        if self.llm_provider == "gemini":
            try:
                import google.generativeai as genai

                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not set")

                genai.configure(api_key=api_key)
                self._llm_client = genai.GenerativeModel(self.model)
                logger.info("Initialized Gemini model: %s", self.model)
            except ImportError:
                logger.error("google-generativeai not installed")
                raise
        elif self.llm_provider == "anthropic":
            # Future: Add Claude support
            raise NotImplementedError("Anthropic provider not yet implemented")
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    async def discover_capabilities(self) -> dict[str, Any]:
        """Discover what the MCP server can do.

        Returns:
            Dict with tools, resources, prompts, and capabilities
        """
        logger.info("Discovering MCP server capabilities...")

        capabilities = {
            "tools": [],
            "resources": [],
            "prompts": [],
            "metadata": {},
        }

        # List all tools
        try:
            tools = await self.client.list_tools()
            capabilities["tools"] = tools
            self.discovered_tools = tools
            logger.info("Discovered %d tools", len(tools))
        except Exception as e:
            logger.error("Failed to list tools: %s", e)

        # TODO: Add resource and prompt discovery when MCP client supports it

        return capabilities

    async def generate_test_plan(
        self,
        objective: TestObjective,
        capabilities: dict[str, Any],
    ) -> list[TestAction]:
        """Use LLM to generate a test plan based on discovered capabilities.

        Args:
            objective: High-level testing goal
            capabilities: Discovered MCP server capabilities

        Returns:
            List of test actions to execute
        """
        logger.info("Generating test plan for objective: %s", objective.goal)

        # Build prompt for LLM
        prompt = self._build_planning_prompt(objective, capabilities)

        # Get LLM response
        try:
            response = await self._call_llm(prompt)
            actions = self._parse_test_actions(response)
            logger.info("Generated %d test actions", len(actions))
            return actions
        except Exception as e:
            logger.error("Failed to generate test plan: %s", e)
            return []

    async def execute_action(self, action: TestAction) -> dict[str, Any]:
        """Execute a single test action.

        Args:
            action: The action to execute

        Returns:
            Result of the action
        """
        logger.info("Executing: %s - %s", action.action_type, action.reasoning)

        result = {
            "action": action.model_dump(),
            "success": False,
            "response": "",
            "error": None,
        }

        try:
            if action.action_type == "send_message":
                message = action.parameters.get("message", "")
                target = action.parameters.get("target")
                timeout = action.parameters.get("timeout", 30)

                response = await self.client.send_message(
                    message=message,
                    target=target,
                    timeout=timeout,
                )

                # Ensure we have a string response
                text = response.get("text", "")
                if not isinstance(text, str):
                    # Convert non-string responses to string representation
                    text = str(text)
                    logger.warning(
                        "Response was not a string, got %s: %s",
                        type(text).__name__,
                        repr(text)[:100],
                    )

                result["success"] = not response.get("is_error", False)
                result["response"] = text

            elif action.action_type == "call_tool":
                tool_name = action.parameters.get("tool", "")
                arguments = action.parameters.get("arguments", {})
                timeout = action.parameters.get("timeout", 30)

                # Validate tool name is not empty
                if not tool_name:
                    result["error"] = (
                        f"Tool name is empty. Parameters: {action.parameters}. "
                        f"LLM must include 'tool' key in parameters for call_tool actions."
                    )
                    logger.error(result["error"])
                else:
                    response = await self.client.call_tool(
                        name=tool_name,
                        arguments=arguments,
                        timeout=timeout,
                    )

                    result["success"] = not response.get("is_error", False)
                    result["response"] = response.get("text", "")

            elif action.action_type == "analyze":
                # Use LLM to analyze previous results
                analysis = await self._analyze_results()
                result["success"] = True
                result["response"] = analysis

            else:
                result["error"] = f"Unknown action type: {action.action_type}"

        except Exception as e:
            result["error"] = str(e)
            logger.exception("Action failed")

        # Record in history
        self.test_history.append(result)

        return result

    async def analyze_result(
        self,
        action: TestAction,
        result: dict[str, Any],
    ) -> TestInsight | None:
        """Use LLM to analyze test result and extract insights.

        Args:
            action: The action that was executed
            result: The result of the action

        Returns:
            Insight if any discovered, None otherwise
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt(action, result)

        try:
            response = await self._call_llm(prompt)
            insight = self._parse_insight(response)

            if insight:
                self.insights.append(insight)
                logger.info(
                    "Discovered insight: [%s] %s",
                    insight.category,
                    insight.description,
                )

            return insight

        except Exception as e:
            logger.error("Failed to analyze result: %s", e)
            return None

    async def autonomous_exploration(
        self,
        objective: TestObjective,
        max_actions: int = 20,
    ) -> dict[str, Any]:
        """Autonomously explore and test the MCP server.

        Args:
            objective: High-level testing goal
            max_actions: Maximum number of actions to take

        Returns:
            Summary of exploration results
        """
        logger.info("Starting autonomous exploration: %s", objective.goal)

        # 1. Discover capabilities
        capabilities = await self.discover_capabilities()

        # 2. Generate initial test plan
        actions = await self.generate_test_plan(objective, capabilities)

        # 3. Execute actions and adapt
        executed_count = 0
        for action in actions[:max_actions]:
            if executed_count >= max_actions:
                break

            # Execute action
            result = await self.execute_action(action)
            executed_count += 1

            # Analyze result
            insight = await self.analyze_result(action, result)

            # Adapt strategy based on insights
            if insight and insight.severity in {"critical", "high"}:
                # Generate follow-up actions to investigate
                followup = await self._generate_followup_actions(insight)
                actions.extend(followup)

        # 4. Generate summary report
        report = self._generate_report(objective, executed_count)

        logger.info(
            "Exploration complete: %d actions, %d insights",
            executed_count,
            len(self.insights),
        )

        return report

    # --- LLM Interaction Helpers ---

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        if self.llm_provider == "gemini":
            response = self._llm_client.generate_content(prompt)
            return response.text
        else:
            raise NotImplementedError(f"Provider {self.llm_provider} not supported")

    def _build_planning_prompt(
        self,
        objective: TestObjective,
        capabilities: dict[str, Any],
    ) -> str:
        """Build a prompt for test planning."""
        tools_summary = "\n".join(
            f"- {t['name']}: {t.get('description', 'No description')}"
            for t in capabilities.get("tools", [])
        )

        return f"""You are an autonomous test agent for an MCP (Model Context Protocol) server.

Your objective: {objective.goal}

Focus areas: {', '.join(objective.focus_areas) if objective.focus_areas else 'General exploration'}

Available MCP Tools:
{tools_summary}

Generate a test plan with 5-10 intelligent test actions. For each action, provide:
1. action_type: "send_message" or "call_tool"
2. reasoning: Why this test is important
3. parameters: The specific parameters (see examples below)
4. expected_outcome: What you expect to happen

**IMPORTANT**: Return ONLY valid JSON, no markdown formatting or extra text.

For "send_message" actions, parameters should be:
{{"message": "your message here", "target": "optional-agent-name"}}

For "call_tool" actions, parameters MUST include the tool name:
{{"tool": "tool_name_here", "arguments": {{"arg1": "value1"}}}}

Return your response as a JSON array of test actions in this format:
[
  {{
    "action_type": "send_message",
    "reasoning": "Test basic agent interaction",
    "parameters": {{"message": "What can you help me with?"}},
    "expected_outcome": "Agent should describe its capabilities"
  }},
  {{
    "action_type": "call_tool",
    "reasoning": "Verify agents tool works",
    "parameters": {{"tool": "agents", "arguments": {{}}}},
    "expected_outcome": "List of available agents returned"
  }}
]

Focus on:
- Testing core functionality
- Exploring edge cases
- Validating error handling
- Checking response quality

REMEMBER: For call_tool, ALWAYS include "tool" in parameters!
"""

    def _build_analysis_prompt(
        self,
        action: TestAction,
        result: dict[str, Any],
    ) -> str:
        """Build a prompt for result analysis."""
        return f"""Analyze this test result and identify any insights.

Action Taken:
- Type: {action.action_type}
- Reasoning: {action.reasoning}
- Expected: {action.expected_outcome}

Actual Result:
- Success: {result.get('success')}
- Response: {result.get('response', '')[:500]}
- Error: {result.get('error', 'None')}

Identify any:
- Bugs or errors
- Unexpected behavior
- Performance issues
- Missing features
- Edge cases

Return your analysis as JSON:
{{
  "has_insight": true/false,
  "category": "bug|feature|limitation|edge_case",
  "description": "Brief description",
  "severity": "critical|high|medium|low|info",
  "evidence": ["observation 1", "observation 2"]
}}

If no significant insight, set has_insight to false.
"""

    def _parse_test_actions(self, llm_response: str) -> list[TestAction]:
        """Parse LLM response into test actions."""
        try:
            # Extract JSON from response (LLM might add markdown formatting)
            json_start = llm_response.find("[")
            json_end = llm_response.rfind("]") + 1

            if json_start == -1 or json_end == 0:
                logger.error("No JSON array found in LLM response")
                return []

            json_str = llm_response[json_start:json_end]
            actions_data = json.loads(json_str)

            return [TestAction(**action) for action in actions_data]

        except Exception as e:
            logger.error("Failed to parse test actions: %s", e)
            return []

    def _parse_insight(self, llm_response: str) -> TestInsight | None:
        """Parse LLM response into an insight."""
        try:
            # Extract JSON from response
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                return None

            json_str = llm_response[json_start:json_end]
            data = json.loads(json_str)

            if not data.get("has_insight", False):
                return None

            return TestInsight(
                category=data.get("category", "info"),
                description=data.get("description", ""),
                severity=data.get("severity", "info"),
                evidence=data.get("evidence", []),
            )

        except Exception as e:
            logger.error("Failed to parse insight: %s", e)
            return None

    async def _analyze_results(self) -> str:
        """Analyze all test results so far."""
        summary = f"Analyzed {len(self.test_history)} test results.\n"
        success_count = sum(1 for r in self.test_history if r.get("success"))
        summary += f"Success rate: {success_count}/{len(self.test_history)}\n"
        return summary

    async def _generate_followup_actions(
        self,
        insight: TestInsight,
    ) -> list[TestAction]:
        """Generate follow-up actions to investigate an insight."""
        # For now, return empty list - could use LLM to generate followups
        return []

    def _generate_report(
        self,
        objective: TestObjective,
        actions_executed: int,
    ) -> dict[str, Any]:
        """Generate final test report."""
        success_count = sum(1 for r in self.test_history if r.get("success"))

        return {
            "objective": objective.goal,
            "actions_executed": actions_executed,
            "success_rate": success_count / max(actions_executed, 1),
            "insights_discovered": len(self.insights),
            "insights": [i.model_dump() for i in self.insights],
            "coverage": {
                "tools_tested": len(
                    {
                        r["action"]["parameters"].get("tool")
                        for r in self.test_history
                        if r["action"]["action_type"] == "call_tool"
                    }
                ),
                "total_tools": len(self.discovered_tools),
            },
        }
