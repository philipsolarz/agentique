"""Unit tests for the agentique.simulation package."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentique.simulation.models import (
    ConversationTurn,
    EventType,
    SessionEvent,
    SimulationConfig,
    SimulationInsight,
    SimulationObjective,
    SimulationPersona,
    SimulationResult,
    SimulationState,
)
from agentique.simulation.simulation_agent import SimulationAgent


# ---- Model Tests ----


class TestSimulationModels:
    def test_persona_defaults(self):
        persona = SimulationPersona()
        assert persona.name == "Alex"
        assert persona.role == "curious user"
        assert len(persona.personality_traits) > 0
        assert persona.typing_speed_ms == 80.0
        assert persona.reading_speed_ms == 20.0

    def test_persona_custom(self):
        persona = SimulationPersona(
            name="Bob",
            role="developer",
            personality_traits=["technical"],
            conversation_style="formal",
        )
        assert persona.name == "Bob"
        assert persona.role == "developer"

    def test_objective_defaults(self):
        obj = SimulationObjective()
        assert obj.max_turns == 10
        assert obj.max_duration_seconds == 600.0
        assert isinstance(obj.topics, list)

    def test_config_composition(self):
        config = SimulationConfig(
            persona=SimulationPersona(name="Test"),
            objective=SimulationObjective(goal="Test goal", max_turns=5),
        )
        assert config.persona.name == "Test"
        assert config.objective.goal == "Test goal"
        assert config.llm_provider == "gemini"

    def test_conversation_turn(self):
        turn = ConversationTurn(
            turn_number=1,
            role="human",
            message="Hello!",
            thinking_time_ms=100.0,
        )
        assert turn.turn_number == 1
        assert turn.role == "human"
        assert turn.timestamp > 0

    def test_simulation_state_values(self):
        assert SimulationState.IDLE == "idle"
        assert SimulationState.RUNNING == "running"
        assert SimulationState.COMPLETED == "completed"

    def test_simulation_insight(self):
        insight = SimulationInsight(
            category="bug",
            description="Response was empty",
            severity="high",
            turn_number=3,
            evidence=["Empty string returned"],
        )
        assert insight.category == "bug"
        assert insight.severity == "high"
        assert len(insight.evidence) == 1

    def test_simulation_result(self):
        result = SimulationResult(
            state=SimulationState.COMPLETED,
            summary="Went well",
        )
        assert result.simulation_id  # auto-generated
        assert result.state == SimulationState.COMPLETED
        assert result.summary == "Went well"

    def test_event_type_simulation_events(self):
        # Verify all simulation-specific event types exist
        assert EventType.SIMULATION_STARTED == "simulation_started"
        assert EventType.SIMULATION_COMPLETED == "simulation_completed"
        assert EventType.THINKING == "thinking"
        assert EventType.TYPING == "typing"
        assert EventType.MESSAGE_SENT == "message_sent"
        assert EventType.WAITING == "waiting"
        assert EventType.RESPONSE_RECEIVED == "response_received"
        assert EventType.READING == "reading"
        assert EventType.TURN_COMPLETE == "turn_complete"
        assert EventType.INSIGHT == "insight"
        assert EventType.SUMMARY == "summary"

    def test_session_event(self):
        event = SessionEvent(
            type=EventType.THINKING,
            data={"persona": "Alex"},
            simulation_id="abc123",
        )
        assert event.type == EventType.THINKING
        assert event.simulation_id == "abc123"
        assert event.timestamp > 0


# ---- SimulationAgent Tests ----


class TestSimulationAgent:
    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        client.send_message = AsyncMock(return_value={
            "text": "I can help with math, text, and more!",
            "is_error": False,
        })
        return client

    @pytest.fixture
    def config(self):
        return SimulationConfig(
            persona=SimulationPersona(name="TestUser", typing_speed_ms=1, reading_speed_ms=1),
            objective=SimulationObjective(goal="Test the AI", max_turns=2),
        )

    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def event_handler(self, events):
        async def handler(event):
            events.append(event)
        return handler

    def _make_agent(self, mock_client, config, event_handler):
        """Create a SimulationAgent with mocked LLM."""
        with patch.object(SimulationAgent, "_initialize_llm"):
            agent = SimulationAgent(
                client=mock_client,
                config=config,
                on_event=event_handler,
            )
            agent._llm_client = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_run_emits_started_and_completed(self, mock_client, config, events, event_handler):
        agent = self._make_agent(mock_client, config, event_handler)

        # Mock LLM responses for max_turns=2:
        # Turn 1: opening message -> send -> response -> should_continue? yes
        # Turn 2: next message -> send -> response -> (no should_continue, at max)
        # Then: analyze conversation
        agent._call_llm = AsyncMock(side_effect=[
            "Hi, what can you do?",           # opening message
            "yes",                             # should_continue after turn 1
            "Tell me more about math",         # next message for turn 2
            json.dumps({                       # analysis
                "summary": "Good conversation",
                "insights": [],
            }),
        ])

        result = await agent.run()

        assert result.state == SimulationState.COMPLETED
        assert len(result.conversation) == 4  # 2 human + 2 agent turns

        event_types = [e.type for e in events]
        assert EventType.SIMULATION_STARTED in event_types
        assert EventType.MESSAGE_SENT in event_types
        assert EventType.RESPONSE_RECEIVED in event_types
        assert EventType.TURN_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_run_respects_max_turns(self, mock_client, config, events, event_handler):
        config.objective.max_turns = 1

        agent = self._make_agent(mock_client, config, event_handler)

        # Only need 1 turn worth of LLM calls
        agent._call_llm = AsyncMock(side_effect=[
            "Hello!",                          # opening
            json.dumps({                       # analysis
                "summary": "Short conversation",
                "insights": [],
            }),
        ])

        result = await agent.run()

        assert len(result.conversation) == 2  # 1 human + 1 agent

    @pytest.mark.asyncio
    async def test_stop_ends_simulation(self, mock_client, config, events, event_handler):
        config.objective.max_turns = 5
        agent = self._make_agent(mock_client, config, event_handler)

        call_count = 0

        async def llm_that_stops(prompt):
            nonlocal call_count
            call_count += 1
            # After the opening message is sent and should_continue is asked,
            # trigger stop so the next loop iteration catches it
            if call_count == 1:
                return "Hello there!"  # opening message
            elif call_count == 2:
                # should_continue after turn 1 — say yes but trigger stop
                agent.stop()
                return "yes"
            elif "analyze" in prompt.lower() or "identify" in prompt.lower():
                return json.dumps({"summary": "Stopped", "insights": []})
            return "More chat"

        agent._call_llm = AsyncMock(side_effect=llm_that_stops)

        result = await agent.run()

        assert result.state == SimulationState.STOPPED

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, mock_client, config, events, event_handler):
        agent = self._make_agent(mock_client, config, event_handler)

        assert agent.state == SimulationState.IDLE

        await agent.pause()
        assert agent.state == SimulationState.PAUSED

        await agent.resume()
        assert agent.state == SimulationState.RUNNING

    @pytest.mark.asyncio
    async def test_failed_simulation(self, mock_client, config, events, event_handler):
        agent = self._make_agent(mock_client, config, event_handler)

        # LLM raises an error
        agent._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        result = await agent.run()

        assert result.state == SimulationState.FAILED
        assert "failed" in result.summary.lower()

        event_types = [e.type for e in events]
        assert EventType.SIMULATION_FAILED in event_types

    @pytest.mark.asyncio
    async def test_mcp_error_handled(self, mock_client, config, events, event_handler):
        mock_client.send_message = AsyncMock(side_effect=ConnectionError("Connection refused"))

        agent = self._make_agent(mock_client, config, event_handler)
        agent._call_llm = AsyncMock(side_effect=[
            "Hello!",                          # opening
            "no",                              # should_continue
            json.dumps({                       # analysis
                "summary": "Connection issues",
                "insights": [{"category": "bug", "description": "Connection refused", "severity": "high", "evidence": []}],
            }),
        ])

        result = await agent.run()

        # Should complete (not crash) even with MCP errors
        assert result.state in (SimulationState.COMPLETED, SimulationState.STOPPED)
        # Should have recorded the error response
        agent_turns = [t for t in result.conversation if t.role == "agent"]
        assert any("Error" in t.message for t in agent_turns)

    @pytest.mark.asyncio
    async def test_events_have_simulation_id(self, mock_client, config, events, event_handler):
        agent = self._make_agent(mock_client, config, event_handler)
        agent._call_llm = AsyncMock(side_effect=[
            "Hi!",
            json.dumps({"summary": "Done", "insights": []}),
        ])
        config.objective.max_turns = 1

        await agent.run()

        for event in events:
            assert event.simulation_id == agent.simulation_id

    @pytest.mark.asyncio
    async def test_generate_opening_uses_persona(self, mock_client, config, event_handler):
        config.persona.name = "CustomName"
        config.persona.role = "tester"

        agent = self._make_agent(mock_client, config, event_handler)

        captured_prompts = []

        async def capture_llm(prompt):
            captured_prompts.append(prompt)
            return "Hello!"

        agent._call_llm = AsyncMock(side_effect=capture_llm)

        await agent._generate_opening_message()

        assert len(captured_prompts) == 1
        assert "CustomName" in captured_prompts[0]
        assert "tester" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_should_continue_returns_bool(self, mock_client, config, event_handler):
        agent = self._make_agent(mock_client, config, event_handler)
        agent.conversation = [
            ConversationTurn(turn_number=1, role="human", message="Hi"),
            ConversationTurn(turn_number=1, role="agent", message="Hello!"),
        ]

        agent._call_llm = AsyncMock(return_value="yes")
        assert await agent._should_continue() is True

        agent._call_llm = AsyncMock(return_value="no")
        assert await agent._should_continue() is False


# ---- SimulationHarness Tests ----


class TestSimulationHarness:
    @pytest.fixture
    def harness(self):
        from agentique.simulation.app import SimulationHarness
        return SimulationHarness(mcp_url="http://localhost:8000/mcp")

    def test_list_simulations_empty(self, harness):
        sims = harness.list_simulations()
        assert sims == []

    def test_get_status_not_found(self, harness):
        assert harness.get_status("nonexistent") is None

    def test_get_events_empty(self, harness):
        events = harness.get_events("nonexistent")
        assert events == []

    def test_get_insights_empty(self, harness):
        insights = harness.get_insights("nonexistent")
        assert insights == []

    def test_get_result_not_found(self, harness):
        assert harness.get_result("nonexistent") is None

    def test_stop_unknown_simulation(self, harness):
        assert harness.stop_simulation("nonexistent") is False

    @pytest.mark.asyncio
    async def test_pause_unknown_simulation(self, harness):
        assert await harness.pause_simulation("nonexistent") is False

    @pytest.mark.asyncio
    async def test_resume_unknown_simulation(self, harness):
        assert await harness.resume_simulation("nonexistent") is False

    def test_ws_management(self, harness):
        mock_ws = MagicMock()
        harness.add_ws(mock_ws)
        assert mock_ws in harness._ws_connections
        harness.remove_ws(mock_ws)
        assert mock_ws not in harness._ws_connections


# ---- REST API Tests ----


class TestSimulationRESTAPI:
    @pytest.fixture
    def app(self):
        from agentique.simulation.app import create_app
        return create_app(mcp_url="http://localhost:8000/mcp")

    @pytest.fixture
    def client(self, app):
        from starlette.testclient import TestClient
        return TestClient(app)

    def test_homepage(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Agentique Simulation" in resp.text

    def test_list_simulations_empty(self, client):
        resp = client.get("/api/simulations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulations"] == []

    def test_get_simulation_not_found(self, client):
        resp = client.get("/api/simulations/nonexistent")
        assert resp.status_code == 404

    def test_delete_simulation_not_found(self, client):
        resp = client.delete("/api/simulations/nonexistent")
        assert resp.status_code == 404

    def test_get_events_empty(self, client):
        resp = client.get("/api/simulations/nonexistent/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []

    def test_get_insights_empty(self, client):
        resp = client.get("/api/simulations/nonexistent/insights")
        assert resp.status_code == 200
        data = resp.json()
        assert data["insights"] == []

    def test_static_files(self, client):
        resp = client.get("/static/style.css")
        assert resp.status_code == 200

        resp = client.get("/static/app.js")
        assert resp.status_code == 200

        resp = client.get("/static/simulator.js")
        assert resp.status_code == 200


# ---- Simulator MCP Server Tests ----


class TestSimulatorMCPServer:
    def test_create_server_headless(self):
        from agentique.simulation.mcp_server.server import create_simulator_mcp_server
        server = create_simulator_mcp_server(mcp_url="http://localhost:8000/mcp")
        assert server is not None
        assert server.name == "Agentique Simulator"

    def test_create_server_with_ui_url(self):
        from agentique.simulation.mcp_server.server import create_simulator_mcp_server
        server = create_simulator_mcp_server(
            simulation_ui_url="http://localhost:8080",
            mcp_url="http://localhost:8000/mcp",
        )
        assert server is not None


# ---- Resilience Tests ----


class TestRetryConfig:
    def test_config_retry_defaults(self):
        config = SimulationConfig()
        assert config.llm_max_retries == 3
        assert config.llm_retry_base_delay == 2.0

    def test_config_retry_custom(self):
        config = SimulationConfig(llm_max_retries=5, llm_retry_base_delay=1.0)
        assert config.llm_max_retries == 5
        assert config.llm_retry_base_delay == 1.0


class TestLLMRetry:
    @pytest.fixture
    def mock_client(self):
        return AsyncMock()

    @pytest.fixture
    def config(self):
        return SimulationConfig(
            persona=SimulationPersona(name="TestUser", typing_speed_ms=1, reading_speed_ms=1),
            objective=SimulationObjective(goal="Test", max_turns=1),
            llm_max_retries=2,
            llm_retry_base_delay=0.01,  # Fast for tests
        )

    def _make_agent(self, mock_client, config):
        with patch.object(SimulationAgent, "_initialize_llm"):
            agent = SimulationAgent(client=mock_client, config=config)
            agent._llm_client = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_call_llm_retries_on_429(self, mock_client, config):
        agent = self._make_agent(mock_client, config)

        # Simulate 2 rate limit failures then success
        rate_limit_error = Exception("429 Resource has been exhausted")
        mock_response = MagicMock()
        mock_response.text = "Success!"
        agent._llm_client.generate_content = MagicMock(
            side_effect=[rate_limit_error, rate_limit_error, mock_response]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await agent._call_llm("test prompt")

        assert result == "Success!"
        assert agent._llm_client.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_call_llm_gives_up_after_max_retries(self, mock_client, config):
        agent = self._make_agent(mock_client, config)

        rate_limit_error = Exception("429 rate limit exceeded")
        agent._llm_client.generate_content = MagicMock(side_effect=rate_limit_error)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="429"):
                await agent._call_llm("test prompt")

        # Initial attempt + max_retries = 3 total
        assert agent._llm_client.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_call_llm_no_retry_on_non_retryable(self, mock_client, config):
        agent = self._make_agent(mock_client, config)

        # PermissionDenied is not retryable
        perm_error = PermissionError("API key invalid")
        agent._llm_client.generate_content = MagicMock(side_effect=perm_error)

        with pytest.raises(PermissionError, match="API key invalid"):
            await agent._call_llm("test prompt")

        # Should fail immediately, no retries
        assert agent._llm_client.generate_content.call_count == 1

    def test_is_retryable_error(self):
        # Retryable: 429 in message
        assert SimulationAgent._is_retryable_error(Exception("429 rate limit")) is True
        # Retryable: rate limit in message
        assert SimulationAgent._is_retryable_error(Exception("rate limit exceeded")) is True
        # Retryable: ConnectionError
        assert SimulationAgent._is_retryable_error(ConnectionError("refused")) is True
        # Retryable: TimeoutError
        assert SimulationAgent._is_retryable_error(TimeoutError("timed out")) is True
        # Retryable: 503 in message
        assert SimulationAgent._is_retryable_error(Exception("503 Service Unavailable")) is True
        # Not retryable: generic error
        assert SimulationAgent._is_retryable_error(ValueError("bad input")) is False
        # Not retryable: permission error
        assert SimulationAgent._is_retryable_error(PermissionError("denied")) is False

    @pytest.mark.asyncio
    async def test_call_llm_uses_asyncio_to_thread(self, mock_client, config):
        agent = self._make_agent(mock_client, config)

        mock_response = MagicMock()
        mock_response.text = "threaded response"
        agent._llm_client.generate_content = MagicMock(return_value=mock_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_response) as mock_to_thread:
            result = await agent._call_llm("test prompt")

        assert result == "threaded response"
        mock_to_thread.assert_called_once_with(
            agent._llm_client.generate_content, "test prompt"
        )


class TestBestEffortInsights:
    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        client.send_message = AsyncMock(return_value={"text": "Hello!", "is_error": False})
        return client

    @pytest.fixture
    def config(self):
        return SimulationConfig(
            persona=SimulationPersona(name="TestUser", typing_speed_ms=1, reading_speed_ms=1),
            objective=SimulationObjective(goal="Test", max_turns=3),
        )

    def _make_agent(self, mock_client, config, on_event=None):
        with patch.object(SimulationAgent, "_initialize_llm"):
            agent = SimulationAgent(client=mock_client, config=config, on_event=on_event)
            agent._llm_client = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_failed_simulation_attempts_analysis(self, mock_client, config):
        agent = self._make_agent(mock_client, config)

        call_count = 0

        async def llm_side_effect(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Hello!"  # opening message
            elif call_count == 2:
                # should_continue — raise to simulate crash
                raise RuntimeError("LLM crashed on turn 2")
            elif "analyze" in prompt.lower() or "identify" in prompt.lower():
                return json.dumps({
                    "summary": "Partial analysis from 1 turn",
                    "insights": [{"category": "observation", "description": "Agent greeted user", "severity": "info", "evidence": ["Hello!"]}],
                })
            return "fallback"

        agent._call_llm = AsyncMock(side_effect=llm_side_effect)

        result = await agent.run()

        assert result.state == SimulationState.FAILED
        # Best-effort analysis should have produced insights
        assert len(result.insights) > 0
        assert "Partial analysis" in result.summary

    @pytest.mark.asyncio
    async def test_failed_simulation_analysis_also_fails(self, mock_client, config):
        agent = self._make_agent(mock_client, config)

        call_count = 0

        async def llm_side_effect(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Hello!"
            else:
                raise RuntimeError("LLM is completely down")

        agent._call_llm = AsyncMock(side_effect=llm_side_effect)

        result = await agent.run()

        assert result.state == SimulationState.FAILED
        # Should fall back to error summary since analysis also failed
        assert "failed" in result.summary.lower() or "LLM" in result.summary


class TestHarnessPreservesData:
    @pytest.mark.asyncio
    async def test_harness_preserves_agent_data_on_failure(self):
        from agentique.simulation.app import SimulationHarness

        harness = SimulationHarness(mcp_url="http://localhost:8000/mcp")

        # Create a mock agent with conversation data
        mock_agent = MagicMock()
        mock_agent.simulation_id = "test-123"
        mock_agent.conversation = [
            ConversationTurn(turn_number=1, role="human", message="Hi"),
            ConversationTurn(turn_number=1, role="agent", message="Hello!"),
        ]
        mock_agent.insights = [
            SimulationInsight(category="observation", description="Greeting worked", severity="info"),
        ]
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Unexpected crash"))

        mock_client = AsyncMock()

        harness._simulations["test-123"] = mock_agent
        harness._events["test-123"] = []

        await harness._run_simulation("test-123", mock_client, mock_agent)

        result = harness._results["test-123"]
        assert result["state"] == SimulationState.FAILED.value
        assert len(result["conversation"]) == 2
        assert len(result["insights"]) == 1
        assert result["insights"][0]["category"] == "observation"


class TestMCPServerAPIErrorHandling:
    @pytest.mark.asyncio
    async def test_api_call_handles_http_error(self):
        from agentique.simulation.mcp_server.server import create_simulator_mcp_server

        server = create_simulator_mcp_server(
            simulation_ui_url="http://localhost:9999",
            mcp_url="http://localhost:8000/mcp",
        )

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await server.call_tool("list_simulations", {})
            result_data = json.loads(result.content[0].text)

        assert "error" in result_data
        assert result_data["status_code"] == 500

    @pytest.mark.asyncio
    async def test_api_call_handles_connection_error(self):
        from agentique.simulation.mcp_server.server import create_simulator_mcp_server

        server = create_simulator_mcp_server(
            simulation_ui_url="http://localhost:9999",
            mcp_url="http://localhost:8000/mcp",
        )

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await server.call_tool("list_simulations", {})
            result_data = json.loads(result.content[0].text)

        assert "error" in result_data
        assert "Connection failed" in result_data["error"]


# ---- Cogito Integration Tests ----


class TestCogitoPersona:
    def test_cogito_persona_factory(self):
        persona = SimulationPersona.cogito()
        assert persona.name == "Cogito"
        assert persona.role == "autonomous improvement agent in a self-improving development loop"
        assert "relentlessly improvement-driven" in persona.personality_traits
        assert "demands concrete code over discussion" in persona.personality_traits
        assert "pushes for implementable diffs" in persona.personality_traits
        assert persona.conversation_style == "directive, demanding, and diff-oriented"
        assert persona.typing_speed_ms == 120.0
        assert persona.reading_speed_ms == 30.0

    def test_cogito_persona_custom_name(self):
        persona = SimulationPersona.cogito("Aristotle")
        assert persona.name == "Aristotle"
        assert persona.role == "autonomous improvement agent in a self-improving development loop"

    def test_config_cogito_reasoning_defaults(self):
        config = SimulationConfig()
        assert config.use_cogito_reasoning is False
        assert config.reasoning_mcp_url is None


class TestCogitoReasoningEngine:
    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        client.send_message = AsyncMock(return_value={
            "text": "I can help with reasoning!",
            "is_error": False,
        })
        return client

    @pytest.fixture
    def mock_reasoning_client(self):
        client = AsyncMock()
        client.send_message = AsyncMock(return_value={
            "text": "Through analysis of the MCP bridge architecture...",
            "is_error": False,
        })
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()
        return client

    @pytest.fixture
    def cogito_config(self):
        return SimulationConfig(
            persona=SimulationPersona.cogito(),
            objective=SimulationObjective(goal="Explore self-referential reasoning", max_turns=2),
            use_cogito_reasoning=True,
        )

    @pytest.fixture
    def normal_config(self):
        return SimulationConfig(
            persona=SimulationPersona(name="TestUser", typing_speed_ms=1, reading_speed_ms=1),
            objective=SimulationObjective(goal="Test", max_turns=1),
            use_cogito_reasoning=False,
        )

    def _make_agent(self, client, config, reasoning_client=None, on_event=None):
        with patch.object(SimulationAgent, "_initialize_llm"):
            agent = SimulationAgent(
                client=client,
                config=config,
                on_event=on_event,
                reasoning_client=reasoning_client,
            )
            agent._llm_client = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_call_cogito_routes_through_mcp(self, mock_client, mock_reasoning_client, cogito_config):
        agent = self._make_agent(mock_client, cogito_config, reasoning_client=mock_reasoning_client)

        result = await agent._call_cogito("What is the nature of the bridge?")

        mock_reasoning_client.send_message.assert_called_once_with(
            "What is the nature of the bridge?", timeout=300.0,
        )
        assert "MCP bridge architecture" in result

    def test_get_reasoning_fn_cogito_mode(self, mock_client, mock_reasoning_client, cogito_config):
        agent = self._make_agent(mock_client, cogito_config, reasoning_client=mock_reasoning_client)

        fn = agent._get_reasoning_fn()
        assert fn == agent._call_cogito

    def test_get_reasoning_fn_llm_mode(self, mock_client, normal_config):
        agent = self._make_agent(mock_client, normal_config)

        fn = agent._get_reasoning_fn()
        assert fn == agent._call_llm

    @pytest.mark.asyncio
    async def test_cogito_opening_includes_system_context(
        self, mock_client, mock_reasoning_client, cogito_config
    ):
        agent = self._make_agent(mock_client, cogito_config, reasoning_client=mock_reasoning_client)

        captured_prompts = []

        async def capture(prompt):
            captured_prompts.append(prompt)
            return "Let me explore the architecture..."

        agent._call_cogito = AsyncMock(side_effect=capture)

        await agent._generate_opening_message()

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "Agentique" in prompt
        assert "improvement" in prompt.lower() or "IMPROVEMENT" in prompt
        assert "diff" in prompt.lower() or "fix" in prompt.lower() or "concrete" in prompt.lower()

    @pytest.mark.asyncio
    async def test_run_connects_reasoning_client(
        self, mock_client, mock_reasoning_client, cogito_config
    ):
        """Verify the harness (not agent) manages reasoning client lifecycle."""
        from agentique.simulation.app import SimulationHarness

        harness = SimulationHarness(mcp_url="http://localhost:8000/mcp")

        # Override start_simulation to check reasoning client creation
        cogito_config.use_cogito_reasoning = True

        # Verify the SimulationAgent accepts a reasoning client
        agent = self._make_agent(
            mock_client, cogito_config, reasoning_client=mock_reasoning_client
        )
        assert agent._reasoning_client is mock_reasoning_client
        assert agent._get_reasoning_fn() == agent._call_cogito

    @pytest.mark.asyncio
    async def test_call_cogito_falls_back_on_error(
        self, mock_client, mock_reasoning_client, cogito_config
    ):
        """Verify _call_cogito falls back to _call_llm when reasoning client fails."""
        mock_reasoning_client.send_message = AsyncMock(side_effect=ConnectionError("down"))

        agent = self._make_agent(mock_client, cogito_config, reasoning_client=mock_reasoning_client)

        mock_response = MagicMock()
        mock_response.text = "LLM fallback response"
        agent._llm_client.generate_content = MagicMock(return_value=mock_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_response):
            result = await agent._call_cogito("test prompt")

        assert result == "LLM fallback response"


# ---- Pause/Resume Event Tests ----


class TestPauseResumeEvents:
    @pytest.fixture
    def mock_client(self):
        return AsyncMock()

    @pytest.fixture
    def config(self):
        return SimulationConfig(
            persona=SimulationPersona(name="TestUser", typing_speed_ms=1, reading_speed_ms=1),
            objective=SimulationObjective(goal="Test", max_turns=2),
        )

    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def event_handler(self, events):
        async def handler(event):
            events.append(event)
        return handler

    def _make_agent(self, mock_client, config, event_handler):
        with patch.object(SimulationAgent, "_initialize_llm"):
            agent = SimulationAgent(
                client=mock_client,
                config=config,
                on_event=event_handler,
            )
            agent._llm_client = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_pause_emits_event(self, mock_client, config, events, event_handler):
        agent = self._make_agent(mock_client, config, event_handler)
        agent.conversation = [
            ConversationTurn(turn_number=1, role="human", message="Hi"),
            ConversationTurn(turn_number=1, role="agent", message="Hello!"),
        ]

        await agent.pause()

        assert len(events) == 1
        assert events[0].type == EventType.SIMULATION_PAUSED
        assert events[0].data["turns_completed"] == 1

    @pytest.mark.asyncio
    async def test_resume_emits_event(self, mock_client, config, events, event_handler):
        agent = self._make_agent(mock_client, config, event_handler)
        agent.conversation = [
            ConversationTurn(turn_number=1, role="human", message="Hi"),
            ConversationTurn(turn_number=1, role="agent", message="Hello!"),
            ConversationTurn(turn_number=2, role="human", message="More"),
            ConversationTurn(turn_number=2, role="agent", message="Sure!"),
        ]

        await agent.resume()

        assert len(events) == 1
        assert events[0].type == EventType.SIMULATION_RESUMED
        assert events[0].data["turns_completed"] == 2


class TestPauseResumeRESTEndpoints:
    @pytest.fixture
    def app(self):
        from agentique.simulation.app import create_app
        return create_app(mcp_url="http://localhost:8000/mcp")

    @pytest.fixture
    def client(self, app):
        from starlette.testclient import TestClient
        return TestClient(app)

    def test_pause_not_found(self, client):
        resp = client.post("/api/simulations/nonexistent/pause")
        assert resp.status_code == 404
        assert resp.json()["error"] == "Simulation not found"

    def test_resume_not_found(self, client):
        resp = client.post("/api/simulations/nonexistent/resume")
        assert resp.status_code == 404
        assert resp.json()["error"] == "Simulation not found"
