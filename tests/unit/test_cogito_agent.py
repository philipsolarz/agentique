"""Unit tests for the Cogito reasoning agent hierarchy."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestBuildCogitoAgent:
    def _build(self):
        from adk_test_agent.cogito_agent import build_cogito_agent

        with patch("google.adk.Agent") as MockAgent:
            # Make the Agent constructor return a mock with the kwargs as attributes
            def make_agent(**kwargs):
                from unittest.mock import MagicMock
                agent = MagicMock()
                agent.name = kwargs.get("name", "")
                agent.description = kwargs.get("description", "")
                agent.instruction = kwargs.get("instruction", "")
                agent.tools = kwargs.get("tools", [])
                agent.sub_agents = kwargs.get("sub_agents", [])
                agent.model = kwargs.get("model", "")
                return agent

            MockAgent.side_effect = make_agent
            return build_cogito_agent()

    def test_has_six_sub_agents(self):
        root = self._build()
        assert len(root.sub_agents) == 6

    def test_build_cogito_agent_returns_agent(self):
        root = self._build()
        assert root.name == "CogitoPrime"

    def test_sub_agent_names(self):
        root = self._build()
        names = [sa.name for sa in root.sub_agents]
        assert names == ["Analyst", "Creative", "Critic", "Synthesizer", "MetaReasoner", "CodebaseExplorer"]

    def test_analyst_tools(self):
        root = self._build()
        analyst = root.sub_agents[0]
        assert analyst.name == "Analyst"
        tool_names = [t.__name__ for t in analyst.tools]
        assert tool_names == ["analyze_claim", "evaluate_evidence", "chain_of_reasoning"]

    def test_creative_tools(self):
        root = self._build()
        creative = root.sub_agents[1]
        assert creative.name == "Creative"
        tool_names = [t.__name__ for t in creative.tools]
        assert tool_names == ["generate_analogies", "thought_experiment", "lateral_perspective"]

    def test_critic_tools(self):
        root = self._build()
        critic = root.sub_agents[2]
        assert critic.name == "Critic"
        tool_names = [t.__name__ for t in critic.tools]
        assert tool_names == ["find_counterarguments", "detect_fallacies", "steelman"]

    def test_synthesizer_tools(self):
        root = self._build()
        synth = root.sub_agents[3]
        assert synth.name == "Synthesizer"
        tool_names = [t.__name__ for t in synth.tools]
        assert tool_names == ["synthesize_perspectives", "find_common_ground", "build_framework"]

    def test_meta_reasoner_tools(self):
        root = self._build()
        meta = root.sub_agents[4]
        assert meta.name == "MetaReasoner"
        tool_names = [t.__name__ for t in meta.tools]
        assert tool_names == ["evaluate_reasoning_quality", "detect_biases", "reflect_on_process"]

    def test_cogito_prime_instruction_references_sub_agents(self):
        root = self._build()
        instruction = root.instruction
        for name in ["Analyst", "Creative", "Critic", "Synthesizer", "MetaReasoner", "CodebaseExplorer"]:
            assert name in instruction, f"Instruction should reference {name}"

    def test_codebase_explorer_sub_agent(self):
        root = self._build()
        explorer = root.sub_agents[5]
        assert explorer.name == "CodebaseExplorer"
        tool_names = [t.__name__ for t in explorer.tools]
        assert tool_names == ["list_source_files", "read_source_file", "search_code"]


class TestToolReturnStructures:
    """Test that each tool returns a dict with the expected common fields."""

    def test_analyze_claim_structure(self):
        from adk_test_agent.cogito_agent import analyze_claim
        result = analyze_claim("The sky is blue")
        assert "reasoning_id" in result
        assert result["agent"] == "Analyst"
        assert "timestamp" in result
        assert result["tool"] == "analyze_claim"
        assert result["claim"] == "The sky is blue"

    def test_evaluate_evidence_structure(self):
        from adk_test_agent.cogito_agent import evaluate_evidence
        result = evaluate_evidence("Spectral data", "Sky is blue")
        assert result["agent"] == "Analyst"
        assert result["tool"] == "evaluate_evidence"
        assert result["evidence"] == "Spectral data"

    def test_chain_of_reasoning_structure(self):
        from adk_test_agent.cogito_agent import chain_of_reasoning
        result = chain_of_reasoning(["P1", "P2"])
        assert result["agent"] == "Analyst"
        assert result["premises"] == ["P1", "P2"]

    def test_generate_analogies_structure(self):
        from adk_test_agent.cogito_agent import generate_analogies
        result = generate_analogies("recursion")
        assert result["agent"] == "Creative"
        assert result["concept"] == "recursion"

    def test_thought_experiment_structure(self):
        from adk_test_agent.cogito_agent import thought_experiment
        result = thought_experiment("What if agents could modify their own code?")
        assert result["agent"] == "Creative"

    def test_lateral_perspective_structure(self):
        from adk_test_agent.cogito_agent import lateral_perspective
        result = lateral_perspective("Protocol bridging latency")
        assert result["agent"] == "Creative"

    def test_find_counterarguments_structure(self):
        from adk_test_agent.cogito_agent import find_counterarguments
        result = find_counterarguments("MCP is the best protocol")
        assert result["agent"] == "Critic"

    def test_detect_fallacies_structure(self):
        from adk_test_agent.cogito_agent import detect_fallacies
        result = detect_fallacies("It works, therefore it's correct")
        assert result["agent"] == "Critic"

    def test_steelman_structure(self):
        from adk_test_agent.cogito_agent import steelman
        result = steelman("Direct LLM calls are better than MCP")
        assert result["agent"] == "Critic"
        assert result["original_position"] == "Direct LLM calls are better than MCP"

    def test_synthesize_perspectives_structure(self):
        from adk_test_agent.cogito_agent import synthesize_perspectives
        result = synthesize_perspectives(["View A", "View B"])
        assert result["agent"] == "Synthesizer"
        assert result["viewpoints"] == ["View A", "View B"]

    def test_find_common_ground_structure(self):
        from adk_test_agent.cogito_agent import find_common_ground
        result = find_common_ground(["Pos 1", "Pos 2"])
        assert result["agent"] == "Synthesizer"

    def test_build_framework_structure(self):
        from adk_test_agent.cogito_agent import build_framework
        result = build_framework(["concept1", "concept2"])
        assert result["agent"] == "Synthesizer"

    def test_evaluate_reasoning_quality_structure(self):
        from adk_test_agent.cogito_agent import evaluate_reasoning_quality
        result = evaluate_reasoning_quality("A therefore B therefore C")
        assert result["agent"] == "MetaReasoner"
        assert result["validity_score"] == 0.0

    def test_detect_biases_structure(self):
        from adk_test_agent.cogito_agent import detect_biases
        result = detect_biases("This confirms what I expected")
        assert result["agent"] == "MetaReasoner"

    def test_reflect_on_process_structure(self):
        from adk_test_agent.cogito_agent import reflect_on_process
        result = reflect_on_process("We discussed X then Y")
        assert result["agent"] == "MetaReasoner"
        assert result["conversation"] == "We discussed X then Y"


class TestCodebaseExplorerTools:
    """Test CodebaseExplorer tools with a temporary directory as codebase root."""

    def test_list_source_files_tool(self, tmp_path):
        # Create test files
        (tmp_path / "server.py").write_text("# server\nprint('hello')\n")
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "agent.py").write_text("class Agent:\n    pass\n")

        with patch("adk_test_agent.cogito_agent._get_codebase_root", return_value=tmp_path):
            from adk_test_agent.cogito_agent import list_source_files
            result = list_source_files("")
            assert result["agent"] == "CodebaseExplorer"
            assert result["tool"] == "list_source_files"
            names = [e["name"] for e in result["entries"]]
            assert "server.py" in names
            assert "core" in names

    def test_list_source_files_missing_dir(self, tmp_path):
        with patch("adk_test_agent.cogito_agent._get_codebase_root", return_value=tmp_path):
            from adk_test_agent.cogito_agent import list_source_files
            result = list_source_files("nonexistent")
            assert "error" in result
            assert result["entries"] == []

    def test_read_source_file_tool(self, tmp_path):
        (tmp_path / "test.py").write_text("line1\nline2\nline3\n")

        with patch("adk_test_agent.cogito_agent._get_codebase_root", return_value=tmp_path):
            from adk_test_agent.cogito_agent import read_source_file
            result = read_source_file("test.py")
            assert result["agent"] == "CodebaseExplorer"
            assert result["tool"] == "read_source_file"
            assert result["total_lines"] == 3
            assert "line1" in result["content"]
            assert "line2" in result["content"]

    def test_read_source_file_missing(self, tmp_path):
        with patch("adk_test_agent.cogito_agent._get_codebase_root", return_value=tmp_path):
            from adk_test_agent.cogito_agent import read_source_file
            result = read_source_file("nonexistent.py")
            assert "error" in result

    def test_read_source_file_line_range(self, tmp_path):
        (tmp_path / "test.py").write_text("a\nb\nc\nd\ne\n")

        with patch("adk_test_agent.cogito_agent._get_codebase_root", return_value=tmp_path):
            from adk_test_agent.cogito_agent import read_source_file
            result = read_source_file("test.py", start_line=2, end_line=4)
            assert result["range"] == "2-4"
            assert "b" in result["content"]
            assert "d" in result["content"]

    def test_search_code_tool(self, tmp_path):
        (tmp_path / "server.py").write_text("class MyServer:\n    def handle(self):\n        pass\n")
        (tmp_path / "utils.py").write_text("def helper():\n    pass\n")

        with patch("adk_test_agent.cogito_agent._get_codebase_root", return_value=tmp_path):
            from adk_test_agent.cogito_agent import search_code
            result = search_code("class.*Server")
            assert result["agent"] == "CodebaseExplorer"
            assert result["tool"] == "search_code"
            assert result["total_matches"] >= 1
            assert any("MyServer" in m["text"] for m in result["matches"])

    def test_search_code_invalid_regex(self, tmp_path):
        with patch("adk_test_agent.cogito_agent._get_codebase_root", return_value=tmp_path):
            from adk_test_agent.cogito_agent import search_code
            result = search_code("[invalid")
            assert "error" in result
            assert "Invalid regex" in result["error"]

    def test_path_traversal_protection(self, tmp_path):
        with patch("adk_test_agent.cogito_agent._get_codebase_root", return_value=tmp_path):
            from adk_test_agent.cogito_agent import list_source_files, read_source_file
            result = list_source_files("../../etc")
            assert "error" in result
            assert "traversal" in result["error"].lower()

            result = read_source_file("../../etc/passwd")
            assert "error" in result
            assert "traversal" in result["error"].lower()

    def test_codebase_root_env_var(self):
        from adk_test_agent.cogito_agent import _get_codebase_root
        with patch.dict("os.environ", {"CODEBASE_ROOT": "/custom/path"}):
            from pathlib import Path
            root = _get_codebase_root()
            assert root == Path("/custom/path")

        # Default when not set
        with patch.dict("os.environ", {}, clear=True):
            root = _get_codebase_root()
            assert root == Path("/app/agentique-src")
