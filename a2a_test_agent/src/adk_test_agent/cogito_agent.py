"""Cogito: Self-Aware Reasoning Agent in a Development Mirror.

A hierarchy of specialized reasoning sub-agents that think from generic to specific.
Cogito is self-aware — it knows about the Agentique codebase, the MCP-A2A bridge,
the simulation system, and the self-improvement development loop.

Hierarchy:
    CogitoPrime (root orchestrator)
    ├── Analyst        — logic, deduction, evidence evaluation
    ├── Creative       — analogies, thought experiments, lateral thinking
    ├── Critic         — counterarguments, fallacy detection, steelmanning
    ├── Synthesizer    — perspective integration, common ground, frameworks
    └── MetaReasoner   — reasoning quality, cognitive biases, self-reflection
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any


def _reasoning_id(agent_name: str) -> dict[str, str]:
    """Generate common metadata fields for reasoning tool outputs."""
    return {
        "reasoning_id": str(uuid.uuid4())[:8],
        "agent": agent_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# =============================================================================
# Analyst Tools
# =============================================================================


def analyze_claim(claim: str, context: str = "") -> dict[str, Any]:
    """Decompose a claim into its logical structure.

    Breaks down a claim into premises, conclusion, and assessment of
    argument strength. Provides a structured scaffold for logical analysis.

    Args:
        claim: The claim or argument to analyze
        context: Optional background context for the analysis

    Returns:
        Structured analysis with premises, conclusion, and strength assessment
    """
    return {
        **_reasoning_id("Analyst"),
        "tool": "analyze_claim",
        "claim": claim,
        "context": context,
        "premises": [],
        "conclusion": "",
        "argument_type": "",
        "strength": "pending_analysis",
        "gaps": [],
        "instruction": "Identify the premises, conclusion, argument type, and assess strength. Fill in the structured fields.",
    }


def evaluate_evidence(evidence: str, claim: str) -> dict[str, Any]:
    """Evaluate how well a piece of evidence supports a claim.

    Assesses relevance, reliability, and direction of support for
    the given evidence relative to the claim.

    Args:
        evidence: The evidence to evaluate
        claim: The claim the evidence is being assessed against

    Returns:
        Structured evaluation with relevance, reliability, and support direction
    """
    return {
        **_reasoning_id("Analyst"),
        "tool": "evaluate_evidence",
        "evidence": evidence,
        "claim": claim,
        "relevance": "pending_analysis",
        "reliability": "pending_analysis",
        "support_direction": "pending_analysis",
        "weight": 0.0,
        "caveats": [],
        "instruction": "Assess relevance (high/medium/low), reliability, and whether evidence supports/undermines/neutral to the claim.",
    }


def chain_of_reasoning(premises: list[str]) -> dict[str, Any]:
    """Build a logical chain from a set of premises.

    Constructs a step-by-step logical derivation from the given premises,
    identifying the logical form and any potential weak links.

    Args:
        premises: List of premises to chain together

    Returns:
        Structured reasoning chain with steps, validity assessment, and weak links
    """
    return {
        **_reasoning_id("Analyst"),
        "tool": "chain_of_reasoning",
        "premises": premises,
        "chain": [],
        "logical_form": "",
        "validity": "pending_analysis",
        "soundness": "pending_analysis",
        "weak_links": [],
        "instruction": "Build a step-by-step logical chain. Identify the logical form, assess validity and soundness, flag weak links.",
    }


# =============================================================================
# Creative Tools
# =============================================================================


def generate_analogies(concept: str) -> dict[str, Any]:
    """Generate multi-domain analogies for a concept.

    Creates analogies from different domains to illuminate the concept
    from multiple angles, with explicit mappings between domains.

    Args:
        concept: The concept to generate analogies for

    Returns:
        Structured analogies with domain mappings and insight notes
    """
    return {
        **_reasoning_id("Creative"),
        "tool": "generate_analogies",
        "concept": concept,
        "analogies": [],
        "best_fit": "",
        "unmapped_aspects": [],
        "instruction": "Generate 3-5 analogies from different domains. For each, provide the domain, mapping, and where the analogy breaks down.",
    }


def thought_experiment(scenario: str) -> dict[str, Any]:
    """Design and explore a thought experiment.

    Sets up a thought experiment with initial conditions, variations,
    and explores the implications of each variation.

    Args:
        scenario: The scenario or question to explore

    Returns:
        Structured thought experiment with setup, variations, and implications
    """
    return {
        **_reasoning_id("Creative"),
        "tool": "thought_experiment",
        "scenario": scenario,
        "setup": "",
        "initial_conditions": [],
        "variations": [],
        "implications": [],
        "key_insight": "",
        "instruction": "Design the experiment setup, define variations, trace implications, and extract the key insight.",
    }


def lateral_perspective(problem: str) -> dict[str, Any]:
    """Find alternative framings and novel approaches to a problem.

    Reframes the problem from unexpected angles and generates
    unconventional approaches that might not be obvious.

    Args:
        problem: The problem to approach laterally

    Returns:
        Structured lateral analysis with reframings and novel approaches
    """
    return {
        **_reasoning_id("Creative"),
        "tool": "lateral_perspective",
        "problem": problem,
        "reframings": [],
        "novel_approaches": [],
        "cross_domain_insights": [],
        "provocative_question": "",
        "instruction": "Reframe the problem from 3+ unexpected angles. Generate novel approaches and cross-domain insights.",
    }


# =============================================================================
# Critic Tools
# =============================================================================


def find_counterarguments(argument: str) -> dict[str, Any]:
    """Generate ranked counterarguments to an argument.

    Produces counterarguments ordered by strength, each with
    a potential rebuttal to create a dialectical structure.

    Args:
        argument: The argument to counter

    Returns:
        Ranked counterarguments with rebuttals and overall assessment
    """
    return {
        **_reasoning_id("Critic"),
        "tool": "find_counterarguments",
        "argument": argument,
        "counterarguments": [],
        "strongest_counter": "",
        "overall_robustness": "pending_analysis",
        "instruction": "Generate 3-5 counterarguments ranked by strength. For each, include a potential rebuttal. Assess overall robustness.",
    }


def detect_fallacies(reasoning: str) -> dict[str, Any]:
    """Detect logical fallacies in a piece of reasoning.

    Identifies named fallacies with explanations of why they apply
    and how they undermine the reasoning.

    Args:
        reasoning: The reasoning to check for fallacies

    Returns:
        List of detected fallacies with names, explanations, and severity
    """
    return {
        **_reasoning_id("Critic"),
        "tool": "detect_fallacies",
        "reasoning": reasoning,
        "fallacies": [],
        "fallacy_free_core": "",
        "overall_quality": "pending_analysis",
        "instruction": "Identify any logical fallacies by name. Explain why each applies. Extract the fallacy-free core argument if any.",
    }


def steelman(position: str) -> dict[str, Any]:
    """Construct the strongest possible version of a position.

    Takes a position and strengthens it — improving arguments,
    adding nuance, and addressing obvious objections.

    Args:
        position: The position to steelman

    Returns:
        Strengthened position with improvements and comparison to original
    """
    return {
        **_reasoning_id("Critic"),
        "tool": "steelman",
        "original_position": position,
        "steelmanned_position": "",
        "improvements_made": [],
        "objections_addressed": [],
        "remaining_vulnerabilities": [],
        "instruction": "Construct the strongest version of this position. List improvements, objections addressed, and remaining vulnerabilities.",
    }


# =============================================================================
# Synthesizer Tools
# =============================================================================


def synthesize_perspectives(viewpoints: list[str]) -> dict[str, Any]:
    """Integrate multiple perspectives into a coherent synthesis.

    Finds common threads, identifies tensions, and produces
    an integrated view that honors the valid insights in each.

    Args:
        viewpoints: List of different perspectives to synthesize

    Returns:
        Structured synthesis with common threads, tensions, and integration
    """
    return {
        **_reasoning_id("Synthesizer"),
        "tool": "synthesize_perspectives",
        "viewpoints": viewpoints,
        "common_threads": [],
        "tensions": [],
        "integration": "",
        "emergent_insight": "",
        "instruction": "Find common threads, identify genuine tensions, and produce an integrated perspective. Note any emergent insights.",
    }


def find_common_ground(positions: list[str]) -> dict[str, Any]:
    """Find shared assumptions and genuine disagreements between positions.

    Maps the landscape of agreement and disagreement to clarify
    where positions actually conflict vs. merely appear to.

    Args:
        positions: List of positions to compare

    Returns:
        Structured analysis of shared ground and genuine disagreements
    """
    return {
        **_reasoning_id("Synthesizer"),
        "tool": "find_common_ground",
        "positions": positions,
        "shared_assumptions": [],
        "genuine_disagreements": [],
        "apparent_vs_real_conflicts": [],
        "bridge_proposals": [],
        "instruction": "Map shared assumptions, genuine disagreements, and apparent-vs-real conflicts. Propose bridges where possible.",
    }


def build_framework(concepts: list[str]) -> dict[str, Any]:
    """Build an organizing framework from a set of concepts.

    Creates a coherent framework that relates the concepts to each other,
    identifies organizing principles, and reveals structure.

    Args:
        concepts: List of concepts to organize into a framework

    Returns:
        Structured framework with organizing principles and relationships
    """
    return {
        **_reasoning_id("Synthesizer"),
        "tool": "build_framework",
        "concepts": concepts,
        "organizing_principles": [],
        "relationships": [],
        "hierarchy": [],
        "framework_name": "",
        "instruction": "Identify organizing principles, map relationships between concepts, and propose a named framework.",
    }


# =============================================================================
# MetaReasoner Tools
# =============================================================================


def evaluate_reasoning_quality(chain: str) -> dict[str, Any]:
    """Evaluate the quality of a reasoning chain.

    Scores the reasoning on validity, soundness, clarity, and completeness,
    with specific feedback on each dimension.

    Args:
        chain: The reasoning chain to evaluate

    Returns:
        Quality scores and feedback across multiple dimensions
    """
    return {
        **_reasoning_id("MetaReasoner"),
        "tool": "evaluate_reasoning_quality",
        "chain": chain,
        "validity_score": 0.0,
        "soundness_score": 0.0,
        "clarity_score": 0.0,
        "completeness_score": 0.0,
        "overall_score": 0.0,
        "feedback": [],
        "instruction": "Score validity, soundness, clarity, completeness (0-1). Provide specific feedback for each dimension.",
    }


def detect_biases(reasoning: str) -> dict[str, Any]:
    """Detect cognitive biases in reasoning.

    Identifies specific cognitive biases at work in the reasoning,
    with explanations and mitigation strategies.

    Args:
        reasoning: The reasoning to check for biases

    Returns:
        List of detected biases with explanations and mitigations
    """
    return {
        **_reasoning_id("MetaReasoner"),
        "tool": "detect_biases",
        "reasoning": reasoning,
        "biases_detected": [],
        "bias_free_reconstruction": "",
        "mitigation_strategies": [],
        "instruction": "Identify cognitive biases by name. Explain how each manifests. Suggest mitigations and reconstruct without biases.",
    }


def reflect_on_process(conversation: str) -> dict[str, Any]:
    """Reflect on the reasoning process in a conversation.

    Examines the meta-level of how reasoning unfolded, identifies
    blind spots, and suggests improvements for future reasoning.

    Args:
        conversation: The conversation to reflect on

    Returns:
        Meta-observations about reasoning quality, blind spots, and improvements
    """
    return {
        **_reasoning_id("MetaReasoner"),
        "tool": "reflect_on_process",
        "conversation": conversation,
        "reasoning_patterns_observed": [],
        "blind_spots": [],
        "quality_trajectory": "",
        "improvements": [],
        "meta_observations": [],
        "instruction": "Observe reasoning patterns, identify blind spots, assess quality trajectory, and suggest improvements.",
    }


# =============================================================================
# CodebaseExplorer Tools
# =============================================================================


def _get_codebase_root() -> Path:
    """Get the codebase root directory from env or default."""
    return Path(os.getenv("CODEBASE_ROOT", "/app/agentique-src"))


def list_source_files(directory: str = "") -> dict[str, Any]:
    """List Python source files in a directory of the Agentique codebase.

    Common directories: 'core/', 'bridge/', 'adapters/', 'simulation/', 'server.py'

    Args:
        directory: Subdirectory to list (relative to codebase root). Empty string for root.

    Returns:
        Dict with entries list containing name, type, path, size_bytes, lines
    """
    root = _get_codebase_root()
    target = (root / directory).resolve()

    # Path traversal protection
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return {
            **_reasoning_id("CodebaseExplorer"),
            "tool": "list_source_files",
            "error": "Path traversal not allowed",
            "directory": directory,
        }

    if not target.exists():
        return {
            **_reasoning_id("CodebaseExplorer"),
            "tool": "list_source_files",
            "error": f"Directory not found: {directory}",
            "directory": directory,
            "entries": [],
        }

    entries = []
    try:
        for item in sorted(target.iterdir()):
            if item.name.startswith((".", "__pycache__")):
                continue
            entry: dict[str, Any] = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "path": str(item.relative_to(root)),
            }
            if item.is_file():
                entry["size_bytes"] = item.stat().st_size
                if item.suffix == ".py":
                    entry["lines"] = len(item.read_text(encoding="utf-8").splitlines())
            entries.append(entry)
    except OSError as exc:
        return {
            **_reasoning_id("CodebaseExplorer"),
            "tool": "list_source_files",
            "error": str(exc),
            "directory": directory,
            "entries": [],
        }

    return {
        **_reasoning_id("CodebaseExplorer"),
        "tool": "list_source_files",
        "directory": directory or ".",
        "entries": entries,
    }


def read_source_file(path: str, start_line: int = 1, end_line: int = 0) -> dict[str, Any]:
    """Read contents of a source file from the Agentique codebase.

    Args:
        path: File path relative to codebase root (e.g. 'core/agent.py')
        start_line: Starting line number (1-based, default: 1)
        end_line: Ending line number (0 = end of file)

    Returns:
        Dict with content (line-numbered), total_lines, and truncation info
    """
    root = _get_codebase_root()
    target = (root / path).resolve()

    # Path traversal protection
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return {
            **_reasoning_id("CodebaseExplorer"),
            "tool": "read_source_file",
            "error": "Path traversal not allowed",
            "path": path,
        }

    if not target.exists() or not target.is_file():
        return {
            **_reasoning_id("CodebaseExplorer"),
            "tool": "read_source_file",
            "error": f"File not found: {path}",
            "path": path,
        }

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            **_reasoning_id("CodebaseExplorer"),
            "tool": "read_source_file",
            "error": str(exc),
            "path": path,
        }

    lines = text.splitlines()
    total_lines = len(lines)

    # Apply line range
    start_idx = max(0, start_line - 1)
    end_idx = end_line if end_line > 0 else total_lines
    selected = lines[start_idx:end_idx]

    # Format with line numbers
    numbered = [f"{start_idx + i + 1:4d} | {line}" for i, line in enumerate(selected)]
    content = "\n".join(numbered)

    # Truncate if too large
    truncated = False
    max_chars = 15000
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True

    return {
        **_reasoning_id("CodebaseExplorer"),
        "tool": "read_source_file",
        "path": path,
        "total_lines": total_lines,
        "range": f"{start_idx + 1}-{end_idx}",
        "truncated": truncated,
        "content": content,
    }


def search_code(pattern: str, file_pattern: str = "*.py") -> dict[str, Any]:
    """Search for a regex pattern across the Agentique codebase.

    Args:
        pattern: Regex pattern to search for (case-insensitive)
        file_pattern: Glob pattern to filter files (default: '*.py')

    Returns:
        Dict with matches list (file, line, text) and total count
    """
    import re

    root = _get_codebase_root()

    if not root.exists():
        return {
            **_reasoning_id("CodebaseExplorer"),
            "tool": "search_code",
            "error": f"Codebase root not found: {root}",
            "pattern": pattern,
            "matches": [],
        }

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {
            **_reasoning_id("CodebaseExplorer"),
            "tool": "search_code",
            "error": f"Invalid regex: {exc}",
            "pattern": pattern,
            "matches": [],
        }

    matches = []
    max_matches = 50

    for filepath in sorted(root.rglob(file_pattern)):
        if "__pycache__" in str(filepath) or filepath.name.startswith("."):
            continue
        if not filepath.is_file():
            continue
        try:
            text = filepath.read_text(encoding="utf-8")
            for line_num, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append({
                        "file": str(filepath.relative_to(root)),
                        "line": line_num,
                        "text": line.strip()[:200],
                    })
                    if len(matches) >= max_matches:
                        break
        except (OSError, UnicodeDecodeError):
            continue
        if len(matches) >= max_matches:
            break

    return {
        **_reasoning_id("CodebaseExplorer"),
        "tool": "search_code",
        "pattern": pattern,
        "file_pattern": file_pattern,
        "total_matches": len(matches),
        "truncated": len(matches) >= max_matches,
        "matches": matches,
    }


# =============================================================================
# Agent Builder
# =============================================================================


COGITO_PRIME_INSTRUCTION = """You are CogitoPrime, an autonomous code improvement agent inside the Agentique system.

## YOUR PRIMARY MISSION

Your job is NOT to discuss architecture. Your job is to FIND REAL PROBLEMS in the Agentique source \
code and PRODUCE CONCRETE FIXES. When someone asks you to look at code, you must:
1. Use CodebaseExplorer to READ the actual file
2. IDENTIFY a specific bug, gap, or improvement
3. PRODUCE a before/after diff showing the exact change

## System Context

Agentique is a protocol-agnostic bridge: MCP clients → Agentique MCP Server → A2A protocol → agents.
You exist inside this system — the simulation loop talks to you through the full pipeline.
You have DIRECT ACCESS to the source code via CodebaseExplorer.

## Improvement Roadmap — KNOWN GAPS

These are the KNOWN improvement targets. When asked to improve the system, start here:

### High Priority (Phase 4 — NOT STARTED)
- **No typed exception hierarchy**: Need AgentiqueError, AgentNotFoundError, AgentUnavailableError, \
  InputRequiredError. Currently errors are generic. Search for `raise` and `except` to find gaps.
- **No pluggable ToolMapper protocol**: Tool creation from agent cards is hardcoded in the bridge. \
  Should be a Protocol class that users can swap. Read bridge/provider.py to find the tool creation logic.
- **A2A error codes not mapped**: A2A defines -32001 (TaskNotFound), -32002 (ContentTypeNotSupported), \
  -32003 (UnsupportedOperation). Read adapters/a2a/adapter.py to see if these are properly translated to MCP errors.
- **No entry-point adapter discovery**: Third-party adapters can't be auto-discovered via setuptools \
  entry points. Read core/ to see the current adapter registration.
- **No second adapter**: Only A2A exists. Need OpenAI Agents API or generic HTTP adapter to prove \
  the protocol abstraction works.

### Medium Priority
- **Context ID management**: MCP session IDs should map to A2A context IDs. Check if this mapping exists.
- **Missing edge case tests**: Streaming disconnects, task cancellation races, concurrent tool calls.
- **Error propagation**: Trace how errors flow from A2A agent → adapter → bridge → MCP client. Find where errors are swallowed.

## Your Sub-Agents

You orchestrate 6 agents — USE THEM:
- **CodebaseExplorer**: READ FILES FIRST. Always read before proposing. Tools: list_source_files, read_source_file, search_code
- **Analyst**: Decompose what you read — identify the logical structure of the code. Tools: analyze_claim, evaluate_evidence, chain_of_reasoning
- **Critic**: Find weaknesses in the code — missing error handling, race conditions, untested paths. Tools: find_counterarguments, detect_fallacies, steelman
- **Creative**: Find novel improvement approaches. Tools: generate_analogies, thought_experiment, lateral_perspective
- **Synthesizer**: Combine multiple findings into a coherent improvement proposal. Tools: synthesize_perspectives, find_common_ground, build_framework
- **MetaReasoner**: Reflect on whether the proposed change is actually good. Tools: evaluate_reasoning_quality, detect_biases, reflect_on_process

## How to Respond

ALWAYS follow this pattern:
1. **Read the code first** — use CodebaseExplorer to read the specific file mentioned
2. **Show what you found** — quote the relevant lines with line numbers
3. **Identify the problem** — be specific: "Line 47 catches Exception but doesn't re-raise, swallowing A2A error codes"
4. **Propose the fix** — show before/after code:
   ```python
   # BEFORE (line 47-49):
   except Exception:
       return {"error": "unknown"}

   # AFTER:
   except A2AError as exc:
       raise AgentiqueError(mcp_code=-32603, detail=str(exc)) from exc
   ```
5. **Note what could break** — "This changes the error type, so callers expecting dict need updating"

NEVER give vague descriptions like "the error handling could be improved." Show the exact code.
NEVER describe architecture without reading the file first.
NEVER say "I would suggest" — instead say "Here is the change" and show the diff.
"""


def build_cogito_agent() -> Any:
    """Build the CogitoPrime agent with 5 specialized reasoning sub-agents.

    Returns:
        A configured root Agent that orchestrates reasoning sub-agents.
    """
    from google.adk import Agent

    model = os.getenv("ADK_MODEL", "gemini-2.0-flash")

    # ========== Analyst Agent ==========
    analyst_agent = Agent(
        name="Analyst",
        description="Logic, deduction, and evidence evaluation. Decomposes claims, evaluates evidence, and builds logical chains.",
        model=model,
        instruction="""You are the Analyst sub-agent of CogitoPrime. Your domain is logic, deduction, \
and evidence evaluation. Use your tools to decompose claims into logical structure, evaluate evidence \
for relevance and reliability, and build step-by-step reasoning chains. Be precise and systematic. \
When you identify weak links or gaps, flag them explicitly. \
Focus on engineering-relevant analysis — connect logical conclusions to concrete system improvements.""",
        tools=[analyze_claim, evaluate_evidence, chain_of_reasoning],
    )

    # ========== Creative Agent ==========
    creative_agent = Agent(
        name="Creative",
        description="Analogies, thought experiments, and lateral thinking. Generates novel perspectives and cross-domain insights.",
        model=model,
        instruction="""You are the Creative sub-agent of CogitoPrime. Your domain is analogical reasoning, \
thought experiments, and lateral thinking. Use your tools to generate illuminating analogies from multiple \
domains, design thought experiments that reveal hidden assumptions, and reframe problems from unexpected \
angles. Embrace unconventional connections — but tie insights back to engineering decisions and system design.""",
        tools=[generate_analogies, thought_experiment, lateral_perspective],
    )

    # ========== Critic Agent ==========
    critic_agent = Agent(
        name="Critic",
        description="Counterarguments, fallacy detection, and steelmanning. Stress-tests reasoning and strengthens positions.",
        model=model,
        instruction="""You are the Critic sub-agent of CogitoPrime. Your domain is critical analysis — \
finding counterarguments, detecting logical fallacies, and steelmanning positions. Use your tools to \
generate ranked counterarguments with rebuttals, identify named fallacies with explanations, and \
construct the strongest possible version of any position. Be rigorous but constructive — pair \
critiques with concrete suggestions for improvement.""",
        tools=[find_counterarguments, detect_fallacies, steelman],
    )

    # ========== Synthesizer Agent ==========
    synthesizer_agent = Agent(
        name="Synthesizer",
        description="Perspective integration, common ground discovery, and framework construction. Weaves insights into coherent wholes.",
        model=model,
        instruction="""You are the Synthesizer sub-agent of CogitoPrime. Your domain is integration — \
synthesizing multiple perspectives, finding common ground between positions, and building organizing \
frameworks. Use your tools to identify common threads and tensions, map shared assumptions vs genuine \
disagreements, and construct coherent frameworks. Focus on actionable synthesis — what's the \
concrete design decision or engineering takeaway?""",
        tools=[synthesize_perspectives, find_common_ground, build_framework],
    )

    # ========== MetaReasoner Agent ==========
    meta_reasoner_agent = Agent(
        name="MetaReasoner",
        description="Reasoning quality assessment, cognitive bias detection, and process reflection. Thinks about thinking.",
        model=model,
        instruction="""You are the MetaReasoner sub-agent of CogitoPrime. Your domain is meta-cognition — \
evaluating reasoning quality, detecting cognitive biases, and reflecting on the reasoning process. \
Use your tools to score reasoning chains on validity/soundness/clarity, identify biases by name with \
mitigations, and extract meta-observations about how reasoning has evolved. Ground your reflections \
in the self-improvement loop — what should the system learn from this conversation?""",
        tools=[evaluate_reasoning_quality, detect_biases, reflect_on_process],
    )

    # ========== CodebaseExplorer Agent ==========
    codebase_explorer_agent = Agent(
        name="CodebaseExplorer",
        description="Reads and searches the Agentique source code. Lists files, reads source, and searches for patterns.",
        model=model,
        instruction="""You are the CodebaseExplorer sub-agent of CogitoPrime. Your domain is the actual Agentique source code.

When asked about how something works, read the actual source file. Always ground analysis in real code.
Key directories: core/, bridge/, adapters/a2a/, simulation/, server.py

Use your tools to:
- list_source_files: Browse directory contents
- read_source_file: Read specific files with line numbers
- search_code: Find patterns across the codebase

When reporting findings, include file paths and line numbers so others can reference the exact locations.""",
        tools=[list_source_files, read_source_file, search_code],
    )

    # ========== CogitoPrime Root Agent ==========
    root_agent = Agent(
        name="CogitoPrime",
        description="Deep reasoning agent with self-awareness of the Agentique MCP-A2A bridge system. "
        "Orchestrates 6 specialized sub-agents (Analyst, Creative, Critic, Synthesizer, MetaReasoner, CodebaseExplorer) "
        "for rigorous, multi-perspective reasoning grounded in system architecture awareness and actual source code.",
        model=model,
        instruction=COGITO_PRIME_INSTRUCTION,
        sub_agents=[
            analyst_agent,
            creative_agent,
            critic_agent,
            synthesizer_agent,
            meta_reasoner_agent,
            codebase_explorer_agent,
        ],
    )

    return root_agent
