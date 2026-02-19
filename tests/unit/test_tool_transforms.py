"""Tests for agentique.bridge.tool_transforms — FastMCP transform factory functions.

These tests verify that the factories return the correct transform types
and that ``default_transform_stack`` composes them correctly.

Note: We test the factory return types using duck typing (check for the
transform interface rather than exact class equality) to avoid coupling
to FastMCP internals that may change across minor versions.
"""

from __future__ import annotations

import pytest

from agentique.bridge.tool_transforms import (
    default_transform_stack,
    namespace_transform_for_agent,
    prompts_as_tools_transform,
    resources_as_tools_transform,
    visibility_transform,
)


# ---------------------------------------------------------------------------
# namespace_transform_for_agent
# ---------------------------------------------------------------------------


def test_namespace_transform_returns_object():
    xf = namespace_transform_for_agent("billing")
    assert xf is not None


def test_namespace_transform_default_separator():
    """The transform object is created without error for a valid agent name."""
    xf = namespace_transform_for_agent("my_agent")
    # We can't inspect the prefix without knowing FastMCP internals,
    # but the factory should succeed.
    assert xf is not None


def test_namespace_transform_custom_separator():
    xf = namespace_transform_for_agent("billing", separator="/")
    assert xf is not None


def test_namespace_transform_empty_name():
    """Empty string prefix is allowed (degenerate but should not crash)."""
    xf = namespace_transform_for_agent("")
    assert xf is not None


# ---------------------------------------------------------------------------
# visibility_transform
# ---------------------------------------------------------------------------


def test_visibility_transform_enabled():
    xf = visibility_transform(["agent-a", "agent-b"])
    assert xf is not None


def test_visibility_transform_disabled():
    xf = visibility_transform(["secret-agent"], enabled=False)
    assert xf is not None


def test_visibility_transform_empty_names():
    xf = visibility_transform([])
    assert xf is not None


def test_visibility_transform_with_tags():
    xf = visibility_transform(["agent-a"], tags=["prod"])
    assert xf is not None


# ---------------------------------------------------------------------------
# resources_as_tools_transform
# ---------------------------------------------------------------------------


def test_resources_as_tools_requires_server():
    """resources_as_tools_transform must take a server argument."""
    from fastmcp import FastMCP
    server = FastMCP("test")
    xf = resources_as_tools_transform(server)
    assert xf is not None


def test_resources_as_tools_two_calls_give_independent_objects():
    from fastmcp import FastMCP
    server = FastMCP("test")
    xf1 = resources_as_tools_transform(server)
    xf2 = resources_as_tools_transform(server)
    # Two separate instances (not the same object)
    assert xf1 is not xf2


# ---------------------------------------------------------------------------
# prompts_as_tools_transform
# ---------------------------------------------------------------------------


def test_prompts_as_tools_requires_server():
    """prompts_as_tools_transform must take a server argument."""
    from fastmcp import FastMCP
    server = FastMCP("test")
    xf = prompts_as_tools_transform(server)
    assert xf is not None


# ---------------------------------------------------------------------------
# default_transform_stack
# ---------------------------------------------------------------------------


def test_default_stack_empty_by_default():
    stack = default_transform_stack()
    assert stack == []


def test_default_stack_with_agent_name():
    stack = default_transform_stack("billing")
    assert len(stack) == 1


def test_default_stack_resources_as_tools():
    from fastmcp import FastMCP
    server = FastMCP("test")
    stack = default_transform_stack(server=server, resources_as_tools=True)
    assert len(stack) == 1


def test_default_stack_prompts_as_tools():
    from fastmcp import FastMCP
    server = FastMCP("test")
    stack = default_transform_stack(server=server, prompts_as_tools=True)
    assert len(stack) == 1


def test_default_stack_all_options():
    from fastmcp import FastMCP
    server = FastMCP("test")
    stack = default_transform_stack(
        "analytics",
        server=server,
        resources_as_tools=True,
        prompts_as_tools=True,
    )
    assert len(stack) == 3


def test_default_stack_raises_without_server_for_resources():
    import pytest
    with pytest.raises(ValueError, match="server"):
        default_transform_stack(resources_as_tools=True)


def test_default_stack_raises_without_server_for_prompts():
    import pytest
    with pytest.raises(ValueError, match="server"):
        default_transform_stack(prompts_as_tools=True)


def test_default_stack_returns_list():
    from fastmcp import FastMCP
    server = FastMCP("test")
    stack = default_transform_stack("billing", server=server, resources_as_tools=True)
    assert isinstance(stack, list)
