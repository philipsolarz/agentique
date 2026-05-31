"""Permissions: deny > ask > allow precedence, argument-scoped rules, and the
engine paths — a ``deny`` blocks the run, an ``ask`` joins the human-pause spine."""

from agentique.core import (
    Agent,
    Blocked,
    Completed,
    Engine,
    NeedsHuman,
    Permissions,
    Rule,
)
from agentique.testing import EchoTool, StubModel


def test_deny_beats_ask_beats_allow() -> None:
    perms = Permissions(
        rules=(
            Rule("allow"),  # wildcard allow
            Rule("ask", tool_name="x"),
            Rule("deny", tool_name="x"),
        )
    )
    assert perms.decide("x", {}) == "deny"  # deny outranks the others
    assert perms.decide("y", {}) == "allow"  # only the wildcard allow matches


def test_ask_outranks_allow() -> None:
    perms = Permissions(rules=(Rule("allow"), Rule("ask", tool_name="x")))
    assert perms.decide("x", {}) == "ask"
    assert perms.decide("y", {}) == "allow"


def test_default_permits_when_no_rule_matches() -> None:
    assert Permissions().decide("anything", {}) == "allow"


def test_allowlist_denies_unlisted() -> None:
    perms = Permissions.allowlist({"echo"})
    assert perms.decide("echo", {}) == "allow"
    assert perms.decide("other", {}) == "deny"
    assert Permissions.allowlist(()).decide("echo", {}) == "deny"


def test_argument_predicate_scopes_a_rule() -> None:
    perms = Permissions(
        rules=(
            Rule(
                "deny",
                tool_name="write",
                arg_predicate=lambda a: a.get("path") == "/etc",
            ),
        ),
        default="allow",
    )
    assert perms.decide("write", {"path": "/etc"}) == "deny"
    assert perms.decide("write", {"path": "/tmp"}) == "allow"


def _agent(model: StubModel, tool: EchoTool, perms: Permissions) -> Agent:
    return Agent(
        name="t", instructions="s", model=model, tools=(tool,), permissions=perms
    )


async def test_deny_rule_blocks_the_run_without_running_the_tool() -> None:
    tool = EchoTool()
    perms = Permissions(rules=(Rule("deny", tool_name="echo"),))
    model = StubModel([StubModel.tool_call("c1", "echo", {"value": "x"})])
    result = await Engine().run(_agent(model, tool, perms), "go")
    assert isinstance(result, Blocked)
    assert "permission denied" in result.reason
    assert tool.calls == []


async def test_ask_rule_pauses_then_resume_folds_the_answer() -> None:
    tool = EchoTool()
    perms = Permissions(rules=(Rule("ask", tool_name="echo"),))
    model = StubModel(
        [
            StubModel.tool_call("c1", "echo", {"value": "x"}),
            StubModel.text("done"),
        ]
    )
    agent = _agent(model, tool, perms)
    engine = Engine()

    paused = await engine.run(agent, "go")
    assert isinstance(paused, NeedsHuman)
    assert "echo" in paused.question
    assert paused.paused.pending_tool_use_id == "c1"

    resumed = await engine.resume(agent, paused.paused, "approved")
    assert isinstance(resumed, Completed)
    assert resumed.output == "done"
    # an ask pauses *before* the tool runs; the human's answer is folded as the
    # call's result (approve-then-run is headroom), so the tool itself never ran.
    assert tool.calls == []
