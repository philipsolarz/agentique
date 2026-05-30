"""Scenario driver: a wrapped run produces the three artifacts in a deterministic
directory, leaves the original agent untouched, and rolls up correct counts —
exercised offline through StubModel/EchoTool.
"""

from agentique.core import Agent
from agentique.core.result import Completed
from agentique.testing import EchoTool, StubModel
from scenarios.driver import run_scenario


def _script() -> list:
    return [StubModel.tool_call("c1", "echo", {"value": "hi"}), StubModel.text("done")]


async def test_run_scenario_writes_run_dir(tmp_path) -> None:
    model = StubModel(_script())
    agent = Agent(name="a", instructions="sys", model=model, tools=(EchoTool(),))

    run = await run_scenario(
        "demo", agent, "go", runs_root=tmp_path, timestamp="20260530-000000"
    )

    assert isinstance(run.result, Completed)
    assert run.out_dir == tmp_path / "20260530-000000-demo"
    assert (run.out_dir / "events.jsonl").exists()
    assert (run.out_dir / "manifest.json").exists()
    assert (run.out_dir / "digest.md").exists()
    assert run.record.turns == 2
    assert run.record.tool_calls == 1
    # The driver instrumented a copy; the original agent is untouched.
    assert agent.model is model
    assert isinstance(agent.tools[0], EchoTool)
