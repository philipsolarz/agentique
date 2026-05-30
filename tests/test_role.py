"""Role registry + dispatch-by-name: an explicit Coordinator capability.

A registered Role is dispatched by name, producing a Session whose artifact takes
the role's kind. Exercised offline with StubModel.
"""

import pytest

from agentique import Coordinator, Role
from agentique.core import Agent
from agentique.testing import StubModel


def _role(name: str, kind: str, output: str) -> Role:
    return Role(
        name=name,
        agent=Agent(
            name=name,
            instructions="do the thing",
            model=StubModel([StubModel.text(output)]),
        ),
        kind=kind,
    )


def test_role_defaults_kind_to_result() -> None:
    agent = Agent(name="x", instructions="", model=StubModel([]))
    assert Role(name="x", agent=agent).kind == "result"


async def test_register_and_dispatch_role_by_name() -> None:
    coord = Coordinator()
    coord.register_role(_role("planner", "plan", "the plan"))

    session = await coord.dispatch_role("planner", "make a plan")

    assert session.state == "done"
    assert session.artifact is not None
    assert session.artifact.kind == "plan"
    assert session.artifact.payload == "the plan"


def test_registry_accessors() -> None:
    coord = Coordinator()
    assert coord.role("planner") is None
    planner = _role("planner", "plan", "p")
    builder = _role("builder", "change", "b")
    coord.register_role(planner)
    coord.register_role(builder)
    assert coord.role("planner") is planner
    assert {r.name for r in coord.roles()} == {"planner", "builder"}


async def test_dispatch_unknown_role_raises() -> None:
    with pytest.raises(KeyError, match="no role"):
        await Coordinator().dispatch_role("ghost", "go")
