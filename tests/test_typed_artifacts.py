"""Typed artifact payloads, the kind->model registry, and the provenance DAG.

A registered kind yields a concrete typed payload that round-trips through the
store; an unregistered kind falls back to TextPayload; derived_from edges are
recorded verbatim.
"""

from pydantic import BaseModel

from agentique import Coordinator, Store, TextPayload
from agentique.core import Agent
from agentique.testing import StubModel


class ChangePayload(BaseModel):
    summary: str
    files: list[str]


def _agent(output: str, *, output_type: type[BaseModel] | None = None) -> Agent:
    return Agent(
        name="a",
        instructions="x",
        model=StubModel([StubModel.text(output)]),
        output_type=output_type,
    )


async def test_registered_kind_produces_typed_payload_that_round_trips() -> None:
    store = Store(payload_models={"change": ChangePayload})
    coord = Coordinator(store=store)
    out = '{"summary": "did it", "files": ["a.py", "b.py"]}'
    session = await coord.dispatch(
        _agent(out, output_type=ChangePayload), "go", kind="change"
    )

    assert session.artifact is not None
    payload = session.artifact.payload
    assert isinstance(payload, ChangePayload)
    assert payload.summary == "did it"
    assert payload.files == ["a.py", "b.py"]

    # rehydrates from the store as the same concrete typed payload.
    stored = await store.get_artifact(session.artifact.id)
    assert stored is not None
    assert stored == session.artifact
    assert isinstance(stored.payload, ChangePayload)


async def test_unregistered_kind_falls_back_to_text_payload() -> None:
    coord = Coordinator()  # no registry
    session = await coord.dispatch(_agent("free-form output"), "go", kind="whatever")
    assert session.artifact is not None
    assert session.artifact.payload == TextPayload(text="free-form output")


async def test_provenance_edges_are_recorded_verbatim() -> None:
    coord = Coordinator()
    plan = await coord.dispatch(_agent("the plan"), "plan it", kind="plan")
    assert plan.artifact is not None

    change = await coord.dispatch(
        _agent("the change"),
        "build it",
        kind="change",
        derived_from=(plan.artifact.id,),
    )
    assert change.artifact is not None
    assert change.artifact.derived_from == (plan.artifact.id,)

    stored = await coord.store.get_artifact(change.artifact.id)
    assert stored is not None
    assert stored.derived_from == (plan.artifact.id,)
