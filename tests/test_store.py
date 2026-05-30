"""Shared store: artifacts and session records round-trip through the Memory seam,
listing works, and re-putting the same id overwrites without duplicating the index.
"""

from agentique import Artifact, Store
from agentique.session import SessionRecord


async def test_artifact_round_trips_and_lists() -> None:
    store = Store()
    artifact = Artifact(id="a1", kind="plan", payload="hello", status="proposed")
    await store.put_artifact(artifact)

    assert await store.get_artifact("a1") == artifact
    assert await store.artifacts() == (artifact,)


async def test_missing_artifact_is_none() -> None:
    assert await Store().get_artifact("nope") is None


async def test_session_record_round_trips_and_lists() -> None:
    store = Store()
    record = SessionRecord(
        id="s1", agent_name="planner", state="paused", question="ok?"
    )
    await store.put_session(record)

    assert await store.get_session("s1") == record
    assert await store.sessions() == (record,)


async def test_put_artifact_overwrites_without_duplicating_index() -> None:
    store = Store()
    artifact = Artifact(id="a1", kind="plan", payload="v1")
    await store.put_artifact(artifact)
    await store.put_artifact(artifact.approved())

    artifacts = await store.artifacts()
    assert len(artifacts) == 1
    assert artifacts[0].status == "approved"
