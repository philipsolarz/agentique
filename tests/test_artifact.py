"""Artifact: defaults to ``proposed`` and transitions are pure (return copies)."""

from agentique import Artifact, TextPayload


def test_artifact_defaults_to_proposed() -> None:
    artifact = Artifact(id="a1", kind="plan", payload=TextPayload(text="do x"))
    assert artifact.status == "proposed"


def test_approve_and_reject_return_new_values() -> None:
    artifact = Artifact(id="a1", kind="plan", payload=TextPayload(text="do x"))
    assert artifact.approved().status == "approved"
    assert artifact.rejected().status == "rejected"
    # the original is untouched — Artifact is a frozen value.
    assert artifact.status == "proposed"


def test_with_status_accepts_application_defined_states() -> None:
    artifact = Artifact(id="a1", kind="change", payload=TextPayload(text="diff"))
    # the lifecycle is application-defined, not a fixed set.
    advanced = artifact.with_status("executing").with_status("done")
    assert advanced.status == "done"
    assert artifact.status == "proposed"
