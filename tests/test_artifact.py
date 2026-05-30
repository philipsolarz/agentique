"""Artifact: defaults to ``proposed`` and transitions are pure (return copies)."""

from agentique.code import Artifact


def test_artifact_defaults_to_proposed() -> None:
    artifact = Artifact(id="a1", kind="plan", payload="do x")
    assert artifact.status == "proposed"


def test_approve_and_reject_return_new_values() -> None:
    artifact = Artifact(id="a1", kind="plan", payload="do x")
    assert artifact.approved().status == "approved"
    assert artifact.rejected().status == "rejected"
    # the original is untouched — Artifact is a frozen value.
    assert artifact.status == "proposed"
