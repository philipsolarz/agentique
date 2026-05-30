"""Smoke test proving the package imports and the test gate runs."""

from agentique.core import __version__


def test_version_is_nonempty_string() -> None:
    assert isinstance(__version__, str)
    assert __version__
