"""RunCommand: allowlist, shell-free argv, confinement, timeout, output bounds.

Uses standard POSIX utilities (``ls``, ``cat``, ``sleep``) that are reliably on
PATH, exercising the policy without depending on a specific interpreter path.
"""

from pathlib import Path

from agentique.core.run_context import RunContext
from agentique.tools.run_command import DEFAULT_ALLOWLIST, RunCommand
from agentique.tools.workspace import Workspace


def test_spec_requires_command_list() -> None:
    spec = RunCommand(Workspace(Path("."))).spec
    assert spec.name == "run_command"
    assert spec.input_schema["required"] == ["command"]


async def test_runs_allowlisted_command_cwd_pinned(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    tool = RunCommand(Workspace(tmp_path), allowlist=frozenset({"ls"}))
    result = await tool(RunContext.for_test(), {"command": ["ls"]})
    assert result.is_error is False
    assert "exit code 0" in result.content
    assert "marker.txt" in result.content


async def test_rejects_non_allowlisted_executable(tmp_path: Path) -> None:
    tool = RunCommand(Workspace(tmp_path), allowlist=frozenset({"ls"}))
    result = await tool(RunContext.for_test(), {"command": ["rm", "-rf", "."]})
    assert result.is_error is True
    assert "not an allowed command" in result.content


async def test_rejects_path_separator_executable(tmp_path: Path) -> None:
    tool = RunCommand(Workspace(tmp_path), allowlist=frozenset({"ls"}))
    result = await tool(RunContext.for_test(), {"command": ["/bin/ls"]})
    assert result.is_error is True
    assert "bare executable name" in result.content


async def test_nonzero_exit_is_error_result(tmp_path: Path) -> None:
    tool = RunCommand(Workspace(tmp_path), allowlist=frozenset({"cat"}))
    result = await tool(
        RunContext.for_test(), {"command": ["cat", "does_not_exist.txt"]}
    )
    assert result.is_error is True
    assert "exit code" in result.content
    assert "exit code 0" not in result.content


async def test_timeout_is_error(tmp_path: Path) -> None:
    tool = RunCommand(Workspace(tmp_path), allowlist=frozenset({"sleep"}), timeout=0.2)
    result = await tool(RunContext.for_test(), {"command": ["sleep", "5"]})
    assert result.is_error is True
    assert "timed out" in result.content


async def test_output_truncation(tmp_path: Path) -> None:
    # `cat` a large file back, capped to a tiny output limit.
    (tmp_path / "big.txt").write_text("y" * 1000, encoding="utf-8")
    tool = RunCommand(
        Workspace(tmp_path), allowlist=frozenset({"cat"}), output_limit=40
    )
    result = await tool(RunContext.for_test(), {"command": ["cat", "big.txt"]})
    assert "[output truncated]" in result.content


async def test_non_list_command_is_error(tmp_path: Path) -> None:
    tool = RunCommand(Workspace(tmp_path))
    result = await tool(RunContext.for_test(), {"command": "ls -la"})
    assert result.is_error is True
    assert "must be a list of strings" in result.content


async def test_empty_command_is_error(tmp_path: Path) -> None:
    tool = RunCommand(Workspace(tmp_path))
    result = await tool(RunContext.for_test(), {"command": []})
    assert result.is_error is True
    assert "must not be empty" in result.content


def test_default_allowlist_covers_builder_needs() -> None:
    assert {"python", "node", "pytest", "ls", "cat"} <= DEFAULT_ALLOWLIST
