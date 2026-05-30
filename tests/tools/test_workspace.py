"""Workspace jail: path-escape rejection (the security spine) and the runner."""

import sys
from pathlib import Path

import pytest

from agentique.tools.workspace import Workspace, WorkspaceError


def test_resolve_keeps_paths_inside_root(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    resolved = ws.resolve("sub/file.txt")
    assert resolved == (tmp_path / "sub" / "file.txt").resolve()
    assert resolved.is_relative_to(tmp_path.resolve())


def test_resolve_allows_interior_dotdot(tmp_path: Path) -> None:
    # A `..` that stays within the root is fine — only escapes are rejected.
    ws = Workspace(tmp_path)
    assert ws.resolve("sub/../file.txt") == (tmp_path / "file.txt").resolve()


def test_resolve_rejects_absolute_path(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="must be relative"):
        ws.resolve("/etc/passwd")


def test_resolve_rejects_dotdot_escape(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "inner")
    with pytest.raises(WorkspaceError, match="escapes the workspace"):
        ws.resolve("../outside.txt")


def test_read_and_write_text_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    target = ws.write_text("nested/dir/note.txt", "hello jail")
    assert target.read_text(encoding="utf-8") == "hello jail"
    assert ws.read_text("nested/dir/note.txt") == "hello jail"


async def test_run_captures_output_with_cwd_pinned(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    result = await ws.run(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        timeout=10,
        output_limit=4096,
    )
    assert result.returncode == 0
    assert result.output.strip() == str(tmp_path.resolve())
    assert result.truncated is False


async def test_run_truncates_oversized_output(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    result = await ws.run(
        [sys.executable, "-c", "print('x' * 1000)"],
        timeout=10,
        output_limit=50,
    )
    assert len(result.output) == 50
    assert result.truncated is True


async def test_run_times_out(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="timed out"):
        await ws.run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.2,
            output_limit=4096,
        )


async def test_run_minimal_env_does_not_leak_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    ws = Workspace(tmp_path)
    result = await ws.run(
        [sys.executable, "-c", "import os; print(os.environ.get('ANTHROPIC_API_KEY'))"],
        timeout=10,
        output_limit=4096,
    )
    assert result.output.strip() == "None"
