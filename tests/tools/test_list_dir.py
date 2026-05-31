"""ListDir tool: confined directory listing against a real temp workspace."""

from pathlib import Path

from agentique.core.run_context import RunContext
from agentique.tools.list_dir import ListDir
from agentique.tools.workspace import Workspace


def _tool(root: Path) -> ListDir:
    return ListDir(Workspace(root))


def test_spec_declares_optional_path() -> None:
    spec = _tool(Path(".")).spec
    assert spec.name == "list_dir"
    assert "required" not in spec.input_schema


async def test_lists_entries_with_dir_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    result = await _tool(tmp_path)(RunContext.for_test(), {"path": "."})
    assert result.is_error is False
    assert result.content.splitlines() == ["a.txt", "sub/"]


async def test_empty_directory(tmp_path: Path) -> None:
    result = await _tool(tmp_path)(RunContext.for_test(), {})
    assert result.is_error is False
    assert result.content == "(empty)"


async def test_escape_is_error(tmp_path: Path) -> None:
    result = await _tool(tmp_path / "inner")(RunContext.for_test(), {"path": "../.."})
    assert result.is_error is True
    assert "escapes" in result.content


async def test_not_a_directory_is_error(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    result = await _tool(tmp_path)(RunContext.for_test(), {"path": "file.txt"})
    assert result.is_error is True
    assert "not a directory" in result.content


async def test_non_string_path_is_error(tmp_path: Path) -> None:
    result = await _tool(tmp_path)(RunContext.for_test(), {"path": 123})
    assert result.is_error is True
    assert "must be a string" in result.content
