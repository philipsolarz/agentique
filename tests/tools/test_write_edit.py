"""WriteFile and EditFile: confined writes and unambiguous string-replace edits."""

from pathlib import Path

from agentique.tools.edit_file import EditFile
from agentique.tools.workspace import Workspace
from agentique.tools.write_file import WriteFile


def test_write_spec_requires_path_and_content() -> None:
    spec = WriteFile(Workspace(Path("."))).spec
    assert spec.name == "write_file"
    assert spec.input_schema["required"] == ["path", "content"]


async def test_write_creates_file_and_parents(tmp_path: Path) -> None:
    tool = WriteFile(Workspace(tmp_path))
    result = await tool({"path": "game/snake.html", "content": "<html></html>"})
    assert result.is_error is False
    assert (tmp_path / "game" / "snake.html").read_text(encoding="utf-8") == (
        "<html></html>"
    )


async def test_write_overwrites(tmp_path: Path) -> None:
    tool = WriteFile(Workspace(tmp_path))
    await tool({"path": "f.txt", "content": "first"})
    await tool({"path": "f.txt", "content": "second"})
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "second"


async def test_write_escape_is_error(tmp_path: Path) -> None:
    tool = WriteFile(Workspace(tmp_path / "inner"))
    result = await tool({"path": "../escape.txt", "content": "x"})
    assert result.is_error is True
    assert "could not write" in result.content


async def test_write_non_string_content_is_error(tmp_path: Path) -> None:
    tool = WriteFile(Workspace(tmp_path))
    result = await tool({"path": "f.txt", "content": 123})
    assert result.is_error is True
    assert "'content' must be a string" in result.content


async def test_edit_replaces_unique_match(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello world", encoding="utf-8")
    tool = EditFile(Workspace(tmp_path))
    result = await tool({"path": "f.txt", "old": "world", "new": "snake"})
    assert result.is_error is False
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "hello snake"


async def test_edit_no_match_is_error(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    tool = EditFile(Workspace(tmp_path))
    result = await tool({"path": "f.txt", "old": "absent", "new": "x"})
    assert result.is_error is True
    assert "no match" in result.content


async def test_edit_ambiguous_match_is_error(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a a a", encoding="utf-8")
    tool = EditFile(Workspace(tmp_path))
    result = await tool({"path": "f.txt", "old": "a", "new": "b"})
    assert result.is_error is True
    assert "ambiguous" in result.content
    # unchanged on ambiguity
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "a a a"


async def test_edit_missing_file_is_error(tmp_path: Path) -> None:
    tool = EditFile(Workspace(tmp_path))
    result = await tool({"path": "nope.txt", "old": "a", "new": "b"})
    assert result.is_error is True
    assert "could not read" in result.content
