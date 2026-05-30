"""ReadFile tool, tested in isolation against a real temp filesystem."""

from pathlib import Path

from agentique.tools import ReadFile


def test_spec_declares_required_path() -> None:
    spec = ReadFile().spec
    assert spec.name == "read_file"
    assert spec.input_schema["required"] == ["path"]


async def test_reads_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("contents here", encoding="utf-8")
    result = await ReadFile()({"path": str(target)})
    assert result.is_error is False
    assert result.content == "contents here"


async def test_missing_file_returns_error_result(tmp_path: Path) -> None:
    result = await ReadFile()({"path": str(tmp_path / "nope.txt")})
    assert result.is_error is True
    assert "could not read" in result.content


async def test_non_string_path_returns_error_result() -> None:
    result = await ReadFile()({"path": 123})
    assert result.is_error is True
    assert "must be a string" in result.content
