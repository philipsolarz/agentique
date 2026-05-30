"""The CLI's stdlib .env loader parses keys, strips quotes, and skips junk lines."""

from pathlib import Path

from agentique.console.cli import load_env


def test_load_env_parses_keys_and_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# a comment\nANTHROPIC_API_KEY="sk-123"\nFOO=bar\n\nNO_EQUALS_LINE\n',
        encoding="utf-8",
    )
    env = load_env(env_file)
    assert env["ANTHROPIC_API_KEY"] == "sk-123"
    assert env["FOO"] == "bar"
    assert "NO_EQUALS_LINE" not in env


def test_load_env_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_env(tmp_path / "nope.env") == {}
