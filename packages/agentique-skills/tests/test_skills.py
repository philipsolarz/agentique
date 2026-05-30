"""The ExtractText skill, tested in complete isolation (no model, no runtime)."""

from agentique.core import Message, TextBlock, ToolUseBlock
from agentique.skills import ExtractText


def test_joins_text_blocks_with_default_separator() -> None:
    msg = Message(
        role="assistant",
        content=(TextBlock("hello"), TextBlock("world")),
    )
    assert ExtractText()(msg) == "helloworld"


def test_honours_custom_separator() -> None:
    msg = Message(role="assistant", content=(TextBlock("a"), TextBlock("b")))
    assert ExtractText(separator="\n")(msg) == "a\nb"


def test_ignores_non_text_blocks() -> None:
    msg = Message(
        role="assistant",
        content=(
            TextBlock("keep"),
            ToolUseBlock(id="t1", name="x", input={}),
        ),
    )
    assert ExtractText()(msg) == "keep"


def test_empty_message_yields_empty_string() -> None:
    assert ExtractText()(Message(role="assistant", content=())) == ""
