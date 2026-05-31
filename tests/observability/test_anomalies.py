"""Anomaly detector: each run-specific rule fires on the synthetic event stream it
targets, a clean run stays silent, locators carry turn numbers (qualified for
multi-call turns), and the static standing notes live apart from per-run findings.
"""

from agentique.core.context import Context
from agentique.core.result import Blocked, Completed
from observability.anomalies import detect_anomalies, standing_notes
from observability.events import (
    ContentBlockSummary,
    ModelCallEvent,
    ToolCallEvent,
)

_CTX = Context()


def _model(
    stop_reason: str | None,
    *,
    blocks: tuple[ContentBlockSummary, ...] = (ContentBlockSummary("text", 5),),
    raised: str | None = None,
) -> ModelCallEvent:
    return ModelCallEvent(
        system_len=10,
        message_count=1,
        tool_names=(),
        stop_reason=stop_reason,
        blocks=() if raised else blocks,
        latency_s=0.1,
        raised=raised,
    )


def _tool(*, is_error: bool = False, raised: str | None = None) -> ToolCallEvent:
    return ToolCallEvent(
        tool_name="read_file",
        arguments={"path": "x"},
        result_len=None if raised else 10,
        is_error=None if raised else is_error,
        latency_s=0.02,
        raised=raised,
    )


def test_clean_text_run_is_silent() -> None:
    # The static usage note is hoisted to the summary, so a clean run says nothing.
    anomalies = detect_anomalies([_model("end_turn")], Completed("hi", _CTX))
    assert anomalies == ()


def test_tool_error_result_is_flagged() -> None:
    events = [
        _model("tool_use", blocks=(ContentBlockSummary("tool_use", 8),)),
        _tool(is_error=True),
    ]
    anomalies = detect_anomalies(events, Completed("", _CTX))
    assert any(
        a.startswith("turn 1:") and "returned an error result" in a for a in anomalies
    )


def test_raised_tool_is_flagged_as_recoverable() -> None:
    events = [
        _model("tool_use", blocks=(ContentBlockSummary("tool_use", 8),)),
        _tool(raised="RuntimeError"),
    ]
    anomalies = detect_anomalies(events, Completed("", _CTX))
    assert any("can recover from" in a for a in anomalies)


def test_pause_requested_pause_is_not_flagged() -> None:
    # ask_human's PauseRequested control signal is expected, not an anomaly.
    events = [
        _model("tool_use", blocks=(ContentBlockSummary("tool_use", 8),)),
        _tool(raised="PauseRequested"),
    ]
    anomalies = detect_anomalies(events, Completed("", _CTX))
    assert not any("PauseRequested" in a for a in anomalies)
    assert not any("raised" in a for a in anomalies)


def test_multi_call_turn_qualifies_the_locator() -> None:
    # Two tool calls in one turn: the second errors -> 'turn 1 call 2'.
    events = [
        _model("tool_use", blocks=(ContentBlockSummary("tool_use", 8),)),
        _tool(),
        _tool(is_error=True),
    ]
    anomalies = detect_anomalies(events, Completed("", _CTX))
    assert any(a.startswith("turn 1 call 2:") for a in anomalies)


def test_tool_use_without_tool_block_is_flagged() -> None:
    # stop_reason says tool_use but the model returned only text — a dropped block.
    anomalies = detect_anomalies(
        [_model("tool_use", blocks=(ContentBlockSummary("text", 3),))],
        Completed("", _CTX),
    )
    assert any("no tool_use block present" in a for a in anomalies)


def test_max_turns_block_is_flagged() -> None:
    result = Blocked("exceeded max_turns (8)", _CTX)
    anomalies = detect_anomalies(
        [_model("tool_use", blocks=(ContentBlockSummary("tool_use", 8),))], result
    )
    assert any("turn limit" in a for a in anomalies)


def test_standing_notes_are_empty_after_the_gaps_closed() -> None:
    # The realignment closed both standing gaps: the Runtime now surfaces unhandled
    # stop reasons explicitly, and ModelResponse carries usage. No standing note
    # remains — in particular the old "usage is not capturable" note is gone.
    assert standing_notes() == ()
