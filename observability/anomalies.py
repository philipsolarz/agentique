"""Anomaly detection: the gap between what the contracts assume and what a real
run did.

This is the payload of the whole harness. Each detector is a small, explainable
rule over the recorded events plus the terminal Result — never a guess. Where a
finding points at a known spot in the frozen core, the message names it, so the
digest reads as a to-do for the contract-change session that follows. Findings are
located by turn number (``turn 2``), qualified to a specific call (``turn 2 call 1``)
only when a turn made more than one tool call.

Two tiers:

* :func:`detect_anomalies` returns the **run-specific** findings for one digest —
  what *this* run did.
* :func:`standing_notes` returns the **framework-wide** limitations that hold for
  every run identically (e.g. usage is dropped). These belong in the cross-scenario
  summary, not repeated in every digest.

Nothing here changes core; it only observes.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentique.core.result import Blocked, Result
from observability.events import (
    CallEvent,
    ModelCallEvent,
    ToolCallEvent,
    group_into_turns,
)

# stop_reasons the Runtime loop acts on explicitly (runtime.py:97). Every other
# value is silently folded into a terminal ``Completed``.
_HANDLED_STOP_REASONS = frozenset({"tool_use", "end_turn"})


def detect_anomalies(
    events: Sequence[CallEvent],
    result: Result,
) -> tuple[str, ...]:
    """Return the run-specific contract deltas a run surfaced, turn-ordered."""
    anomalies: list[str] = []

    for number, turn in enumerate(group_into_turns(events), start=1):
        anomalies.extend(_model_anomalies(number, turn.model))
        multi = len(turn.tools) > 1
        for call_index, tool in enumerate(turn.tools, start=1):
            locator = f"turn {number} call {call_index}" if multi else f"turn {number}"
            anomalies.extend(_tool_anomalies(locator, tool))

    if isinstance(result, Blocked) and "max_turns" in result.reason:
        anomalies.append(
            f"run hit the turn limit and terminated as Blocked ({result.reason})"
        )

    return tuple(anomalies)


def standing_notes() -> tuple[str, ...]:
    """Framework-wide limitations that hold identically for every run. Emitted once
    in the cross-scenario summary, not repeated per digest."""
    return (
        "usage is not capturable without a contract change: ModelResponse carries "
        "no usage and the Anthropic converter drops it (anthropic/model.py:107)",
    )


def _model_anomalies(number: int, event: ModelCallEvent) -> list[str]:
    found: list[str] = []

    if event.raised is not None:
        found.append(f"turn {number}: model call raised {event.raised}")
        return found

    if event.stop_reason not in _HANDLED_STOP_REASONS:
        found.append(
            f"turn {number}: stop_reason '{event.stop_reason}' is folded into a "
            "terminal Completed; the Runtime has no explicit handling (runtime.py:97)"
        )
    if not event.blocks:
        found.append(f"turn {number}: model returned no content blocks")
    if event.stop_reason == "tool_use" and not any(
        b.kind == "tool_use" for b in event.blocks
    ):
        found.append(
            f"turn {number}: stop_reason 'tool_use' but no tool_use block present "
            "(converter may have dropped a block kind)"
        )
    return found


def _tool_anomalies(locator: str, event: ToolCallEvent) -> list[str]:
    found: list[str] = []
    if event.raised == "PauseRequested":
        # ask_human pauses by raising PauseRequested, a control signal the Runtime
        # turns into NeedsHuman — expected, not a fault.
        return found
    if event.raised is not None:
        found.append(
            f"{locator}: tool '{event.tool_name}' raised {event.raised}; the Runtime "
            "folds a raised tool into an error result the model can recover from"
        )
    elif event.is_error:
        found.append(
            f"{locator}: tool '{event.tool_name}' returned an error result (is_error)"
        )
    return found
