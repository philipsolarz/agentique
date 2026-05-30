"""Control signals: in-band exceptions the Runtime interprets, not errors.

A tool ordinarily reports failure by returning ``ToolResult(is_error=True)``, which
the Runtime folds back so the model can recover. A *control signal* is different: it
is a cooperating tool telling the Runtime to change the run's flow. The Runtime
catches it explicitly — ahead of the generic raised-error handler — and acts on it,
so it never reaches the model as an error.
"""

from __future__ import annotations


class PauseRequested(Exception):
    """Raised by the ``ask_human`` tool to pause the run for human input.

    Not an error: the Runtime catches it and returns
    :class:`~agentique.core.result.NeedsHuman` carrying a resumable
    :class:`~agentique.core.result.Paused` snapshot. ``question`` is what to ask.
    """

    def __init__(self, question: str) -> None:
        super().__init__(question)
        self.question = question
