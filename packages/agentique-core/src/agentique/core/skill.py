"""The Skill seam: a pure, deterministic routine.

A Skill touches neither the world nor the model — it parses, validates,
transforms, or composes. "Determinism inside, discretion outside": the model
decides *when* and *what*; the skill owns *how*. Because a skill is pure and
typed in its input and output, it is unit-testable in complete isolation.

Skills are synchronous: a pure transform has no reason to ``await``. (Tools,
which may perform I/O, are the async seam.)

How a skill is surfaced to the model for invocation is deliberately left open
here — that mechanism is established in A2, when the first real skill is wired.
A1 commits only to the essential contract: a typed, pure callable.
"""

from __future__ import annotations

from typing import Protocol


class Skill[In, Out](Protocol):
    """A pure function from a typed input to a typed output."""

    def __call__(self, input: In) -> Out:
        """Transform ``input`` into an ``Out`` deterministically, with no side
        effects and no dependence on external state."""
        ...
