"""Typed I/O validation: the one Pydantic rung behind a thin validator seam.

The engine validates a tool's arguments before it runs and an agent's final output
before it completes. When a declared type is present and the data does not match,
the mismatch is turned into feedback the *model* can self-correct from — the same
teach-the-model-to-retry pattern a tool's own argument check uses — rather than
crashing the run. When no type is declared the engine skips validation entirely,
so the hot path pays nothing.

:class:`Validator` is the seam (headroom for an alternate validator); the single
shipped rung is :class:`PydanticValidator`, which wraps a Pydantic model class.
Per the realignment, there is deliberately no second validator implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ValidationError


class Validator(Protocol):
    """Validates raw model-produced data into a typed value, raising on mismatch.

    The extension seam: a non-Pydantic validator could implement this. We ship one
    rung (:class:`PydanticValidator`) and build no second implementation.
    """

    def validate(self, raw: object) -> object: ...

    def validate_json(self, raw: str) -> object: ...


@dataclass(frozen=True, slots=True)
class PydanticValidator:
    """The one validator rung: a Pydantic model class used to validate/coerce."""

    model: type[BaseModel]

    def validate(self, raw: object) -> BaseModel:
        return self.model.model_validate(raw)

    def validate_json(self, raw: str) -> BaseModel:
        return self.model.model_validate_json(raw)


def validate_tool_args(
    args_model: type[BaseModel], raw: Mapping[str, object]
) -> str | None:
    """Validate a tool's arguments. Returns ``None`` when valid, or a self-correction
    message (the validation errors) the model can act on when not."""
    try:
        PydanticValidator(args_model).validate(dict(raw))
        return None
    except ValidationError as exc:
        return (
            "arguments did not match this tool's schema; fix them and call "
            f"again:\n{exc}"
        )


def validate_output(output_type: type[BaseModel], text: str) -> str | None:
    """Validate an agent's final output text against its declared output type.
    Returns ``None`` when valid, or a self-correction message when not."""
    try:
        PydanticValidator(output_type).validate_json(text)
        return None
    except ValidationError as exc:
        return (
            "your final answer must be JSON matching the required schema; fix it "
            f"and reply again:\n{exc}"
        )
