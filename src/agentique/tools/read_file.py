"""A real tool: read a UTF-8 text file from the filesystem.

Crosses the filesystem boundary, so it is a Tool and is subject to
the Runtime's permission check. It is read-only and non-destructive. Failures
(missing file, decode error, a path that is a directory) are returned as error
results rather than raised, so the Runtime can feed them back to the model for
recovery instead of aborting the run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from agentique.core.tool import ToolResult, ToolSpec


class ReadFile:
    """Read the contents of a text file given a ``path`` argument."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_file",
            description="Read and return the UTF-8 text contents of a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filesystem path of the file to read.",
                    }
                },
                "required": ["path"],
            },
        )

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(content="argument 'path' must be a string", is_error=True)
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult(content=f"could not read {path!r}: {exc}", is_error=True)
        return ToolResult(content=text)
