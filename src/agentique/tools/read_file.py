"""A real tool: read a UTF-8 text file from the filesystem.

Crosses the filesystem boundary, so it is a Tool and is subject to
the Engine's permission check. It is read-only and non-destructive. Failures
(missing file, decode error, a path that is a directory) are returned as error
results rather than raised, so the Engine can feed them back to the model for
recovery instead of aborting the run.

Given a :class:`~agentique.tools.workspace.Workspace`, reads are confined to it and
paths are workspace-relative — so a fleet of agents sharing a workspace reads and
writes against the same root. Without one (the default), it reads any path the
process can, as before.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from agentique.core.run_context import RunContext
from agentique.core.tool import ToolResult, ToolSpec
from agentique.tools.workspace import Workspace, WorkspaceError


class ReadFile:
    """Read the contents of a text file given a ``path`` argument."""

    def __init__(self, workspace: Workspace | None = None) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        scope = (
            " The path is relative to the workspace root."
            if self._workspace is not None
            else ""
        )
        return ToolSpec(
            name="read_file",
            description=f"Read and return the UTF-8 text contents of a file.{scope}",
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

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(content="argument 'path' must be a string", is_error=True)
        try:
            if self._workspace is not None:
                text = self._workspace.read_text(path)
            else:
                text = Path(path).read_text(encoding="utf-8")
        except (WorkspaceError, OSError, UnicodeDecodeError) as exc:
            return ToolResult(content=f"could not read {path!r}: {exc}", is_error=True)
        return ToolResult(content=text)
