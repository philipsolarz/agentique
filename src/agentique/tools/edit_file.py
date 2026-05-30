"""A confined, world-changing tool: a single string-replace edit of a file.

Like :class:`~agentique.tools.write_file.WriteFile`, it is jailed to a
:class:`~agentique.tools.workspace.Workspace`. The edit must be unambiguous: the
``old`` string has to occur *exactly once* in the file. Zero matches (nothing to
replace) or multiple matches (ambiguous) are returned as error results, so the
model can widen its context and retry rather than silently corrupting the file.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.tool import ToolResult, ToolSpec
from agentique.tools.workspace import Workspace, WorkspaceError


class EditFile:
    """Replace the sole occurrence of ``old`` with ``new`` in a workspace file."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="edit_file",
            description=(
                "Replace an exact substring in a workspace file. 'old' must match "
                "exactly once in the file; include enough surrounding text to make "
                "it unique. The path is relative to the workspace root."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "old": {
                        "type": "string",
                        "description": "Exact text to find (must occur exactly once).",
                    },
                    "new": {
                        "type": "string",
                        "description": "Text to replace the match with.",
                    },
                },
                "required": ["path", "old", "new"],
            },
        )

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        path = arguments.get("path")
        old = arguments.get("old")
        new = arguments.get("new")
        if not isinstance(path, str):
            return ToolResult(content="argument 'path' must be a string", is_error=True)
        if not isinstance(old, str):
            return ToolResult(content="argument 'old' must be a string", is_error=True)
        if not isinstance(new, str):
            return ToolResult(content="argument 'new' must be a string", is_error=True)
        try:
            text = self._workspace.read_text(path)
        except (WorkspaceError, OSError) as exc:
            return ToolResult(content=f"could not read {path!r}: {exc}", is_error=True)
        occurrences = text.count(old)
        if occurrences == 0:
            return ToolResult(
                content=f"no match for the given text in {path}", is_error=True
            )
        if occurrences > 1:
            return ToolResult(
                content=(
                    f"ambiguous edit: {occurrences} matches in {path}; include more "
                    "surrounding text to make 'old' unique"
                ),
                is_error=True,
            )
        try:
            self._workspace.write_text(path, text.replace(old, new, 1))
        except (WorkspaceError, OSError) as exc:
            return ToolResult(content=f"could not write {path!r}: {exc}", is_error=True)
        return ToolResult(content=f"edited {path} (1 replacement)")
