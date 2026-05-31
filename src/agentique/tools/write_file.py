"""A confined, world-changing tool: write (or overwrite) a file.

This crosses from reading the world to changing it, so it is gated by a
:class:`~agentique.tools.workspace.Workspace`: the write lands immediately
(do-first) but only ever inside the authorized root, with parent directories
created as needed. The operator's consequential gate is approving the resulting
artifact, not this call (see the build plan's gating design). A path that escapes
the root is returned as an error result, not raised.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.run_context import RunContext
from agentique.core.tool import ToolResult, ToolSpec
from agentique.tools.workspace import Workspace, WorkspaceError


class WriteFile:
    """Write ``content`` to a file at ``path``, relative to the workspace root."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write_file",
            description=(
                "Write text to a file within the workspace, creating or overwriting "
                "it (and any parent directories). The path is relative to the "
                "workspace root."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        )

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str):
            return ToolResult(content="argument 'path' must be a string", is_error=True)
        if not isinstance(content, str):
            return ToolResult(
                content="argument 'content' must be a string", is_error=True
            )
        try:
            self._workspace.write_text(path, content)
        except (WorkspaceError, OSError) as exc:
            return ToolResult(content=f"could not write {path!r}: {exc}", is_error=True)
        return ToolResult(content=f"wrote {len(content)} characters to {path}")
