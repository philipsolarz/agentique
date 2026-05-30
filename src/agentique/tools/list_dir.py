"""A confined tool: list the entries of a directory inside the workspace.

Read-only and non-destructive, like :class:`~agentique.tools.read_file.ReadFile`,
but jailed to a :class:`~agentique.tools.workspace.Workspace` so an agent can only
see within its authorized scope. A path that escapes the root, or that is not a
directory, is returned as an error result for the model to recover from.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.tool import ToolResult, ToolSpec
from agentique.tools.workspace import Workspace, WorkspaceError


class ListDir:
    """List the names of entries in a directory, relative to the workspace root."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_dir",
            description=(
                "List the entries of a directory within the workspace. The path is "
                "relative to the workspace root; '.' lists the root itself. "
                "Directories are suffixed with '/'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the workspace "
                        "root (default '.').",
                    }
                },
            },
        )

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        path = arguments.get("path", ".")
        if not isinstance(path, str):
            return ToolResult(content="argument 'path' must be a string", is_error=True)
        try:
            target = self._workspace.resolve(path)
        except WorkspaceError as exc:
            return ToolResult(content=str(exc), is_error=True)
        if not target.is_dir():
            return ToolResult(content=f"{path!r} is not a directory", is_error=True)
        entries = sorted(
            f"{child.name}/" if child.is_dir() else child.name
            for child in target.iterdir()
        )
        listing = "\n".join(entries)
        return ToolResult(content=listing if listing else "(empty)")
