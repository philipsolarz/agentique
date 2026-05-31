"""A confined, world-changing tool: run a bounded command and capture its output.

The most dangerous tool, so it is bounded on every axis:

* **Allowlist** — only executables named in the allowlist may run; an unknown
  program is refused before anything executes.
* **No shell** — the command is an ``argv`` list run via ``exec`` (no shell), so
  there is no metacharacter injection, piping, or redirection.
* **Bare names only** — the executable must be a bare name resolved via ``PATH``
  (no ``/`` or ``\\``), so ``./evil`` or ``/abs/path`` cannot impersonate an
  allowlisted name.
* **Confinement** — the :class:`~agentique.tools.workspace.Workspace` pins the
  working directory to the root and runs under a minimal environment.
* **Time and output bounds** — a wall-clock timeout kills runaways; captured
  output is truncated to a cap.

A non-zero exit status is reported as an error result (so the model sees the
failure and can recover), but the command *did* run — that is a result, not a
broken tool.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.run_context import RunContext
from agentique.core.tool import ToolResult, ToolSpec
from agentique.tools.workspace import Workspace, WorkspaceError

# The default set of executables the Builder needs to write and self-verify code.
# Kept deliberately small; widen it explicitly when a real need appears.
DEFAULT_ALLOWLIST = frozenset({"python", "python3", "node", "pytest", "ls", "cat"})


class RunCommand:
    """Run an allowlisted, shell-free command confined to the workspace root."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        allowlist: frozenset[str] = DEFAULT_ALLOWLIST,
        timeout: float = 30.0,
        output_limit: int = 16_384,
    ) -> None:
        self._workspace = workspace
        self._allowlist = allowlist
        self._timeout = timeout
        self._output_limit = output_limit

    @property
    def spec(self) -> ToolSpec:
        allowed = ", ".join(sorted(self._allowlist))
        return ToolSpec(
            name="run_command",
            description=(
                "Run a command in the workspace and return its exit status and "
                "output. The command is a list of argv strings (no shell, so no "
                "pipes/redirection); the first element must be one of the allowed "
                f"executables: {allowed}. Runs with the workspace root as the "
                "working directory."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Argv list, e.g. "
                        '["python", "-m", "py_compile", "game.py"].',
                    }
                },
                "required": ["command"],
            },
        )

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, list):
            return ToolResult(
                content="argument 'command' must be a list of strings", is_error=True
            )
        argv = [part for part in command if isinstance(part, str)]
        if len(argv) != len(command):
            return ToolResult(
                content="argument 'command' must be a list of strings", is_error=True
            )
        if not argv:
            return ToolResult(content="'command' must not be empty", is_error=True)
        executable = argv[0]
        if "/" in executable or "\\" in executable:
            return ToolResult(
                content=(
                    f"{executable!r} must be a bare executable name "
                    "(no path separators)"
                ),
                is_error=True,
            )
        if executable not in self._allowlist:
            allowed = ", ".join(sorted(self._allowlist))
            return ToolResult(
                content=(
                    f"{executable!r} is not an allowed command (allowed: {allowed})"
                ),
                is_error=True,
            )
        try:
            result = await self._workspace.run(
                argv, timeout=self._timeout, output_limit=self._output_limit
            )
        except WorkspaceError as exc:
            return ToolResult(content=str(exc), is_error=True)
        suffix = "\n[output truncated]" if result.truncated else ""
        body = f"exit code {result.returncode}\n{result.output}{suffix}"
        return ToolResult(content=body, is_error=result.returncode != 0)
