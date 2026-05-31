"""A confined filesystem workspace: the jail every world-changing tool shares.

The write/edit/list/run tools cross from *reading* the world to *changing* it. A
``Workspace`` bounds that blast radius to a single root directory: every path is
resolved against the root and rejected if it escapes (an absolute path, or a
``..`` traversal that climbs out), and commands run with the root as their working
directory under a minimal environment. Agents act freely *inside* this authorized,
reversible scope; the operator's consequential gate is approving the *result*, not
each call (see the gating design in the build plan).

This is a plain helper, not a :class:`~agentique.core.tool.Tool` — the tools hold a
``Workspace`` and delegate their path resolution and subprocess execution to it, so
the jail logic lives in exactly one place. A jail violation raises
:class:`WorkspaceError`; the tools catch it and fold it into an error result,
keeping ``PauseRequested`` the sole control signal the Engine sees.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(Exception):
    """A path escaped the workspace root, or a confined command could not run."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The outcome of a confined command: its exit code and captured output.

    ``output`` is the merged stdout+stderr, decoded and truncated to the caller's
    limit; ``truncated`` flags that the real output was longer.
    """

    returncode: int
    output: str
    truncated: bool


class Workspace:
    """A root directory all confined tools resolve paths against and run within."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        # Resolve once so jail checks compare against a stable absolute root,
        # independent of the process's later working directory.
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, relpath: str) -> Path:
        """Resolve ``relpath`` under the root, rejecting any escape.

        Absolute inputs are refused outright; otherwise the joined path is fully
        resolved (collapsing ``..`` and symlinks) and must still lie within the
        root. Raises :class:`WorkspaceError` on any escape.
        """
        if Path(relpath).is_absolute():
            raise WorkspaceError(
                f"path {relpath!r} must be relative to the workspace root"
            )
        target = (self._root / relpath).resolve()
        if not target.is_relative_to(self._root):
            raise WorkspaceError(f"path {relpath!r} escapes the workspace root")
        return target

    def read_text(self, relpath: str) -> str:
        """Read a UTF-8 text file confined to the workspace."""
        return self.resolve(relpath).read_text(encoding="utf-8")

    def write_text(self, relpath: str, content: str) -> Path:
        """Write ``content`` to a confined path, creating parent dirs as needed."""
        target = self.resolve(relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    async def run(
        self, argv: Sequence[str], *, timeout: float, output_limit: int
    ) -> CommandResult:
        """Run ``argv`` with the root as cwd, capturing merged, capped output.

        The command runs via ``exec`` (no shell, so no metacharacter injection),
        with its working directory pinned to the root and a minimal environment
        (only ``PATH``, so the executable resolves but process secrets do not
        leak). It is killed if it exceeds ``timeout`` seconds. The allowlisting of
        *which* executables may run is the caller's policy, not the jail's.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except OSError as exc:
            raise WorkspaceError(f"could not run {argv!r}: {exc}") from exc
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise WorkspaceError(f"command timed out after {timeout:g}s") from exc
        text = stdout.decode("utf-8", errors="replace")
        truncated = len(text) > output_limit
        return CommandResult(
            returncode=proc.returncode if proc.returncode is not None else -1,
            output=text[:output_limit],
            truncated=truncated,
        )
