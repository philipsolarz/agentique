"""Dev-only ``.env`` loading for the scenarios harness.

A ~10-line parser rather than a third-party dependency: the scenarios are
throwaway and the shipped package stays zero-dep. Values already present in the
real environment win over the file.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path | None = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file of ``KEY=VALUE`` lines, without
    overriding variables already set. A missing file is a no-op."""
    env_path = path if path is not None else _ENV_PATH
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(name: str) -> str:
    """Return env var ``name``, or raise with guidance if unset/empty."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in "
            "(or export the variable), then re-run."
        )
    return value
