"""Import-discipline guard: only ``agentique.anthropic`` may import third-party code.

Collapsing the six distributions into one removed the dependency isolation that a
separate distribution gave the Anthropic provider. This test preserves that
boundary *structurally* instead: every module under ``src/agentique`` EXCEPT
``agentique.anthropic`` must import only the standard library and ``agentique``
itself. That is what lets ``pip install agentique`` (no extras) import any of those
modules with no third-party package present — the zero-third-party guarantee.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_SRC = (Path(__file__).resolve().parent.parent / "src" / "agentique").resolve()
_EXEMPT_DIR = _SRC / "anthropic"
# The canonical stdlib top-level names for this interpreter, plus our own package.
_ALLOWED_ROOTS = frozenset(sys.stdlib_module_names) | {"agentique"}


def _imported_roots(tree: ast.Module) -> set[str]:
    """Top-level package names imported by absolute imports in ``tree``.

    Relative imports (``from . import x``, ``level > 0``) are intra-package and
    never reach a third-party name, so they are ignored.
    """
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_only_anthropic_imports_third_party() -> None:
    violations: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        if _EXEMPT_DIR in path.parents:
            continue
        roots = _imported_roots(ast.parse(path.read_text()))
        third_party = {root for root in roots if root not in _ALLOWED_ROOTS}
        if third_party:
            violations[str(path.relative_to(_SRC.parent.parent))] = third_party
    assert not violations, (
        "third-party imports found outside agentique.anthropic: "
        f"{ {k: sorted(v) for k, v in violations.items()} }"
    )
