"""Import-discipline guards: the third-party boundary and the three-layer arrows.

Two structural rules, both enforced by AST so a violation fails the gate rather
than waiting to surprise a downstream user:

1. **Third-party boundary.** Collapsing the six distributions into one removed the
   dependency isolation that a separate distribution gave the Anthropic provider.
   Every module under ``src/agentique`` EXCEPT ``agentique.anthropic`` must import
   only the standard library and ``agentique`` itself — what lets ``pip install
   agentique`` (no extras) import any of them with no third-party package present.

2. **Three-layer arrows.** The package is layered ``console -> code -> core``: the
   generic framework (``core`` and its sibling seam packages), the generic
   domain-agnostic harness (``code``), and the human-facing application
   (``console``). The dependency arrow points one way. ``core`` must not import
   ``code`` or ``console``; ``code`` must not import ``console``. Policed here so
   the layering cannot quietly erode as the harness and application fill in.
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


# Which sibling subpackages each layer is forbidden from importing. The arrow runs
# console -> code -> core, so a layer may import only itself and the ones to its
# right; importing leftward (a lower layer reaching up) is the violation.
_FORBIDDEN_LAYER_IMPORTS = {
    "core": {"code", "console"},
    "code": {"console"},
    "console": set(),
}


def _imported_agentique_subpackages(tree: ast.Module) -> set[str]:
    """Second-level names of absolute ``agentique.<name>`` imports in ``tree``.

    ``import agentique.code.x`` and ``from agentique.code.x import y`` both yield
    ``code``. Relative imports (``level > 0``) stay within their own package, so
    they can never reach a *different* layer and are ignored, matching the
    third-party check above.
    """
    subpackages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "agentique" and len(parts) >= 2:
                    subpackages.add(parts[1])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            parts = node.module.split(".")
            if parts[0] == "agentique" and len(parts) >= 2:
                subpackages.add(parts[1])
    return subpackages


def test_layers_respect_the_dependency_arrows() -> None:
    violations: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        layer = path.relative_to(_SRC).parts[0]
        forbidden = _FORBIDDEN_LAYER_IMPORTS.get(layer)
        if not forbidden:
            continue
        imported = _imported_agentique_subpackages(ast.parse(path.read_text()))
        crossed = imported & forbidden
        if crossed:
            violations[str(path.relative_to(_SRC.parent.parent))] = crossed
    assert not violations, (
        "layer-boundary violations (arrow is console -> code -> core): "
        f"{ {k: sorted(v) for k, v in violations.items()} }"
    )
