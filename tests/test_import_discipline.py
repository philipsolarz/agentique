"""Import-discipline guards: the third-party boundary and the three-layer arrows.

Two structural rules, both enforced by AST so a violation fails the gate rather
than waiting to surprise a downstream user:

1. **Third-party boundary.** Collapsing the six distributions into one removed the
   dependency isolation that a separate distribution gave the Anthropic provider.
   Every module under ``src/agentique`` EXCEPT ``agentique.anthropic`` must import
   only the standard library, ``agentique`` itself, and the explicitly approved
   base dependencies (``_APPROVED_BASE_DEPS`` — today just ``pydantic``). That is
   what lets ``pip install agentique`` (no extras) import any of them with no
   provider SDK or telemetry stack present.

2. **Three-layer arrows.** The package is layered ``console -> harness -> core``:
   the generic framework (``core`` and its sibling seam packages ``tools``,
   ``memory``, ``testing``, ``anthropic``), the generic domain-agnostic harness
   (the loose modules at the ``agentique`` root — ``coordinator``, ``session``,
   ``store``, ``role``, ``artifact``), and the human-facing application
   (``console``). The dependency arrow points one way. ``core`` must not import
   ``harness`` or ``console``; ``harness`` must not import ``console``. Policed
   here so the layering cannot quietly erode as the harness and application fill
   in.

   Note on attribution: the harness layer is a set of *loose modules* directly
   under ``src/agentique/``, not a subpackage — so a file's layer cannot be read
   off its top directory alone (that would be the filename for a root module).
   :func:`_layer_of` maps a path to its layer; :func:`_layer_of_import` maps an
   imported ``agentique.<name>`` to the layer that name belongs to. The check
   compares layers, not raw names.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_SRC = (Path(__file__).resolve().parent.parent / "src" / "agentique").resolve()
_EXEMPT_DIR = _SRC / "anthropic"
# Approved *base* third-party deps: importable from any module without an extra.
# Adding one is a one-line, reviewed change here (and in pyproject's `dependencies`).
_APPROVED_BASE_DEPS = frozenset({"pydantic"})
# The canonical stdlib top-level names for this interpreter, our own package, and
# the approved base deps.
_ALLOWED_ROOTS = (
    frozenset(sys.stdlib_module_names) | {"agentique"} | _APPROVED_BASE_DEPS
)

# The loose modules that make up the harness layer at the `agentique` root.
_HARNESS_MODULES = frozenset({"artifact", "coordinator", "role", "session", "store"})


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
        "third-party imports found outside agentique.anthropic "
        f"(approved base deps: {sorted(_APPROVED_BASE_DEPS)}): "
        f"{ {k: sorted(v) for k, v in violations.items()} }"
    )


# Which layers each layer is forbidden from importing. The arrow runs
# console -> harness -> core, so a layer may import only itself and the ones to
# its right; importing leftward (a lower layer reaching up) is the violation.
_FORBIDDEN_LAYER_IMPORTS = {
    "core": {"harness", "console"},
    "harness": {"console"},
    "console": set[str](),
}


def _layer_of(path: Path) -> str:
    """The layer a source file belongs to, by its location under ``src/agentique``.

    A loose ``*.py`` directly at the root is a harness module; ``console/`` is the
    application; ``core/`` and the satellite seam packages (``tools``, ``memory``,
    ``testing``, ``anthropic``) are the framework/core tier.
    """
    head = path.relative_to(_SRC).parts[0]
    if head.endswith(".py"):
        return "harness"
    if head == "console":
        return "console"
    return "core"


def _layer_of_import(name: str) -> str:
    """The layer an imported ``agentique.<name>`` second-level name belongs to."""
    if name == "console":
        return "console"
    if name in _HARNESS_MODULES:
        return "harness"
    return "core"


def _imported_agentique_layers(tree: ast.Module) -> set[str]:
    """Layers reached by absolute ``agentique.<name>`` imports in ``tree``.

    ``import agentique.coordinator`` and ``from agentique.coordinator import y``
    both reach the ``harness`` layer. Relative imports (``level > 0``) stay within
    their own package, so they can never reach a *different* layer and are ignored,
    matching the third-party check above.
    """
    layers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "agentique" and len(parts) >= 2:
                    layers.add(_layer_of_import(parts[1]))
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            parts = node.module.split(".")
            if parts[0] == "agentique" and len(parts) >= 2:
                layers.add(_layer_of_import(parts[1]))
    return layers


def test_layers_respect_the_dependency_arrows() -> None:
    violations: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        forbidden = _FORBIDDEN_LAYER_IMPORTS[_layer_of(path)]
        if not forbidden:
            continue
        crossed = _imported_agentique_layers(ast.parse(path.read_text())) & forbidden
        if crossed:
            violations[str(path.relative_to(_SRC.parent.parent))] = crossed
    assert not violations, (
        "layer-boundary violations (arrow is console -> harness -> core): "
        f"{ {k: sorted(v) for k, v in violations.items()} }"
    )
