"""Audit every public function, class and method of the Studio and telemetry packages has a docstring."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = [ROOT / "python" / "conatus" / "studio", ROOT / "python" / "conatus" / "telemetry"]
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _missing(body, path: Path) -> list[str]:
    """Walk module/class bodies recursively; list public definitions without a docstring."""
    out = []
    for node in body:
        if not isinstance(node, DEFS) or node.name.startswith("_"):
            continue
        if ast.get_docstring(node) is None:
            out.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}")
        if isinstance(node, ast.ClassDef):
            out.extend(_missing(node.body, path))
    return out


def test_public_api_has_docstrings():
    """Every module and every public function, class and method carries a docstring."""
    missing = []
    for package in PACKAGES:
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if ast.get_docstring(tree) is None:
                missing.append(f"{path.relative_to(ROOT)}:1 <module>")
            missing.extend(_missing(tree.body, path))
    assert not missing, f"{len(missing)} missing docstrings:\n" + "\n".join(missing)
