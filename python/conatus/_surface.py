"""The public API surface of Conatus as text (v0.1 freeze).

One line per public object: the user layer ``cn.*`` (``conatus.__all__``) and its utility modules
(``cn.problems``, ``cn.runs``, ``cn.io``, ...), the expert layer ``conatus.expert.__all__``, with
the call signature of every function and class and of every public method of a class. Defaults
are part of the signature: a changed default is a changed API (never change silently).

The development test suite compares this text with a frozen snapshot and fails on any
difference, printing it. After a deliberate
change (an addition, or a deprecation declared in CHANGELOG.md)::

    python -m conatus._surface --write <snapshot.txt>
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys
import types
import warnings

# user-layer utility modules whose public names are part of the surface (``cn.<module>``)
USER_MODULES = ("model", "problems", "problems.native", "runs", "io", "notebook", "resolution",
                "tolerances", "errors")

# Classes that a public constructor returns but whose keywords its own signature hides:
# ``IVPSolver(plan, ...)`` dispatches by ``__new__`` to the polar and spherical solvers, whose
# extra keywords (``project_m0=``, ``border=``) are part of what a user writes.
DISPATCHED = (("expert.IVPSolver[polar]", "conatus.timestep.polar", "PolarIVPSolver"),
              ("expert.IVPSolver[sphere]", "conatus.timestep.sphere", "SphereIVPSolver"))

# Optional-extra modules a documented script imports directly; their public names are frozen
# too (none in this release).
EXTRA_MODULES: tuple[str, ...] = ()

_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _sig(obj) -> str:
    try:
        s = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    return _ADDR.sub("", s)


def _renamed(obj) -> str:
    """`` [renamed: old -> new, ...]`` for a callable with declared aliases (conatus._deprecation
    .RENAMES): the former names stay part of the frozen surface until v0.2."""
    from ._deprecation import aliases_of
    try:
        al = aliases_of(obj)
    except Exception:  # noqa: BLE001 -- an object without aliases
        return ""
    return (" [renamed: " + ", ".join(f"{o} -> {n}" for o, n in sorted(al.items())) + "]") if al else ""


def _describe(prefix: str, name: str, obj) -> list[str]:
    if isinstance(obj, types.ModuleType):
        return [f"{prefix}{name}: module {obj.__name__}"]
    if inspect.isclass(obj):
        lines = [f"{prefix}{name}: class {_sig(obj)}{_renamed(obj)}"]
        for k in sorted(vars(obj)):
            if k.startswith("_"):
                continue
            v = inspect.getattr_static(obj, k)
            if isinstance(v, property):
                lines.append(f"{prefix}{name}.{k}: property")
            elif isinstance(v, (staticmethod, classmethod)):
                lines.append(f"{prefix}{name}.{k}: {type(v).__name__} {_sig(getattr(obj, k))}")
            elif inspect.isfunction(v):
                lines.append(f"{prefix}{name}.{k}: method {_sig(v)}{_renamed(v)}")
        return lines
    if callable(obj):
        return [f"{prefix}{name}: function {_sig(obj)}{_renamed(obj)}"]
    if isinstance(obj, (bool, int, float, complex, str)) and len(repr(obj)) < 60:
        return [f"{prefix}{name}: {type(obj).__name__} = {obj!r}"]
    return [f"{prefix}{name}: {type(obj).__name__}"]


def surface() -> str:
    """The sorted text of the public surface (deterministic: no memory addresses)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import conatus as cn
        import conatus.expert as X
        lines = []
        for n in sorted(cn.__all__):
            lines += _describe("cn.", n, getattr(cn, n))
        for m in USER_MODULES:
            mod = importlib.import_module(f"conatus.{m}")
            for n in sorted(getattr(mod, "__all__", [])):
                lines += _describe(f"cn.{m}.", n, getattr(mod, n))
        for n in sorted(X.__all__):
            lines += _describe("expert.", n, getattr(X, n))
        for label, mod, cls in DISPATCHED:
            # the subclass inherits IVPSolver.__signature__: read its own __init__ instead
            init = getattr(importlib.import_module(mod), cls).__init__
            lines.append(f"{label}: class {_sig(init)}")
        for m in EXTRA_MODULES:
            mod = importlib.import_module(f"conatus.{m}")
            names = getattr(mod, "__all__", None) or [
                n for n, v in vars(mod).items() if not n.startswith("_")
                and getattr(v, "__module__", None) == mod.__name__
                and (inspect.isclass(v) or inspect.isfunction(v))]
            for n in sorted(names):
                lines += _describe(f"{m}.", n, getattr(mod, n))
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    text = surface()
    if len(argv) == 2 and argv[0] == "--write":
        with open(argv[1], "w") as f:
            f.write(text)
        print(f"wrote {text.count(chr(10))} lines to {argv[1]}")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
