"""Runtime: Python face of the C++ core ``conatus._core`` (host backend).

Where the compiled module comes from (first match wins, never a silent fallback):

1. ``$CONATUS_CORE_PATH`` (explicit directory holding ``_core*.so``): if set, it *must* contain the
   module, otherwise the import fails (no fall-through to another copy).
2. The installed copy, found by the normal import system as ``conatus._core`` (``pip install .``,
   wheel, conda package; editable installs too). It was compiled from the same sources as the
   Python package it ships with, so it is preferred over any build tree.
3. The development build ``$CN_BUILD/core/core`` (``bash core/build.sh``; default
   ``<repo>/scratch/build`` only when this package is the repository's ``python/conatus``).
   In a dev checkout there is no ``.so`` inside ``python/conatus`` (never copied there, gitignored),
   so step 2 finds nothing and the behaviour is unchanged.

``CORE_SOURCE`` (``"CONATUS_CORE_PATH"``, ``"installed"`` or ``"dev-build"``) and ``CORE_FILE``
say which copy was loaded; ``conatus.provenance`` records both.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path

from ..errors import MissingDependencyError

_FIX = ("development checkout: `source env/env.sh && bash core/build.sh` (builds into $CN_BUILD); "
        "installed use: install a conatus binary wheel (documentation, Install page), or "
        "`pip install .` from the repository in a conda env with liblapack and libblas (compiles "
        "the core; see pyproject.toml)")


def _missing(message: str, where: str, why: str) -> MissingDependencyError:
    return MissingDependencyError(message, where=where, why=why, fix=_FIX)


_PKG_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PKG_DIR.parents[1]
_IN_REPO = (_REPO_ROOT / "core" / "CMakeLists.txt").is_file() and _PKG_DIR.parent.name == "python"


def _exec_from(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _load_core():
    """Return ``(module, source)``; see the module docstring for the search order."""
    name = "conatus._core"
    if name in sys.modules:
        return sys.modules[name], "preloaded"
    explicit = os.environ.get("CONATUS_CORE_PATH", "")
    if explicit:
        hits = sorted(glob.glob(os.path.join(explicit, "_core*.so")))
        if not hits:
            raise _missing(f"CONATUS_CORE_PATH={explicit!r} contains no _core*.so",
                           where="conatus.runtime (CONATUS_CORE_PATH)",
                           why="an explicit core path is never replaced by another copy; unset it "
                               "to use the installed or development build")
        return _exec_from(name, hits[0]), "CONATUS_CORE_PATH"
    if importlib.util.find_spec(name) is not None:
        return importlib.import_module(name), "installed"
    build = os.environ.get("CN_BUILD") or (str(_REPO_ROOT / "scratch" / "build") if _IN_REPO else "")
    if build:
        hits = sorted(glob.glob(os.path.join(build, "core", "core", "_core*.so")))
        if hits:
            return _exec_from(name, hits[0]), "dev-build"
    looked = "the installed package" + (f", $CN_BUILD/core/core = {build}/core/core" if build else "")
    raise _missing("conatus._core (compiled C++ core) not found", where=f"looked in {looked}",
                   why="the compiled core is neither installed nor built")


_core, CORE_SOURCE = _load_core()
CORE_FILE = getattr(_core, "__file__", None)

from .banded import BorderedBanded, BorderedBandedBatch  # noqa: E402
from .imex import ImexIntegrator  # noqa: E402

__all__ = ["_core", "CORE_SOURCE", "CORE_FILE", "BorderedBanded", "BorderedBandedBatch",
           "ImexIntegrator"]
