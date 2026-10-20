"""The content of the ``conatus`` modules this process actually imported (provenance).

``provenance.code_state()`` and the run-cache key hash the package files on disk. They describe
the code that runs only if no imported module changed on disk since it was imported (an edit, a
``git pull`` or a checkout during a session). This module records, for every ``conatus`` module
the process imports, the sha256 of its source text at import time; ``provenance
.changed_since_import()`` compares it with the file now (audit state review F8:
the first version compared modification times with a 2 s clock slack, so an edit within 2 s of the
import or a deleted module was not seen, and a ``touch`` or a checkout round trip with the same
text was reported).

Cost: one read and one sha256 of each module's source when it is imported (the module is read
from its ``.pyc`` otherwise), only for the modules imported, never for the whole tree. A module
loaded without a source file (``.pyc`` only, zip) is not recorded and cannot be checked.
"""
from __future__ import annotations

import hashlib
import importlib.machinery as _machinery
import os
import sys
import threading

#: absolute source path -> (size, mtime in ns, sha256 hex) of the text the module was loaded from
#: (size and mtime kept for diagnosis only: the comparison is by content)
IMPORTED: dict[str, tuple[int, int, str]] = {}
_LOCK = threading.Lock()


def record(path: str) -> None:
    """Record the current content of source file ``path`` (read errors: not recorded)."""
    try:
        st = os.stat(path)
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return
    with _LOCK:
        IMPORTED[os.path.abspath(path)] = (st.st_size, st.st_mtime_ns, hashlib.sha256(data).hexdigest())


def changed() -> list[tuple[str, str]]:
    """(absolute path, "edited" or "deleted") of the recorded modules whose source differs now,
    by content (a ``touch`` or a checkout back to the same text is not a change). Every recorded
    file is hashed again: no shortcut on size and modification time, which a same-size edit
    within one tick of the file system's coarse clock leaves unchanged (seen in the CI run of
    this review). Cost: a fraction of ``provenance.code_state``'s hash of the whole tree."""
    with _LOCK:
        items = sorted(IMPORTED.items())
    out = []
    for path, (_size, _mtime, sha) in items:
        try:
            with open(path, "rb") as fh:
                now = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            out.append((path, "deleted"))
            continue
        if now != sha:
            out.append((path, "edited"))
    return out


class _RecordingLoader(_machinery.SourceFileLoader):
    """The standard source loader, plus :func:`record` of the source before it is compiled or its
    ``.pyc`` read (recorded first: a change during the import is then reported, never hidden)."""

    def exec_module(self, module) -> None:
        record(self.path)
        super().exec_module(module)


class _Finder:
    """Meta-path finder for the submodules of ``conatus``: the standard path search, with the
    recording loader in place of the plain source loader. Every other name is left to the
    other finders."""

    @classmethod
    def find_spec(cls, name: str, path=None, target=None):
        if not name.startswith("conatus.") or path is None:
            return None
        spec = _machinery.PathFinder.find_spec(name, path, target)
        if spec is not None and type(spec.loader) is _machinery.SourceFileLoader:
            spec.loader = _RecordingLoader(spec.loader.name, spec.loader.path)
        return spec

    @classmethod
    def invalidate_caches(cls) -> None:
        """Nothing cached here (PathFinder keeps its own caches)."""


def install(*already: str | None) -> None:
    """Record the modules imported before the finder existed (``already``), then put the finder
    first on ``sys.meta_path`` (once per process)."""
    for p in already:
        if p:
            record(p)
    if not any(f is _Finder for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder)
