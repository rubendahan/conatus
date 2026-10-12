"""Small HDF5 helpers shared by :mod:`conatus.io` (h5py is optional: imported on use)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import MissingDependencyError, MissingFileError

DOC = "docs/site/guide/io.md"


def h5py():
    """The h5py module, or a MissingDependencyError saying how to install it."""
    try:
        import h5py as _h5py
    except ImportError as err:                         # optional dependency
        raise MissingDependencyError("conatus.io needs h5py (HDF5 files)",
                                     fix="micromamba install -c conda-forge h5py "
                                         "(or pip install h5py)", doc=DOC) from err
    return _h5py


def require_file(path: str | Path, what: str) -> Path:
    """``path`` as a Path if it is an existing file, else a :class:`MissingFileError` naming the
    call (h5py's "Unable to synchronously open file" reached the user)."""
    p = Path(path)
    if p.is_file():
        return p
    near = sorted(q.name for q in p.parent.glob(f"*{p.suffix}"))[:5] if p.parent.is_dir() else []
    raise MissingFileError(
        f"{what}: {str(path)!r} " + ("is a directory" if p.is_dir() else "does not exist"),
        why="the file is read, never created, by this call",
        fix=("cn.io.latest(directory) gives the newest checkpoint of a directory"
             if p.is_dir() else
             "check the path (relative paths start from the working directory "
             f"{Path.cwd()})" + (f"; {p.suffix} files next to it: {', '.join(near)}" if near
                                 else "")),
        doc=DOC)


def open_h5(path: str | Path, mode: str):
    """h5py file without HDF5 file locking: one writer per file by construction, and locking is
    unreliable on network file systems (same choice as conatus.runs). A file opened for reading
    must exist (:func:`require_file`)."""
    if mode in ("r", "r+"):
        require_file(path, f"HDF5 file (mode {mode!r})")
    h = h5py()
    try:
        return h.File(path, mode, locking=False)
    except TypeError:                                  # h5py built without the locking argument
        return h.File(path, mode)


def json_attr(obj: Any) -> str:
    """JSON text of a metadata dict (non-JSON leaves as their repr: never dropped silently)."""
    return json.dumps(obj, default=repr, sort_keys=True)


def read_json_attr(f, key: str) -> Any:
    v = f.attrs.get(key)
    if v is None:
        return None
    return json.loads(v.decode() if isinstance(v, bytes) else str(v))


def provenance(solver: Any) -> dict:
    """Provenance record of a solver (code state, environment, numerics: conatus.provenance)."""
    from ..provenance import record
    return record(solver)


def positive(n: Any, where: str, name: str) -> int:
    """An integer >= 1 (a float or a bool is refused, never truncated)."""
    import numbers
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        from ..errors import OptionError
        raise OptionError(f"{where}: {name}={n!r} must be an integer >= 1",
                          why="cadences and counts are never rounded for you",
                          fix=f"{name}=10, for example", doc=DOC)
    return int(n)
