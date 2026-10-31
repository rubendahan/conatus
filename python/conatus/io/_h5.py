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
    call (N3: h5py's "Unable to synchronously open file" reached the user)."""
    p = Path(path)
    if p.is_file():
        return p
    near = sorted(q.name for q in p.parent.glob(f"*{p.suffix}"))[:5] if p.parent.is_dir() else []
    raise MissingFileError(
        f"{what}: {str(path)!r} " + ("is a directory" if p.is_dir() else "does not exist"),
        why="this call reads an existing file; it does not create one",
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
                          why="cadences and counts must be integers",
                          fix=f"{name}=10, for example", doc=DOC)
    return int(n)


def refuse_indirect_storage(f: Any, p: Path | str, what: str = "this file") -> None:
    """Refuse an HDF5 file that reaches other files: external or soft links, datasets with external
    raw storage, virtual datasets. Metadata only: no payload is read, so call it first."""
    import h5py
    links: list[str] = []
    stored: list[str] = []
    visited: set[int] = set()     # a hard link may close a loop of groups

    def walk(group: Any, prefix: str) -> None:
        for name in group:
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                links.append(prefix + name)
                continue
            obj = group[name]
            if isinstance(obj, h5py.Group):
                addr = h5py.h5o.get_info(obj.id).addr
                if addr not in visited:
                    visited.add(addr)
                    walk(obj, prefix + name + "/")
            elif isinstance(obj, h5py.Dataset) and (
                    obj.is_virtual or obj.id.get_create_plist().get_external_count() > 0):
                stored.append(prefix + name)
    walk(f, "")
    if links:
        raise ValueError(f"{p}: links to other data refused in {what}: {links[:4]}")
    if stored:
        raise ValueError(f"{p}: datasets stored in other files refused in {what}: {stored[:4]}")
