"""Self-describing HDF5: dimension names, scales and metadata.

Every Conatus HDF5 file says, with plain HDF5 means, what its arrays are:

* root attributes ``conatus_version``, ``parameters`` (JSON: name -> value of the declared
  parameters), ``coordinates`` (JSON: coordinate names in grid order), ``geometry`` (text) and
  ``cartesian`` (True for a box or a channel: what the XDMF sidecar covers);
* a ``dims`` attribute on every array (JSON list of dimension names, e.g. ``["t", "component",
  "x", "z"]``), and HDF5 **dimension scales** (``H5DS``: ``make_scale`` / ``attach_scale``) linking
  each labelled dimension to its coordinate dataset (``/scales/<coord>``, ``/scales/sim_time`` for
  the time dimension ``t``, ``/scales/component`` for vector components);
* a ``component_names`` attribute on vector and tensor arrays (the coordinate names).

No unit attribute is written: native models are nondimensional and declare no units (a model
that declares them would add ``units`` here). Files written before this layout keep reading
(``cn.io.read`` infers what it can and says so).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

COMPONENT = "component"


def version() -> str:
    """The Conatus version written into every file."""
    from .. import __version__
    return str(__version__)


def parameters_of(solver: Any) -> dict:
    """Declared parameters of the problem with their values (numbers only; the eigenvalue and
    the time, which have none, are left out): the free parameters of the equations, overridden
    by ``plan.params`` (the values the solver reads at run time)."""
    out: dict = {}
    vals: dict = {}
    try:
        for eq in solver.plan.problem.equations:
            for prm in eq.lhs.free_parameters() | eq.rhs.free_parameters():
                vals[prm.name] = prm.value
    except AttributeError:
        pass
    vals.update({str(k): v for k, v in
                 dict(getattr(getattr(solver, "plan", None), "params", None) or {}).items()})
    for k, v in sorted(vals.items()):
        if isinstance(v, (bool, np.bool_)):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[str(k)] = float(v)
        elif isinstance(v, (complex, np.complexfloating)):
            out[str(k)] = [float(np.real(v)), float(np.imag(v))]
    return out


def solver_coords(solver: Any) -> list[str]:
    """Coordinate names of a solver's domain, in grid order."""
    try:
        return [c.name for c in solver.plan.domain.coordsys.coords]
    except AttributeError:
        return []


def solver_grids(solver: Any) -> dict[str, np.ndarray]:
    """1D grid points of each coordinate matching ``solver.grid(name)`` (the solver's own grid,
    full period on a sine/cosine axis); {} when they cannot be told."""
    names = solver_coords(solver)
    plan = getattr(solver, "plan", None)
    space = getattr(plan, "space", None)
    try:
        if space is not None and hasattr(space, "grids"):
            g = space.grids(False)
            return {n: np.asarray(g[i]).ravel() for i, n in enumerate(names)}
        from ..reference.transforms import grid as _grid
        bases = {b.coord.name: b for b in plan.domain.bases}
        return {n: np.asarray(_grid(bases[n], dealias=False)) for n in names if n in bases}
    except Exception:  # noqa: BLE001 -- coordinates are metadata: their absence is declared
        return {}


def solver_modes(solver: Any) -> tuple:
    """Number of coefficients of each coordinate of a full-domain unknown (coefficient arrays);
    () when the layout is not a plain product of bases (curvilinear spaces)."""
    try:
        bases = {b.coord.name: b for b in solver.plan.domain.bases}
        return tuple(int(bases[c].size) for c in solver_coords(solver))
    except Exception:  # noqa: BLE001 -- labels are metadata: unlabelled dimensions are declared
        return ()


def dims_for(shape: tuple, coords: list[str], grid: tuple, *, rank: int | None,
             lead: str | None = "t", name: str = "data") -> list[str]:
    """Dimension names of an array of ``shape`` (without the leading row dimension when ``lead``
    is given): ``coords`` for trailing dimensions that match ``grid``; ``component`` (rank 1) or
    ``component_i``/``component_j`` (rank 2) before them for a vector/tensor of known ``rank``;
    ``<name>_dim<k>`` for anything else (unlabelled, never guessed)."""
    shape = tuple(shape)
    out: list[str] = []
    nd = len(grid)
    if grid and nd and shape[len(shape) - nd:] == tuple(grid) and len(shape) >= nd:
        extra = len(shape) - nd
        if rank and extra == rank and all(s == len(coords) for s in shape[:extra]):
            out = [COMPONENT] if rank == 1 else [f"{COMPONENT}_{a}" for a in "ijkl"[:rank]]
        else:
            out = [f"{name}_dim{k}" for k in range(extra)]
        out += list(coords)
    else:
        out = [f"{name}_dim{k}" for k in range(len(shape))]
    return ([lead] if lead else []) + out


def label(dset: Any, dims: list[str], scales: dict[str, Any], rank: int | None = None,
          coords: list[str] | None = None) -> None:
    """Write ``dims`` (attribute), HDF5 dimension labels and attached scales on ``dset``."""
    dset.attrs["dims"] = json.dumps(list(dims))
    for i, d in enumerate(dims):
        dset.dims[i].label = d
        sc = scales.get(d)
        # the time scale grows with the rows (both are resized together); a coordinate scale
        # is attached only when its length is that of the dimension
        if sc is not None and (d == "t" or sc.shape[0] == dset.shape[i]):
            dset.dims[i].attach_scale(sc)
    if rank and coords:
        dset.attrs["component_names"] = json.dumps(list(coords))


def make_scale(ds: Any, name: str) -> Any:
    """Turn ``ds`` into a dimension scale named ``name`` (idempotent)."""
    if not getattr(ds, "is_scale", False):
        ds.make_scale(name)
    return ds


def root_attrs(f: Any, solver: Any, source: Any = None) -> None:
    """``conatus_version``, ``parameters``, ``coordinates``, ``geometry``, ``cartesian``."""
    f.attrs["conatus_version"] = version()
    f.attrs["parameters"] = json.dumps(parameters_of(solver), sort_keys=True)
    coords = solver_coords(solver)
    comp = getattr(source, "compiled", None)
    if comp is not None:
        geo = comp.model.geometry
        coords = list(geo.coord_names)
        f.attrs["geometry"] = repr(geo)
        f.attrs["model"] = str(comp.model.name)
        f.attrs["cartesian"] = bool(_is_cartesian(geo))
    else:
        f.attrs["cartesian"] = False      # expert solver: the geometry is not told here
    f.attrs["coordinates"] = json.dumps(coords)


def _is_cartesian(geo: Any) -> bool:
    """A rectilinear grid (box, channel): what XDMF RectMesh can describe. Mapped boxes
    (curved walls), disks, pipes, spheres, balls and shells are not."""
    from ..model.geometry import Box
    try:
        from ..model.terrain import MappedBox
    except ImportError:                      # pragma: no cover - always present in the package
        MappedBox = ()                       # noqa: N806
    return isinstance(geo, Box) and not isinstance(geo, MappedBox)


def read_dims(dset: Any) -> list[str] | None:
    """The ``dims`` attribute of a dataset (None for files written by versions before 0.1)."""
    v = dset.attrs.get("dims")
    if v is None:
        return None
    return list(json.loads(v.decode() if isinstance(v, bytes) else str(v)))
