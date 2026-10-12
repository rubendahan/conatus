"""HDF5 input/output (``cn.io``): analysis files, exact checkpoints, reading them back.

Native use::

    sim = m.simulate(N=..., scheme="RK443", dt=1e-3)
    sim.output("snapshots.h5", tasks=["T", "u.z"], every=100)          # grid values every 100 steps
    sim.output("scalars.h5", tasks={"KE": lambda s: ...}, every=10)    # functions of the simulation
    sim.checkpoints("chk/", every=1000, keep=2)                        # exact restart files
    sim.run(until=10.0)

    sim.restart(cn.io.latest("chk/"))      # next job: continues bit-for-bit
    data = cn.io.read("snapshots.h5")      # data.tasks["T"], data.sim_time, data.explain()
    ds = data.to_xarray()                  # xarray.Dataset with t, x, z, component coordinates
    # ParaView: open snapshots.xmf (Cartesian 2D/3D grids; cn.io.write_xdmf(path) rewrites it)

Every file carries a provenance record (executed code, environment, numerical choices) and is
readable with h5py alone (layouts in :mod:`conatus.io.analysis` and :mod:`conatus.io.checkpoint`):
every array names its dimensions (``dims`` attribute and HDF5 dimension scales), vectors name
their components, the root holds the Conatus version, the parameters and the coordinates
(:mod:`conatus.io._describe`).
Guide:.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._describe import COMPONENT, read_dims
from ._h5 import DOC, open_h5, read_json_attr
from .analysis import FORMAT as ANALYSIS_FORMAT
from .analysis import AnalysisFile
from .checkpoint import FORMAT as CHECKPOINT_FORMAT
from .checkpoint import CheckpointSeries, latest, read_checkpoint, write_checkpoint
from .xdmf import write_xdmf
from .._repr import compact_repr

__all__ = ["read", "latest", "AnalysisFile", "CheckpointSeries", "AnalysisData", "CheckpointData",
           "read_checkpoint", "write_checkpoint", "write_xdmf", "ANALYSIS_FORMAT",
           "CHECKPOINT_FORMAT", "explain"]


def _xarray():
    """The xarray module, or a MissingDependencyError saying how to install it (optional)."""
    try:
        import xarray
    except ImportError as err:                         # optional dependency
        from ..errors import MissingDependencyError
        raise MissingDependencyError("to_xarray() needs xarray (optional)",
                                     fix="pip install xarray (or micromamba install -c "
                                         "conda-forge xarray); the arrays are also in "
                                         ".tasks / .grids as NumPy arrays", doc=DOC) from err
    return xarray


def _infer_dims(name: str, shape: tuple, scales: dict, lead: str | None) -> list[str]:
    """Dimension names of an array of a file written before the self-describing layout: the
    trailing dimensions named after the coordinates when their lengths identify them uniquely,
    the rest ``<name>_dim<k>`` (never guessed beyond that)."""
    body = shape[1:] if lead else shape
    by_len: dict[int, list[str]] = {}
    for c, x in scales.items():
        by_len.setdefault(len(x), []).append(c)
    names = [f"{name}_dim{k}" for k in range(len(body))]
    used: set[str] = set()
    for k in range(len(body) - 1, -1, -1):
        cands = [c for c in by_len.get(body[k], []) if c not in used]
        if len(cands) != 1 or any(len(v) > 1 for v in by_len.values()):
            break
        names[k] = cands[0]
        used.add(cands[0])
    return ([lead] if lead else []) + names


def _code_line(prov: dict | None) -> str:
    code = (prov or {}).get("code") or {}
    head = code.get("git_head")
    if head is None:
        return "code: unknown commit" + (f" ({code['warning']})" if code.get("warning") else "")
    exact = "exactly this commit" if code.get("code_matches_commit") else \
        f"NOT exactly this commit ({code.get('warning', 'see provenance')})"
    return f"code: {str(head)[:12]}, {exact}"


@dataclass
class AnalysisData:
    """Contents of an analysis file (all rows loaded)."""
    __repr__ = compact_repr   # one line, arrays by shape
    path: Path
    tasks: dict[str, np.ndarray]
    sim_time: np.ndarray
    iteration: np.ndarray
    write_number: np.ndarray
    scales: dict[str, np.ndarray]
    attrs: dict[str, Any]
    sources: dict[str, str]
    provenance: dict = field(repr=False, default_factory=dict)
    # self-describing layout: dimension names per task, component names, parameters,
    # the Conatus version that wrote the file ({} / "" for files written before)
    dims: dict = field(default_factory=dict)
    component_names: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    conatus_version: str = ""

    def to_xarray(self):
        """An ``xarray.Dataset``: one variable per task with its dimensions (``t``, ``component``,
        the coordinates), coordinates ``t`` (simulation time, with ``iteration`` and
        ``write_number`` along it), each grid coordinate and ``component``; attributes: problem,
        scheme, dt, every, parameters, conatus_version, and each task's source. xarray is an
        optional dependency (a clear error says how to install it)."""
        xr = _xarray()
        coords: dict[str, Any] = {"t": ("t", np.asarray(self.sim_time)),
                                  "iteration": ("t", np.asarray(self.iteration)),
                                  "write_number": ("t", np.asarray(self.write_number))}
        coords.update({c: (c, np.asarray(x)) for c, x in self.scales.items()})
        data = {}
        for k, a in self.tasks.items():
            dims = self.dims.get(k) or _infer_dims(k, a.shape, self.scales, "t")
            attrs = {"source": self.sources.get(k, "")}
            if k in self.component_names:
                attrs["component_names"] = list(self.component_names[k])
                for d in dims:
                    if d.startswith("component") and d not in coords:
                        coords[d] = (d, np.array(self.component_names[k]))
            data[k] = (dims, np.asarray(a), attrs)
        attrs = {k: (v if isinstance(v, (str, int, float)) else str(v)) for k, v in self.attrs.items()
                 if k != "parameters"}                   # JSON below, as before
        attrs.update(conatus_version=self.conatus_version or "unknown (file written before 0.1 "
                     "self-describing layout)", parameters=json.dumps(self.parameters),
                     path=str(self.path))
        return xr.Dataset(data, coords=coords, attrs=attrs)

    def explain(self) -> str:
        """Rows, time span, problem, scheme, code state and the tasks."""
        n = len(self.sim_time)
        span = f"t = {self.sim_time[0]:g} .. {self.sim_time[-1]:g}" if n else "no row"
        lines = [f"Analysis file {self.path}: {n} write(s), {span}, problem "
                 f"'{self.attrs.get('problem')}', scheme {self.attrs.get('scheme')}, "
                 f"dt = {self.attrs.get('dt')}, every {self.attrs.get('every')} steps",
                 "  " + _code_line(self.provenance)]
        lines += [f"  {k}: {a.shape} {a.dtype}" + (f" dims {tuple(self.dims[k])}" if k in self.dims
                                                    else "") + f" ({self.sources.get(k, '?')})"
                  for k, a in self.tasks.items()]
        if not self.dims and self.tasks:
            lines.append("  dimension names: not stored (file written before the self-describing "
                         "layout); to_xarray() names what the coordinate lengths identify")
        if self.scales:
            lines.append("  coordinates: " + ", ".join(f"{c} ({len(x)})" for c, x in self.scales.items()))
        return "\n".join(lines)

    __str__ = explain


@dataclass
class CheckpointData:
    """Contents of a checkpoint file (state, history, per-unknown coefficients, optional grids)."""
    __repr__ = compact_repr   # one line, arrays by shape
    path: Path
    meta: dict
    fields: dict[str, np.ndarray]
    grids: dict[str, np.ndarray]
    provenance: dict = field(repr=False, default_factory=dict)
    dims: dict = field(default_factory=dict)            # "fields/<name>", "grids/<name>" -> dims
    component_names: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)          # coordinate -> grid points (grids=True)
    parameters: dict = field(default_factory=dict)
    conatus_version: str = ""

    def to_xarray(self):
        """An ``xarray.Dataset`` of the stored unknowns: grid values ``<name>`` (with
        ``grids=True``, on the coordinates) and spectral coefficients ``<name>_coeffs`` (dims
        ``<coord>_mode``); attributes: the clock, scheme, dt, parameters, conatus_version."""
        xr = _xarray()
        coords: dict[str, Any] = {c: (c, np.asarray(x)) for c, x in self.scales.items()}
        data = {}
        for group, arrays, suffix in (("grids", self.grids, ""), ("fields", self.fields, "_coeffs")):
            for k, a in arrays.items():
                dims = self.dims.get(f"{group}/{k}") or [f"{k}{suffix}_dim{i}" for i in range(a.ndim)]
                if k in self.component_names:
                    for d in dims:
                        if d.startswith("component") and d not in coords:
                            coords[d] = (d, np.array(self.component_names[k]))
                data[k + suffix] = (dims, np.asarray(a))
        m = self.meta
        attrs = {"problem": str(m.get("problem", "")), "scheme": str(m["scheme"]),
                 "dt": float(m["dt"]), "iteration": int(m["iteration"]), "sim_time": float(m["t"]),
                 "parameters": json.dumps(self.parameters), "path": str(self.path),
                 "conatus_version": self.conatus_version or "unknown (written before the "
                 "self-describing layout)"}
        return xr.Dataset(data, coords=coords, attrs=attrs)

    @property
    def sim_time(self) -> float:
        """Simulation time of the checkpoint."""
        return float(self.meta["t"])

    @property
    def iteration(self) -> int:
        """Step number of the checkpoint."""
        return int(self.meta["iteration"])

    def explain(self) -> str:
        """Scheme, clock, history levels, code state and the stored unknowns."""
        m = self.meta
        hist = m.get("nhist", 0)
        lines = [f"Checkpoint {self.path}: problem '{m.get('problem')}', scheme {m['scheme']}, "
                 f"dt = {m['dt']:g}, iteration {m['iteration']}, t = {m['t']:g}, "
                 f"{hist} history level(s)" + (f" + {m['nhist_LX']} L X" if m.get("nhist_LX") else ""),
                 "  exact restart: sim.restart(path) / solver.load_checkpoint(path) with the same "
                 "scheme, dt, resolution and options (checked)",
                 "  " + _code_line(self.provenance)]
        lines += [f"  {k}: coefficients {a.shape}" + (" + grid values" if k in self.grids else "")
                  for k, a in self.fields.items()]
        return "\n".join(lines)

    __str__ = explain


def read(path: str | Path) -> AnalysisData | CheckpointData:
    """Read a Conatus HDF5 file (analysis or checkpoint, told by its ``format`` attribute)."""
    from ._h5 import require_file
    path = require_file(path, "cn.io.read")
    with open_h5(path, "r") as f:
        fmt = f.attrs.get("format")
        fmt = fmt.decode() if isinstance(fmt, bytes) else fmt
        prov = read_json_attr(f, "provenance") or {}
        if fmt == ANALYSIS_FORMAT:
            n = int(f.attrs["writes"])                 # rows beyond the counter are incomplete
            sc = f["scales"]
            tasks = {k: f["tasks"][k][:n] for k in f["tasks"]}
            sources = {k: str(f["tasks"][k].attrs.get("source", "")) for k in f["tasks"]}
            scales = {k: sc[k][()] for k in sc if k not in ("sim_time", "iteration", "write_number")}
            attrs = {k: f.attrs[k] for k in ("problem", "scheme", "dt", "every")}
            attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in attrs.items()}
            # the model's parameters, name and geometry with the rest (critique
            # pass 4: they were in the HDF5 root and the xarray attrs, not in .attrs)
            attrs["parameters"] = read_json_attr(f, "parameters") or {}
            attrs.update({k: _str_attr(f, k) for k in ("model", "geometry") if k in f.attrs})
            scales.pop(COMPONENT, None)                # component names: an attribute per task
            coords = read_json_attr(f, "coordinates")
            if coords:                                 # grid order (h5py lists groups by name)
                scales = {c: scales[c] for c in coords if c in scales}
            dims = {k: d for k in f["tasks"] if (d := read_dims(f["tasks"][k])) is not None}
            comps = {k: read_json_attr(f["tasks"][k], "component_names") for k in f["tasks"]
                     if "component_names" in f["tasks"][k].attrs}
            return AnalysisData(path, tasks, sc["sim_time"][:n], sc["iteration"][:n],
                                sc["write_number"][:n], scales, attrs, sources, prov, dims, comps,
                                read_json_attr(f, "parameters") or {},
                                _str_attr(f, "conatus_version"))
        if fmt == CHECKPOINT_FORMAT:
            fields = {k: f["fields"][k][()] for k in f.get("fields", {})}
            grids = {k: f["grids"][k][()] for k in f.get("grids", {})}
            dims, comps = {}, {}
            for g in ("fields", "grids"):
                for k in f.get(g, {}):
                    d = read_dims(f[g][k])
                    if d is not None:
                        dims[f"{g}/{k}"] = d
                    if "component_names" in f[g][k].attrs:
                        comps[k] = read_json_attr(f[g][k], "component_names")
            coords = read_json_attr(f, "coordinates") or []
            scales = {c: f["scales"][c][()] for c in coords if "scales" in f and c in f["scales"]}
            return CheckpointData(path, read_json_attr(f, "meta"), fields, grids, prov, dims,
                                  comps, scales, read_json_attr(f, "parameters") or {},
                                  _str_attr(f, "conatus_version"))
    from ..errors import OptionError
    raise OptionError(f"cn.io.read({str(path)!r}): not a Conatus file (format {fmt!r})",
                      fix=f"a file written by sim.output / sim.save ({ANALYSIS_FORMAT}, "
                          f"{CHECKPOINT_FORMAT}); other HDF5 files: h5py", doc=DOC)


def _str_attr(f, key: str) -> str:
    v = f.attrs.get(key, "")
    return v.decode() if isinstance(v, bytes) else str(v)


def explain() -> str:
    """What cn.io writes and reads."""
    return __doc__ or ""
