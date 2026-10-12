"""HDF5 checkpoints of an IVPSolver: exact restart, readable without Conatus.

Layout of a checkpoint file (``format = "conatus.checkpoint/1"``)::

    /                attrs: format, meta (JSON: scheme, dt, iteration, t, history lengths, ...),
                            provenance (JSON: code, environment, numerics), problem, scheme,
                            dt, sim_time, iteration
    /state/X         the solver state vector (spectral coefficients and tau unknowns, solver order)
    /history/MX<i>   multistep history M X^{n-i}, F(X^{n-i}) and, CN family, L X^{n-i}
    /history/F<i>, /history/LX<i>
    /fields/<name>   spectral coefficients of each unknown (expert layout), for reading only;
                     attrs ``dims`` (e.g. ["component", "x_mode", "z_mode"]), ``component_names``
    /grids/<name>    grid values of each unknown (only with grids=True), for reading only;
                     attrs ``dims``, ``component_names``; dimensions attached to /scales
    /scales/<coord>  grid points of each coordinate (with grids=True; HDF5 dimension scales),
    /scales/component  the coordinate names (component axis of vectors)

Root attributes also carry ``conatus_version``, ``parameters`` (JSON), ``coordinates`` (JSON,
grid order) and, for a native simulation, ``geometry`` and ``model`` (self-describing layout: ``cn.io.read(path).to_xarray()``; files written before keep reading and restarting).

Restart reads ``/state`` and ``/history`` only: the same arrays as the ``.npz`` format, stored
without conversion, so a restart is bit-for-bit (tests/output/test_io_hdf5.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import OptionError
from . import _describe as dsc
from ._h5 import DOC, json_attr, open_h5, provenance, read_json_attr

FORMAT = "conatus.checkpoint/1"


def write_checkpoint(path: str | Path, meta: dict, arrays: dict[str, np.ndarray], *,
                     solver: Any = None, grids: bool = False, prov: dict | None = None) -> Path:
    """Write one checkpoint (atomic: temporary file, then rename). ``prov``: a provenance
    record computed once by the caller (periodic checkpoints); else it is computed here."""
    path = Path(path)
    part = path.with_name(path.name + ".part")
    if prov is None and solver is not None:
        prov = provenance(solver)
    with open_h5(part, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["meta"] = json_attr(meta)
        f.attrs["provenance"] = json_attr(prov or {})
        f.attrs["problem"] = str(meta.get("problem", ""))
        f.attrs["scheme"] = str(meta["scheme"])
        f.attrs["dt"] = float(meta["dt"])
        f.attrs["sim_time"] = float(meta["t"])
        f.attrs["iteration"] = int(meta["iteration"])
        st, hi = f.create_group("state"), f.create_group("history")
        for k, a in arrays.items():
            (st if k == "X" else hi).create_dataset(k, data=np.asarray(a))
        if solver is not None:
            dsc.root_attrs(f, solver)
            coords = dsc.solver_coords(solver)
            pts = dsc.solver_grids(solver) if grids else {}
            scales = {}
            if pts and len(pts) == len(coords):
                sc = f.create_group("scales")
                scales = {c: dsc.make_scale(sc.create_dataset(c, data=pts[c]), c) for c in coords}
                scales[dsc.COMPONENT] = dsc.make_scale(
                    sc.create_dataset(dsc.COMPONENT, data=np.array(coords, dtype="S")), dsc.COMPONENT)
            grid = tuple(len(pts[c]) for c in coords) if scales else ()
            modes = dsc.solver_modes(solver)
            fl = f.create_group("fields")
            gr = f.create_group("grids") if grids else None
            for v in solver.system.variables:
                a = np.asarray(solver.coeffs(v.name))
                d = fl.create_dataset(v.name, data=a)
                dsc.label(d, dsc.dims_for(a.shape, [f"{c}_mode" for c in coords], modes,
                                          rank=v.rank, lead=None, name=v.name), {},
                          rank=v.rank, coords=coords)
                if gr is not None:
                    g = np.asarray(solver.grid(v.name))
                    dg = gr.create_dataset(v.name, data=g)
                    dsc.label(dg, dsc.dims_for(g.shape, coords, grid, rank=v.rank, lead=None,
                                               name=v.name), scales, rank=v.rank, coords=coords)
    os.replace(part, path)
    return path


def read_checkpoint(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """(meta, arrays) of a checkpoint file, for ``IVPSolver.load_checkpoint``."""
    path = Path(path)
    with open_h5(path, "r") as f:
        fmt = f.attrs.get("format")
        fmt = fmt.decode() if isinstance(fmt, bytes) else fmt
        if fmt != FORMAT:
            raise OptionError(f"{path}: not a Conatus checkpoint (format {fmt!r}, expected {FORMAT!r})",
                              why="an analysis file holds grid values, not the solver state and "
                                  "multistep history an exact restart needs",
                              fix="restart from a file written by save_checkpoint / sim.save "
                                  "(or sim.checkpoints)", doc=DOC)
        meta = read_json_attr(f, "meta")
        arrays = {"X": f["state/X"][()]}
        for k in f["history"]:
            arrays[k] = f["history"][k][()]
    return meta, arrays


#: The default pattern of :func:`latest`; with it the NumPy checkpoints ``checkpoint_*.npz``
#: (``sim.save(directory / "checkpoint_<iteration>.npz")``, the format the GPU guide recommends)
#: are searched too (they were never found, so a GPU chain restarted from t = 0).
DEFAULT_PATTERN = "checkpoint_*.h5"
_NPZ_PATTERN = "checkpoint_*.npz"


def checkpoint_iteration(p: Path) -> int | None:
    """The iteration a checkpoint file holds, read from the file (HDF5 attribute or the ``meta``
    of a ``.npz``); None for an unreadable or foreign file and for an unfinished ``.part``."""
    if ".part" in p.name:
        return None
    try:
        if p.suffix == ".npz":
            import json
            with np.load(p) as d:
                return int(json.loads(str(d["meta"]))["iteration"])
        with open_h5(p, "r") as f:
            return int(f.attrs["iteration"])
    except (OSError, KeyError, ValueError, TypeError):    # unreadable or foreign file: skipped
        return None


def _candidates(d: Path, pattern: str) -> list[Path]:
    pats = (pattern, _NPZ_PATTERN) if pattern == DEFAULT_PATTERN else (pattern,)
    return sorted({p for pat in pats for p in d.glob(pat)})


def latest(directory: str | Path, pattern: str = DEFAULT_PATTERN) -> Path | None:
    """The checkpoint of a directory with the largest iteration (read from the file, not the
    name), or None if the directory holds none. Unfinished ``.part`` files are ignored. The
    default pattern ``"checkpoint_*.h5"`` searches both formats: ``checkpoint_*.h5``
    (``sim.checkpoints``) and its NumPy twin ``checkpoint_*.npz`` (``sim.save``); a restart from
    either is bit-for-bit. Any other pattern is that glob only. Two files
    of the same iteration in the two formats are refused (which one is the run's is not
    known). A directory that does not exist is refused (it returned
    None, so a misspelt directory restarted a chain from t = 0 without a word)."""
    d = Path(directory)
    if not d.is_dir():
        raise OptionError(f"cn.io.latest({str(directory)!r}): "
                          + ("not a directory" if d.exists() else "no such directory"),
                          why="a missing directory is not an empty one: returning None would "
                              "restart the run from its initial state",
                          fix="check the path; the first job of a chain creates it "
                              "(sim.checkpoints(directory, ...) or sim.resume_or_start(directory, "
                              "...))", doc=DOC)
    found = [(it, p) for p in _candidates(d, pattern)
             if (it := checkpoint_iteration(p)) is not None]
    if not found:
        return None
    best_it = max(it for it, _ in found)
    best = [p for it, p in found if it == best_it]
    if len(best) > 1:
        raise OptionError(f"cn.io.latest({str(directory)!r}): {len(best)} checkpoints of "
                          f"iteration {best_it}: " + ", ".join(p.name for p in best),
                          why="the latest state exists in two files; which one this run "
                              "continues cannot be told",
                          fix="keep one format per chain (move the other file away)", doc=DOC)
    return best[0]


#: Number of periodic checkpoints kept by default: the
#: latest one to restart from and the one before it, in case a job is killed while writing (a
#: file is deleted only after the next one is complete). A storage count, not a numerical choice:
#: it changes no result. It used to be a required argument that every script gave as keep=2.
CHECKPOINT_KEEP = 2


class CheckpointSeries:
    """Periodic checkpoints ``<directory>/checkpoint_<iteration>.h5`` every ``every`` steps,
    keeping the ``keep`` most recent (default :data:`CHECKPOINT_KEEP` = 2; older ones deleted
    after the new one is complete).

    A multi-job run resumes with ``sim.restart(cn.io.latest(directory))``."""

    def __init__(self, solver: Any, directory: str | Path, *, every: int,
                 keep: int = CHECKPOINT_KEEP, grids: bool = False) -> None:
        from ._h5 import positive
        self.solver, self.directory = solver, Path(directory)
        self.every = positive(every, "checkpoints()", "every")
        self.keep = positive(keep, "checkpoints()", "keep")
        self.grids = bool(grids)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._prov = provenance(solver)                # once: hashing the code tree per write
        self.written: list[Path] = []                  # would cost more than the write

    def _adopt(self) -> None:
        """Files already in the directory, looked at before the first write (they
        used to be ignored, so a new run silently replaced the files of another run
        with the same names, ``cn.io.latest`` could then pick a file of the other run, and ``keep``
        never applied to the files of the previous job). Allowed only when this run continues
        from a checkpoint of this directory: its files up to the restart iteration are adopted
        (``keep`` applies to them); any later file, or any file of another run, is refused."""
        self._adopted = True
        found = []
        for p in _candidates(self.directory, DEFAULT_PATTERN):   # .h5 and .npz (latest finds both)
            if p in self.written:
                continue
            it = checkpoint_iteration(p)
            if it is None:                             # unreadable or foreign: never deleted,
                continue                               # never chosen by latest() either
            found.append((it, p))
        if not found:
            return
        src = getattr(self.solver, "_restart_path", None)
        here = src is not None and Path(src).resolve().parent == self.directory.resolve()
        if not here:
            raise OptionError(
                f"checkpoints({str(self.directory)!r}): the directory already holds "
                f"{len(found)} checkpoint(s) (iterations {', '.join(str(i) for i, _ in found)})",
                why="they belong to another run (this one did not restart from them): writing "
                    "here would replace some and let cn.io.latest pick the others",
                fix=f"sim.restart(cn.io.latest({str(self.directory)!r})) to continue that run, "
                    "or an empty directory", doc=DOC)
        it0 = int(self.solver._restart_iteration)
        later = [p.name for i, p in found if i > it0]
        if later:
            raise OptionError(
                f"checkpoints({str(self.directory)!r}): restarted from iteration {it0} but the "
                f"directory holds later checkpoints ({', '.join(later)})",
                why="they would be replaced by this run, or chosen by cn.io.latest instead of it",
                fix="restart from cn.io.latest(directory), or move those files away", doc=DOC)
        self.written = [p for _, p in sorted(found)] + self.written

    def write(self) -> Path:
        """Write a checkpoint of the current state now, then delete beyond the ``keep`` newest."""
        if not getattr(self, "_adopted", False):
            self._adopt()
        if getattr(self.solver, "_finite_every", 0):     # never a NaN checkpoint
            from ..timestep.integrator import solver_finite_check     # collective if distributed
            solver_finite_check(self.solver, f"checkpoints({str(self.directory)!r})")
        meta, arrays = self.solver._checkpoint_payload()
        p = write_checkpoint(self.directory / f"checkpoint_{self.solver.iteration:09d}.h5", meta,
                             arrays, solver=self.solver, grids=self.grids, prov=self._prov)
        if p not in self.written:
            self.written.append(p)
        while len(self.written) > self.keep:
            old = self.written.pop(0)
            old.unlink(missing_ok=True)
        return p

    def __call__(self, solver: Any) -> None:           # callback of IVPSolver.run
        if solver.iteration % self.every == 0:
            self.write()

    def explain(self) -> str:
        """Directory, cadence, how many are kept and what each file holds."""
        return (f"CheckpointSeries: {self.directory}/checkpoint_<iteration>.h5 every {self.every} "
                f"steps, the {self.keep} most recent kept"
                + (", with grid values" if self.grids else "")
                + "; each file = state + multistep history + clock (exact restart) + provenance")

    __str__ = explain
