"""HDF5 analysis files: declared quantities written every k steps, readable without Conatus.

Layout (``format = "conatus.analysis/1"``), close to Dedalus' file handlers so post-processing
scripts carry over::

    /                  attrs: format, problem, scheme, dt, every, tasks (JSON: name -> source),
                              provenance (JSON: code, environment, numerics), conatus_version,
                              parameters (JSON), coordinates (JSON, grid order), geometry, model,
                              cartesian, identity (JSON: the canonical identity of
                              the computation, ``conatus.io._identity``), task_identity (JSON:
                              name -> the source text or the code of the task function)
    /scales/sim_time   (W,)   simulation time of each write (dimension scale "t")
    /scales/iteration  (W,)   step number of each write
    /scales/write_number (W,) 1, 2, ... (continued by mode="a")
    /scales/<coord>    grid points of each coordinate (native simulations; dimension scale)
    /scales/component  coordinate names, the component axis of vectors (dimension scale)
    /tasks/<name>      (W, ...) one row per write: grid values of an unknown or component
                       (``"T"``, ``"u.x"``) or the value of a function f(sim); attrs ``dims``
                       (JSON, e.g. ["t", "component", "x", "z"]), ``component_names`` for
                       vectors; each labelled dimension is attached to its scale (HDF5 H5DS)

Self-describing: h5py, xarray (``cn.io.read(path).to_xarray()``) and
ParaView (the XDMF sidecar ``<file>.xmf`` of Cartesian 2D/3D grids, ``conatus.io.xdmf``) read
the arrays with their coordinates; files written before keep reading.

The file is opened and closed at each write: a job killed between two writes leaves a valid
file with every completed row (and only those: the row counter is written last).

One file holds **one computation** (0.1.1, review finding 5): ``mode="a"`` and ``mode="resume"``
compare the computation appending with the one the file records (model, equations, parameters,
given data, resolution, geometry, scheme, dt, cadence, and every task's source) and refuse any
difference, naming it; before, a run at nu = 0.4 writing dx(u) was appended under the metadata of
a run at nu = 0.05 writing u.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import InputTypeError, OptionError
from . import _describe as dsc
from ._h5 import DOC, json_attr, open_h5, positive, provenance, read_json_attr
from ._identity import checkpoint_identity

FORMAT = "conatus.analysis/1"
MODES = ("w", "a", "resume")


def _is_name(src: str) -> bool:
    """An unknown or a component ("T", "u.x"), as opposed to an expression ("dz(T)")."""
    import re
    return re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)?", src) is not None


def _host(a: Any) -> np.ndarray:
    if hasattr(a, "get") and not isinstance(a, np.ndarray):     # CuPy array
        a = a.get()
    return np.asarray(a)


class AnalysisFile:
    """Write ``tasks`` of a simulation to the HDF5 file ``path`` every ``every`` steps.

    ``source``: a native Simulation (``sim.output(...)``) or an expert IVPSolver (then pass the
    object as ``solver.run(n, callback=out, every=out.every)``). ``tasks``: a list of unknown
    names / components (``["T", "u.x"]``) or ``{name: source}`` with a source = such a name or a
    function ``f(sim)`` returning a number or an array. ``mode``: required when the file exists
    (``"w"`` replaces it, ``"a"`` appends to it with the same tasks, ``"resume"`` appends after
    dropping the rows written after the current iteration: the next job of a run restarted from
    a checkpoint older than the last rows, which the exact restart writes again); a file is never
    replaced or cut silently.

    ``initial``: also write the state at the start of the first run, when its iteration is a
    multiple of ``every`` and the file has no row there yet (a time series then starts from the initial condition, as in Dedalus). ``sim.output`` passes
    True by default; this class defaults to False because an expert solver's ``run`` only calls
    it after steps (call :meth:`write` once before the run for the initial row)."""

    def __init__(self, source: Any, path: str | Path, tasks: Sequence[str] | Mapping[str, Any], *,
                 every: int, mode: str | None = None, initial: bool = False) -> None:
        self.source, self.path = source, Path(path)
        self.every = positive(every, "output()", "every")
        if not isinstance(initial, bool):
            raise OptionError(f"output(initial={initial!r})",
                              why="initial is on (True: the state at the start of the first run "
                                  "is written) or off (False)", fix="initial=False", doc=DOC)
        self.initial = initial
        self.initial_pending = initial
        self.initial_note = ""
        self.solver = getattr(source, "solver", source)
        self.tasks = self._parse(tasks)
        if mode is not None and mode not in MODES:
            raise OptionError(f"output(mode={mode!r})", fix="mode='w' (replace), 'a' (append) or "
                              "'resume' (append after a restart, dropping later rows)", doc=DOC)
        if self.path.exists() and mode is None:
            raise OptionError(f"output(): {self.path} exists",
                              why="replacing or appending to an existing file needs mode=",
                              fix="mode='w' to replace it, mode='a' to append to it "
                                  "(mode='resume' after a restart from an older checkpoint)", doc=DOC)
        self.mode = mode or "w"
        self.writes = 0
        self.dropped = 0                                # rows removed by mode="resume" (declared)
        if self.mode in ("a", "resume") and self.path.exists():
            self._check_append()
        else:
            self._create()

    # ------------------------------------------------------------------ tasks
    def _parse(self, tasks: Sequence[str] | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(tasks, str) or not isinstance(tasks, (Mapping, Sequence)):
            raise InputTypeError("output(tasks=...): a list of names or a {name: source} dict",
                                 fix="tasks=['T', 'u.x'] or tasks={'Nu': lambda sim: ...}", doc=DOC)
        items = tasks.items() if isinstance(tasks, Mapping) else ((t, t) for t in tasks)
        out = {}
        unknowns = {v.name: v.rank for v in self.solver.system.variables}
        for name, src in items:
            if not isinstance(name, str) or not name or "/" in name:
                # a dict inside the list (tasks=["T", {"KE": f}]) said only "a non-empty name
                # without '/'"
                fix = ("a list holds names only; functions go in a dict: tasks={'T': 'T', "
                       "'KE': f}" if isinstance(name, Mapping) or callable(name)
                       else "a non-empty name without '/'")
                raise InputTypeError(f"output(): task name {name!r}", fix=fix, doc=DOC)
            if isinstance(src, str) and not _is_name(src):
                # an expression of the state ("dz(T)", "0.5*u@u"): native simulations only,
                # parsed and evaluated once now so that a typo fails here, not at the first write
                if not hasattr(self.source, "evaluate"):
                    raise InputTypeError(f"output(): task {name!r} = {src!r} is an expression",
                                         why="expressions are evaluated by a native Simulation",
                                         fix="a function f(solver), or sim.output(...) on the "
                                             "native simulation", doc=DOC)
                self.source.evaluate(src)
            elif isinstance(src, str):
                base, _, comp = src.partition(".")
                if base not in unknowns:
                    raise InputTypeError(f"output(): {base!r} is not an unknown",
                                         fix="unknowns: " + ", ".join(unknowns)
                                             + "; or give a function f(sim)", doc=DOC)
                if comp and unknowns[base] != 1:
                    raise InputTypeError(f"output(): {src!r}: components are for vectors", doc=DOC)
                if comp and comp not in self._coord_names():
                    raise InputTypeError(f"output(): {src!r}: no component {comp!r}",
                                         fix="components: " + ", ".join(self._coord_names()), doc=DOC)
            elif not callable(src):
                raise InputTypeError(f"output(): task {name!r} must be a name or a function f(sim)",
                                     doc=DOC)
            out[name] = src
        if not out:
            raise InputTypeError("output(): no task", doc=DOC)
        return out

    def _coord_names(self) -> list[str]:
        comp = getattr(self.source, "compiled", None)
        if comp is not None:
            return list(comp.model.geometry.coord_names)
        # expert solver: the components of a vector are those of the coordinate system (
        # plan.problem.domain does not exist, "u.x" raised AttributeError)
        return [c.name for c in self.solver.plan.domain.coordsys.coords]

    def _value(self, src: Any) -> np.ndarray:
        if callable(src):
            return _host(src(self.source))
        if not _is_name(src):
            return _host(self.source.evaluate(src))
        base, _, comp = src.partition(".")
        g = _host(self.source.grid(base))
        return g[self._coord_names().index(comp)] if comp else g

    @staticmethod
    def _describe(src: Any) -> str:
        if isinstance(src, str):
            return f"grid values of {src}" if _is_name(src) else f"grid values of the expression {src}"
        return f"function {getattr(src, '__qualname__', type(src).__name__)}"

    # ------------------------------------------------------------------ file
    def _scales(self) -> dict[str, np.ndarray]:
        if not hasattr(self.source, "coordinate"):      # expert solver: no grid read here
            return {}
        return {c: _host(self.source.coordinate(c)) for c in self._coord_names()}

    def _create(self) -> None:
        with open_h5(self.path, "w") as f:
            f.attrs["format"] = FORMAT
            f.attrs["problem"] = str(self.solver.plan.problem.name)
            f.attrs["scheme"] = str(self.solver.scheme)
            f.attrs["dt"] = float(self.solver.dt)
            f.attrs["every"] = int(self.every)
            f.attrs["tasks"] = json_attr({k: self._describe(v) for k, v in self.tasks.items()})
            f.attrs["provenance"] = json_attr(provenance(self.solver))
            f.attrs["identity"] = json_attr(checkpoint_identity(self.solver))
            f.attrs["task_identity"] = json_attr(self._task_identity())
            f.attrs["writes"] = 0
            sc = f.create_group("scales")
            for k, dtype in (("sim_time", float), ("iteration", np.int64), ("write_number", np.int64)):
                sc.create_dataset(k, shape=(0,), maxshape=(None,), dtype=dtype)
            dsc.root_attrs(f, self.solver, self.source)
            dsc.make_scale(sc["sim_time"], "t")
            for c, x in self._scales().items():
                dsc.make_scale(sc.create_dataset(c, data=x), c)
            if self._scales():
                comp = sc.create_dataset(dsc.COMPONENT, data=np.array(self._coord_names(), dtype="S"))
                dsc.make_scale(comp, dsc.COMPONENT)
            f.create_group("tasks")

    def _check_append(self) -> None:
        with open_h5(self.path, "r") as f:
            fmt = f.attrs.get("format")
            fmt = fmt.decode() if isinstance(fmt, bytes) else fmt
            if fmt != FORMAT:
                raise OptionError(f"output(mode='a'): {self.path} is not a Conatus analysis file "
                                  f"(format {fmt!r})", doc=DOC)
            old = read_json_attr(f, "tasks") or {}
            new = {k: self._describe(v) for k, v in self.tasks.items()}
            if set(old) != set(new):
                raise OptionError(f"output(mode='a'): tasks {sorted(new)} differ from the file's "
                                  f"{sorted(old)}", fix="the same task names, or mode='w'", doc=DOC)
            if float(f.attrs["dt"]) != float(self.solver.dt) or int(f.attrs["every"]) != self.every:
                raise OptionError("output(mode='a'): dt or every differs from the file's",
                                  why="one file holds one cadence", fix="a new file", doc=DOC)
            self._check_same_computation(f, old, new)
            self.writes = int(f.attrs["writes"])
            its = f["scales/iteration"][:self.writes]
            last = f["scales/sim_time"][self.writes - 1] if self.writes else None
        keep = int(np.count_nonzero(its <= self.solver.iteration))
        if self.mode == "a" and last is not None and self.solver.t < last:
            raise OptionError(f"output(mode='a'): the simulation time {self.solver.t:g} is before the "
                              f"last write ({last:g})", why="rows stay ordered in time",
                              fix="mode='resume' after a restart from an older checkpoint (drops the "
                                  f"{self.writes - keep} later row(s), written again by the exact "
                                  "restart), or a new file", doc=DOC)
        if self.mode == "resume" and keep < self.writes:
            if np.any(np.diff(its) <= 0):
                raise OptionError(f"output(mode='resume'): {self.path} rows are not in increasing "
                                  "iteration order", fix="a new file", doc=DOC)
            with open_h5(self.path, "a") as f:          # counter first: a crash keeps a valid file
                f.attrs["writes"] = keep
                for d in list(f["tasks"].values()) + [f["scales"][k] for k in
                                                       ("sim_time", "iteration", "write_number")]:
                    d.resize(keep, axis=0)
            self.dropped, self.writes = self.writes - keep, keep
            self._resumed_at = int(self.solver.iteration)

    def _task_identity(self) -> dict:
        """name -> identity of each task's source: the text of a name or an expression, the
        code (and what it reads, :func:`conatus.io._identity.callable_identity`) of a function.
        A function reading a value without a content hash is identified by its code only
        (``"undeclared"`` says why; declared by explain() after an append)."""
        from ._identity import Undeclared, _code_digest, callable_identity
        out = {}
        for k, src in self.tasks.items():
            if isinstance(src, str):
                out[k] = {"source": src}
                continue
            try:
                out[k] = callable_identity(src, f"task {k}")
            except Undeclared as why:
                code = getattr(src, "__code__", None)
                out[k] = {"function": getattr(src, "__qualname__", type(src).__name__),
                          "code_sha256": None if code is None else _code_digest(code),
                          "undeclared": str(why)}
        return out

    def _check_same_computation(self, f: Any, old_tasks: dict, new_tasks: dict) -> None:
        """Refuse an append by another computation (review finding 5), naming every difference;
        a file written before the identity record is compared on what it holds (problem name,
        parameters, scheme, task sources) with an :class:`IdentityWarning`."""
        from ._identity import IdentityWarning, _differences, unrecorded
        here = f"in {self.path.name}"
        diffs = []
        mine = checkpoint_identity(self.solver)
        old = read_json_attr(f, "identity")
        if old is None:
            old = {"model": _attr_text(f, "problem")}
            params = read_json_attr(f, "parameters")
            if params is not None:
                old["parameters"] = params
        diffs += [t for _k, _c, t in _differences(old, mine, there=here)]
        if _attr_text(f, "scheme") != str(self.solver.scheme):
            diffs.append(f"scheme: {_attr_text(f, 'scheme')!r} {here}, {self.solver.scheme!r} now")
        for k in sorted(new_tasks):
            if old_tasks.get(k) != new_tasks[k]:
                diffs.append(f"task {k}: {old_tasks.get(k)} {here}, {new_tasks[k]} now")
        old_ti = read_json_attr(f, "task_identity")
        self.task_notes = []
        if old_ti is not None:
            for k, ti in self._task_identity().items():
                a = old_ti.get(k) or {}
                if "undeclared" in a or "undeclared" in ti:
                    same = (a.get("code_sha256"), a.get("function")) == \
                        (ti.get("code_sha256"), ti.get("function"))
                    self.task_notes.append(
                        f"task {k}: compared by its code only ("
                        f"{ti.get('undeclared') or a.get('undeclared')})")
                else:
                    same = a == ti
                if not same and old_tasks.get(k) == new_tasks.get(k):
                    diffs.append(f"task {k}: the function {new_tasks[k][9:]} has other code or "
                                 "reads other values than the one that wrote the file")
        if diffs:
            raise OptionError(
                f"output(mode={self.mode!r}): {self.path} records another computation: "
                + "; ".join(diffs),
                why="one file holds one computation: rows appended by another model, other "
                    "equations, parameters, data or tasks would be read under the file's "
                    "metadata",
                fix="write this run to a new file (or mode='w' to replace this one)", doc=DOC)
        missing = unrecorded(old, mine) + [k for k in ("equations", "resolution", "geometry")
                                           if k not in old]
        if missing:
            import warnings
            warnings.warn(
                f"output(mode={self.mode!r}): {self.path} was written before analysis files "
                "recorded the identity of their computation: " + ", ".join(sorted(set(missing)))
                + " could not be compared (the model name, parameters, scheme, dt and tasks "
                "were).", IdentityWarning, stacklevel=4)

    def write(self) -> None:
        """Write one row now (the current state), whatever the iteration."""
        if getattr(self.solver, "_finite_every", 0) and hasattr(self.solver, "X"):
            # a blown-up state is refused before it reaches the file; the
            # solver's own check (collective in the distributed solver)
            from ..timestep.integrator import solver_finite_check
            solver_finite_check(self.solver, f"output({str(self.path)!r})")
        values = {k: self._value(v) for k, v in self.tasks.items()}
        with open_h5(self.path, "a") as f:
            w = int(f.attrs["writes"])
            tg = f["tasks"]
            for k, a in values.items():
                if k not in tg:
                    tg.create_dataset(k, shape=(0, *a.shape), maxshape=(None, *a.shape),
                                      dtype=a.dtype, chunks=(1, *a.shape) if a.shape else None)
                    tg[k].attrs["source"] = self._describe(self.tasks[k])
                    self._label(f, tg[k], k, a.shape)
                d = tg[k]
                if d.shape[1:] != a.shape:
                    raise OptionError(f"output(): task {k!r} changed shape {d.shape[1:]} -> {a.shape}",
                                      doc=DOC)
                d.resize(w + 1, axis=0)
                d[w] = a
            sc = f["scales"]
            for k, v in (("sim_time", self.solver.t), ("iteration", self.solver.iteration),
                         ("write_number", w + 1)):
                sc[k].resize(w + 1, axis=0)
                sc[k][w] = v
            f.attrs["writes"] = w + 1                   # last: a partial row is never counted
        self.writes = w + 1

    def _rank(self, src: Any) -> int | None:
        """Tensor rank of a task's grid values: an unknown (0 for a component), an expression
        string (its rank, evaluated once at declaration), None for a function (never guessed)."""
        if callable(src):
            return None
        if _is_name(src):
            base, _, comp = src.partition(".")
            ranks = {v.name: v.rank for v in self.solver.system.variables}
            return 0 if comp else ranks.get(base, 0)
        return None if not hasattr(self.source, "evaluate") else -1     # read off the shape

    def _label(self, f: Any, dset: Any, name: str, shape: tuple) -> None:
        """``dims``, dimension labels and scales of a new task dataset."""
        sc = f["scales"]
        coords = [c for c in self._coord_names() if c in sc]
        grid = tuple(sc[c].shape[0] for c in coords)
        rank = self._rank(self.tasks[name])
        if rank == -1:                                   # expression: its rank from its shape
            extra = len(shape) - len(grid)
            rank = extra if 0 <= extra <= 2 else None
        dims = dsc.dims_for(shape, coords, grid, rank=rank, name=name)
        scales = {"t": sc["sim_time"], **{c: sc[c] for c in coords}}
        if dsc.COMPONENT in sc:
            scales[dsc.COMPONENT] = sc[dsc.COMPONENT]
        has_comp = any(d.startswith(dsc.COMPONENT) for d in dims)
        dsc.label(dset, dims, scales, rank=rank if has_comp else None, coords=coords)

    def write_initial(self) -> None:
        """The row of the start (``initial=True``), at the first run after the declaration: only
        on a multiple of ``every`` (the cadence of the file) and when the file holds no row at
        this iteration (a resumed chain already wrote it). What happened is in explain()."""
        self.initial_pending = False
        it = int(self.solver.iteration)
        if it % self.every:
            self.initial_note = (f"no row at the start: iteration {it} is not a multiple of every "
                                 f"= {self.every}")
            return
        if self.writes:
            with open_h5(self.path, "r") as f:
                last = int(f["scales/iteration"][self.writes - 1])
            if last >= it:
                self.initial_note = (f"the row of the start (iteration {it}) was already in the "
                                     "file (written by an earlier job of the chain)")
                return
        self.write()
        self.initial_note = (f"the state at the start (iteration {it}, t = {self.solver.t:g}) "
                             "written as a row (initial=True, the default of sim.output)")

    def __call__(self, solver: Any) -> None:            # callback of IVPSolver.run
        if solver.iteration % self.every == 0:
            self.write()

    def write_xdmf(self):
        """Rewrite the XDMF sidecar ``<file>.xmf`` (Cartesian 2D/3D grids; see
        :mod:`conatus.io.xdmf`), called at the end of each ``sim.run``. Returns the sidecar path,
        or None with the reason in ``self.xdmf_note`` (declared by :meth:`explain`)."""
        from .xdmf import XDMFNotCovered, write_xdmf
        try:
            out = write_xdmf(self.path)
            self.xdmf_note = f"XDMF sidecar (ParaView): {out}"
            return out
        except XDMFNotCovered as why:
            self.xdmf_note = f"no XDMF sidecar: {why.message}"
            return None

    def explain(self) -> str:
        """File, cadence, tasks and the rows written (or dropped by mode='resume')."""
        lines = [f"AnalysisFile {self.path} (HDF5, {FORMAT}): every {self.every} steps "
                 f"(= {self.every * self.solver.dt:g} time units), {self.writes} write(s) so far"
                 + (f"; mode='resume' dropped {self.dropped} row(s) written after iteration "
                    f"{self._resumed_at} by an earlier job" if self.dropped else "")]
        lines += [f"  {k}: {self._describe(v)}" for k, v in self.tasks.items()]
        if self.initial_note:
            lines.append("  " + self.initial_note)
        elif self.initial_pending:
            lines.append("  initial=True: the state at the start of the next run is written "
                         "(if its iteration is a multiple of every and not yet in the file)")
        else:                                  # initial=False (True always leaves a note)
            lines.append(f"  first row at iteration {self.every} (an expert solver's run calls "
                         "the file after steps; write() adds a row now)")
        grids = ", " + ", ".join(self._coord_names()) if hasattr(self.source, "coordinate") \
            else " (no coordinate grids: expert solver)"
        lines.append("  scales: sim_time, iteration, write_number" + grids
                     + "; every array labelled with its dimensions (HDF5 dimension scales)")
        if getattr(self, "xdmf_note", None):
            lines.append("  " + self.xdmf_note)
        lines += ["  " + n for n in getattr(self, "task_notes", [])]
        return "\n".join(lines)

    __str__ = explain


def _combine(pairs: Sequence[tuple[Callable[[Any], None], int]]) -> tuple[Callable | None, int | None]:
    """One solver callback for several (function, every) pairs: called on the gcd of the
    cadences, each function only on its own multiples."""
    import math
    pairs = [(f, int(e)) for f, e in pairs if f is not None]
    if not pairs:
        return None, None
    if len(pairs) == 1:
        return pairs[0]
    g = 0
    for _f, e in pairs:
        g = math.gcd(g, e)

    def all_(solver: Any) -> None:
        for f, e in pairs:
            if solver.iteration % e == 0:
                f(solver)
    return all_, g


def _attr_text(f: Any, key: str) -> str | None:
    """A text attribute of an HDF5 file (bytes decoded), None if absent."""
    v = f.attrs.get(key)
    if v is None:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)
