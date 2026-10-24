"""Solving a native model: time integration (:class:`Simulation`) and normal modes
(:class:`Linearization`, defined in :mod:`conatus.model.linearization` and re-exported here).
Thin, declared wrappers over IVPSolver, linearize and eigenvalues: the expert objects stay
reachable (``sim.solver``, ``sim.plan``, ``lin.discretize(N)``)."""
from __future__ import annotations

import math
import numbers
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..errors import InputTypeError, OptionError, ResolutionError
from ..ir import Parameter
from ..tolerances import REQUIRED
from .compiler import Compiled, _coord_args, _on_all_grids
from .linearization import Linearization, _without_taus, sincos_eigenvalues  # noqa: F401 (re-export)
from .._deprecation import renames

__all__ = ["Simulation", "Linearization"]

#: Arrays returned by ``Simulation.grid`` (id -> weak reference): on a grid with two axes of the
#: same size, ``initial(T=array)`` takes such an array as laid out.
_GRID_ARRAYS: dict[int, Any] = {}


def _remember_grid(g: Any) -> Any:
    """Record ``g`` as an array of ``sim.grid`` layout (weakly; forgotten when it is freed)."""
    import weakref
    try:
        k = id(g)
        _GRID_ARRAYS[k] = weakref.ref(g, lambda _r, k=k: _GRID_ARRAYS.pop(k, None))
    except TypeError:                  # an array type without weak references: not recorded
        pass
    return g


def _from_grid(v: Any) -> bool:
    r = _GRID_ARRAYS.get(id(v))
    return r is not None and r() is v


_DOC = "docs/site/guide/timesteppers.md"   # public guide page (never an internal design note)


def _native_sentence(comp: Compiled) -> str:
    """One Methods sentence on what the native layer decided."""
    nt = len(comp.taus)
    parts = [f"The equations were declared in the native layer of Conatus as written on paper; "
             f"the {'implicit-explicit split' if comp.kind == 'IVP' else 'operator form'}"]
    if nt:
        parts.append(f"the {nt} tau term(s) (each in the equation where its unknown has its "
                     "highest wall-normal order)")
    if comp.gauge:
        parts.append("the pressure gauge and its integral constraint")
    s = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] if len(parts) > 1 \
        else parts[0]
    s += " were placed by fixed, declared rules (listed by explain())."
    if comp.symmetry is not None:
        w = comp.model.geometry.sincos_coord
        s += (f" Along {w} the unknowns are sine or cosine series (parities read off the wall "
              f"conditions), computed as the reflection-symmetric subspace of a real Fourier "
              f"series on the doubled interval; no tau term is needed there.")
    if comp.dealias_default:
        from .compiler import _frac, _weighted_axes, dealias_factor
        weighted = _weighted_axes(comp.bases)
        if len(weighted) < len(comp.bases):
            f = _frac(dealias_factor(comp.degree))
            s += (f" The dealiasing factor {f} is the declared default of the native layer "
                  f"((p+1)/2 for the highest polynomial degree p = {comp.degree} of the "
                  "equations in the unknowns and given fields, at least 3/2).")
        if weighted:
            fw = _frac(dealias_factor(comp.degree, True, comp.unknown_degree))
            s += (f" Along {', '.join(weighted)} (Hermite/Laguerre functions) the default "
                  f"dealiasing factor is {fw} (at least 2 for products of the unknowns, 3/2 for "
                  "a linear model); no factor makes products alias-free there: it reduces "
                  "aliasing only.")
    if comp.overrides:
        s += (" Parameter values set for this run: "
              + ", ".join(f"{k} = {v:g}" for k, v in comp.overrides.items()) + ".")
    return s


def check_solver_options(kw: Mapping[str, Any], where: str) -> None:
    """Refuse a keyword of ``m.simulate(...)`` that is neither a declared parameter (popped
    before) nor an option of the solver, with the accepted names (an unknown keyword
    would otherwise surface as "IVPSolver.__init__() got an unexpected keyword argument")."""
    import inspect

    from ..errors import did_you_mean
    from ..timestep import IVPSolver
    from ..timestep.polar import PolarIVPSolver
    from ..timestep.sphere import SphereIVPSolver
    accepted = {"ncc_cutoff", "ncc_tail_tol"}
    for cls in (IVPSolver, PolarIVPSolver, SphereIVPSolver):   # Cartesian, polar, spherical
        sig = inspect.signature(cls.__init__).parameters
        accepted |= {n for n, p in sig.items() if p.kind is p.KEYWORD_ONLY}
    accepted = sorted(accepted - {"backend"})              # backend: an argument of simulate
    bad = [k for k in kw if k not in accepted]
    if bad:
        hint = {"timestepper": "scheme", "ts": "scheme", "timestep": "dt", "step": "dt"}
        k = bad[0]
        dym = f" Did you mean {hint[k]!r}?" if k in hint else did_you_mean(k, accepted)
        raise OptionError(f"{where}(..., {k}=...): {k!r} is neither a declared parameter of the "
                          "model nor an option of the solver." + dym,
                          why="parameters are declared in cn.Model(geometry, ..., name=value); "
                              "the other keywords go to the time stepper",
                          fix="solver options: " + ", ".join(accepted), doc=_DOC)


class Simulation:
    """Time integration of a compiled IVP (``Model.simulate``): scheme and dt are explicit."""

    def __init__(self, comp: Compiled, *, scheme: str, dt: float, backend: str = "cpu",
                 **solver_kw: Any) -> None:
        from ..timestep import IVPSolver
        self.compiled = comp
        self.problem = comp.problem
        from .model import ncc_options
        from . import plan_cache
        # a fresh plan, seeded with the operators of an identical earlier build of this model
        # (plan_cache: same equations, data, parameters, N, dealiasing; said by explain())
        self.plan, self.operators_note, cache_key = plan_cache.plan_for(
            comp, ncc_options(solver_kw))                         # ncc_cutoff= / ncc_tail_tol=
        if comp.symmetry is not None:
            # sine/cosine axis: the reflection subspace IS the discretization (the full doubled
            # period would hold states that violate the wall conditions): reduced, explicitly
            if solver_kw.get("reduced", True) is not True:
                raise OptionError(
                    "reduced=False on a sine/cosine axis",
                    why="the sine/cosine basis is the reflection subspace of the doubled Fourier "
                        "series; the full space contains states violating the wall conditions",
                    fix="drop reduced= (or use walls(a, b) with Chebyshev)", doc=_DOC)
            solver_kw["reduced"] = True
        if comp.time is not None:                 # the native time t is the stage time
            tp = solver_kw.setdefault("time_parameter", comp.time.name)
            if tp != comp.time.name:
                raise OptionError(f"time_parameter={tp!r}: the time of a native model is "
                                  f"{comp.time.name!r}", fix="drop time_parameter=", doc=_DOC)
        self.solver = IVPSolver(self.plan, scheme, dt, backend=backend, **solver_kw)
        plan_cache.remember(comp, self.plan, cache_key)
        self.watching = None                       # a conatus.model.watch.Watch, or None
        self.outputs: list = []                    # AnalysisFile / CheckpointSeries (cn.io)
        self.stopped: str | None = None            # reason of a Studio stop (run(on_stop="return"))
        reg = getattr(comp.model, "_register", None)
        if reg is not None:                        # sim.id; named by later parameter changes
            reg(self)

    def __repr__(self) -> str:
        """One line: problem, resolution, scheme, step, backend, where the run is."""
        comp, sol = self.__dict__.get("compiled"), self.__dict__.get("solver")
        if comp is None or sol is None:                # construction stopped half-way
            return "Simulation(<not constructed>)"
        n = ", ".join(f"{c}={k}" for c, k in comp.N.items())
        return (f"Simulation({comp.problem.name!r}, N=({n}), scheme={sol.scheme!r}, dt={sol.dt:g}, "
                f"backend={sol.backend!r}, t={sol.t:g}, iteration={sol.iteration}; .explain())")

    @property
    def t(self) -> float:
        """Current simulation time."""
        return self.solver.t

    @property
    def dt(self) -> float:
        """Time step (fixed at ``simulate``; the run never changes it on its own: only an explicit
        Studio ``set`` on a step declared settable by ``watch(settable=...)``, recorded)."""
        return self.solver.dt

    @property
    def iteration(self) -> int:
        """Number of steps taken."""
        return self.solver.iteration

    def _initial_time(self, t: Any, values: dict, where: str) -> float | None:
        """The start time given as ``t=``, or None. A model with an unknown named ``t``
        has no time t (``t`` is then that unknown): its ``t=`` is the unknown's initial data, as
        before, and the start time cannot be set by name (declared here)."""
        if t is None:
            return None
        if "t" in {u.name for u in self.compiled.model.unknowns}:
            values["t"] = t                        # the unknown named t (v0.1 behaviour)
            return None
        if isinstance(t, bool) or not isinstance(t, numbers.Real) or not math.isfinite(float(t)):
            raise OptionError(f"{where}(t={t!r}): the initial time is a finite real number",
                              why="t is the time at which the initial state is given: forcing, "
                                  "boundary data and every output, checkpoint and run(until=) "
                                  "are read on this clock", fix=f"{where}(..., t=0.0)", doc=_DOC)
        return float(t)

    def initial(self, *, layout: str | None = None, check: bool = True, t: Any = None,
                **values: Any) -> Simulation:
        """Initial state by unknown name: callables of coordinate names (``T=lambda x, z: -z``,
        preferred: unambiguous), numbers, or arrays of grid values with exactly the shape of
        ``sim.grid(name)`` (e.g. the state of another simulation at the same N;
        the layout is the grid of :meth:`grid`, components first, then the coordinates in
        order, i.e. ``np.meshgrid(x, z, indexing="ij")``); unknowns not given start at zero
        (declared by explain()).

        On a grid with two axes of the same size the shape cannot tell ``indexing="ij"`` from
        NumPy's default ``"xy"`` (transposed): such an array is refused unless it is one
        returned by ``sim.grid(...)`` or ``layout="ij"`` declares it.

        Callable data are checked (``conatus.model.initial_check``):
        non-finite values are refused, and so are clearly wrong data (relative spectral tail
        beyond the resolution above ``INITIAL_TAIL_REFUSE`` on some axis, or a jump at the ends of
        a periodic axis: an aliased mode, a function not periodic on the box), naming the axis;
        data that are only not fully resolved (tail between ``INITIAL_TAIL_TOL`` and
        ``INITIAL_TAIL_REFUSE``, e.g. the coarse N of a resolution study) give one
        ``InitialDataWarning``, recorded by :meth:`explain`. ``check=False`` silences both
        (declared by :meth:`explain`).

        ``t``: the time at which this state is given (default 0). Time-dependent forcing
        and boundary data are evaluated from it, every output row and checkpoint carries times
        t + k dt, ``run(until=...)`` is read on that clock, checkpoints record it (a chain
        continued by :meth:`resume_or_start` with another ``t`` is refused) and :meth:`explain`
        states it. In a model with an unknown named ``t`` (which then has no time), ``t=`` is
        that unknown's data."""
        t0 = self._initial_time(t, values, "initial")
        if not isinstance(check, bool):
            raise OptionError(f"initial(check={check!r})",
                              why="check is on (True, the default) or off (False)",
                              fix="check=False to accept callable data the resolution cannot "
                                  "hold (declared by explain())", doc=_DOC)
        if layout is not None and layout != "ij":
            raise OptionError(f"initial(layout={layout!r})",
                              why="arrays are read in the layout of sim.grid(): components "
                                  "first, then the coordinates in the order of the geometry",
                              fix="layout='ij' (np.meshgrid(..., indexing='ij')); a transposed "
                                  "array is transposed by you (np.swapaxes), never here",
                              doc=_DOC)
        names = self.compiled.model.geometry.coord_names
        unknowns = {u.name for u in self.compiled.model.unknowns}
        data = {}
        notes = [] if check else ["Initial data check: off (initial(..., check=False)): callable "
                                  "data were not checked for aliasing, periodicity or finite "
                                  "values"]
        warned: list[str] = []
        for k, v in values.items():
            if k not in unknowns:
                raise InputTypeError(f"initial({k}=...): {k} is not an unknown",
                                     fix="unknowns: " + ", ".join(sorted(unknowns)), doc=_DOC)
            f = self.compiled.fields[k]
            if callable(v):
                data[f] = _on_all_grids(v, _coord_args(v, names, f"initial {k}"), names)
                if check:
                    from .initial_check import check_callable
                    more, warn = check_callable(self, k, v, f, data[f])
                    notes += more
                    warned += warn
            elif isinstance(v, numbers.Number):
                # a non-finite number is refused like a non-finite array or callable value
                # (review finding 30: initial(u=np.nan) started a NaN run)
                if not np.isfinite(complex(v).real) or not np.isfinite(complex(v).imag):
                    raise InputTypeError(f"initial({k}={v!r}): not a finite number",
                                         why="the run would start from a NaN/inf state, stopped "
                                             "only steps later by the finite-state check",
                                         fix=f"a finite value for {k}", doc=_DOC)
                data[f] = (lambda c: (lambda *g: np.asarray(c)))(v)
            elif isinstance(v, np.ndarray) or (hasattr(v, "shape") and hasattr(v, "dtype")
                                                and not hasattr(v, "layout")):
                data[f] = self._grid_array(k, f, v, layout)
                continue
            else:
                from ..reference.evaluate import FieldData
                if not isinstance(v, FieldData):
                    # text, a list, None... (audit inputs: a bare TypeError of the
                    # solver; a list is not read as an array without its shape being checked)
                    raise InputTypeError(
                        f"initial({k}={v!r}): not initial data" if not isinstance(v, (list, tuple))
                        else f"initial({k}=<{type(v).__name__}>): not initial data",
                        why="initial data are a callable of the coordinates, a number, or an "
                            "array of grid values of the shape of sim.grid(name)",
                        fix=f"initial({k}=lambda {', '.join(names)}: ...), or "
                            f"initial({k}=np.asarray(values)) with the shape of sim.grid({k!r})",
                        doc=_DOC)
                data[f] = v                        # FieldData (explicit layout)
                continue
            sym = self.compiled.symmetry
            if sym is not None:
                # sine/cosine axis: evaluated on [a, b] only, extended by the declared parity
                # to the doubled period (odd components must vanish at the walls, else refused)
                from .sincos import extend_to_period, reflection
                R = reflection(self.compiled.model.geometry, self.compiled.coordsys)
                data[f] = extend_to_period(self.compiled, data[f], f.rank,
                                           sym.parity(f, R), f"initial {k}")
        if warned:                    # not fully resolved: warned once, recorded
            import warnings

            from ..errors import InitialDataWarning
            from .initial_check import INITIAL_TAIL_TOL
            text = ("initial data not fully resolved: " + "; ".join(warned) + ". The run starts "
                    "from their projection on the resolved modes; raise N if this is not a "
                    "deliberate coarse resolution (initial(..., check=False) silences it)")
            warnings.warn(text, InitialDataWarning, stacklevel=2)
            notes.append(f"Initial data warning (tail above {INITIAL_TAIL_TOL:g}): " + text)
        self._given = sorted(values)
        self._initial_notes = notes
        self._restarted = None
        self._chain = None
        self._t_given = t0
        self.solver.set_state(data, t=0.0 if t0 is None else t0)
        return self

    def _grid_array(self, name: str, f: Any, v: Any, layout: str | None = None):
        """An array of grid values for ``initial(name=array)``: the shape of ``sim.grid(name)``
        exactly (never broadcast or interpolated), on a Cartesian layout."""
        from ..reference.evaluate import FieldData
        if self.compiled.coordsys.curvilinear or self.compiled.symmetry is not None:
            raise InputTypeError(
                f"initial({name}=<array>) on {self.compiled.model.geometry!r}",
                why="arrays of grid values are accepted on Cartesian walls/periodic axes; on a "
                    "curved geometry or a sine/cosine axis the solver grid is not the grid of "
                    "sim.grid()", fix=f"initial({name}=lambda ...: ...) (a callable of the "
                    "coordinates)", doc=_DOC)
        arr = np.asarray(v.get() if hasattr(v, "get") and not isinstance(v, np.ndarray) else v)
        want = self.solver.grid(name).shape
        if arr.shape != want:
            raise InputTypeError(
                f"initial({name}=<array of shape {arr.shape}>): the grid of {name} has shape "
                f"{want}", why="an array is read as grid values of sim.grid(name) at this "
                "resolution (components first, then the coordinates); it is never "
                "broadcast or interpolated", fix=f"an array of shape {want} "
                "(sim.grid(name) of a simulation at the same N), or a callable of the "
                "coordinates", doc=_DOC)
        if not np.all(np.isfinite(arr)):
            raise InputTypeError(f"initial({name}=<array>): non-finite values", doc=_DOC)
        spatial = want[f.rank:]
        if (len(spatial) >= 2 and len(set(spatial)) < len(spatial) and layout is None
                and not _from_grid(v)):
            coords = self.compiled.model.geometry.coord_names
            raise InputTypeError(
                f"initial({name}=<array of shape {arr.shape}>): two axes of the grid have the "
                f"same size ({', '.join(f'{c}: {n}' for c, n in zip(coords, spatial, strict=True))}), so the "
                "shape does not say which axis is which",
                why="np.meshgrid's default indexing='xy' gives the transposed layout with the "
                    "same shape on such a grid; reading it as the grid of sim.grid() would "
                    "transpose the initial state without a word",
                fix=f"initial({name}=lambda {', '.join(coords)}: ...) (a callable of the "
                    f"coordinates, preferred), or declare the layout: initial({name}=array, "
                    f"layout='ij') for an array indexed [{', '.join(coords)}] "
                    "(np.meshgrid(..., indexing='ij'))", doc=_DOC)
        return FieldData(f, arr, "grid")

    def _check_unknown(self, name: str, what: str) -> None:
        """A typed refusal with a did-you-mean for a name that is not an unknown (earlier versions: a bare ValueError "'w' is not in list")."""
        names = [u.name for u in self.compiled.model.unknowns]
        if name not in names:
            from ..errors import did_you_mean
            raise InputTypeError(f"{what}({name!r}): not an unknown of the model."
                                 + did_you_mean(str(name), names),
                                 fix="unknowns: " + ", ".join(names) + "".join(
                                     f"; a component of the vector {u.name}: "
                                     f"sim.{what}({u.name!r})[i]"
                                     for u in self.compiled.model.unknowns
                                     if self.compiled.fields[u.name].rank == 1) + (
                                     "; an expression of them: sim.evaluate(\"...\")"
                                     if what == "grid" else ""), doc=_DOC)

    def run(self, until: float | None = None, *, steps: int | None = None,
            callback: Callable | None = None, every: int | None = None,
            record: Any = None, cache: bool = False, on_stop: str = "raise"):
        """Advance to time ``until`` (must be a whole number of steps: dt is never changed) or
        by ``steps`` steps. ``callback(sim)`` runs every ``every`` steps (both or neither),
        with this Simulation as argument, like the functions of :meth:`watch` (the expert
        ``IVPSolver.run`` passes the solver). A watched simulation (:meth:`watch`) also calls
        the watch every ``poll`` steps; a ``stop`` from the Studio raises ``StopRun`` out of
        here (the run ends where it is).

        ``record`` / ``cache``: the run registry, as ``IVPSolver.run`` (``conatus.runs``): with
        ``record`` given, the recorded run is returned; otherwise this Simulation (chaining).

        ``on_stop``: what a Studio ``stop`` does: ``"raise"``
        (default, v0.1) raises ``cn.StopRun`` out of here; ``"return"`` ends the run where it is
        and returns normally, so that the post-processing after ``run()`` still runs
        (``sim.stopped`` then holds the reason)."""
        n = self._run_arguments(until, steps=steps, callback=callback, every=every,
                                record=record, cache=cache, on_stop=on_stop)
        self.stopped = None
        if on_stop == "return":
            from ..telemetry.sink import StopRun
            try:
                return self.run(until, steps=steps, callback=callback, every=every,
                                record=record, cache=cache)
            except StopRun as stop:
                self.stopped = str(stop) or "stopped from the Studio"
                return self
        steps = n
        if callback is not None:                   # the native callback sees the Simulation
            user = callback
            callback = lambda _solver: user(self)  # noqa: E731
        w = self.watching
        if w is not None and not w.closed:
            callback, every = _combine(w.callback, w.poll, callback, every)
        for o in self.outputs:                     # particles: the velocity at the start
            begin = getattr(o, "begin_run", None)
            if begin is not None:
                begin()
        for o in self.outputs:                     # the row of the start (output(initial=True))
            if getattr(o, "initial_pending", False):
                o.write_initial()
        if self.outputs:                           # cn.io files, each on its own cadence
            from ..io.analysis import _combine as _combine_io
            callback, every = _combine_io([(callback, every)] + [(o, o.every) for o in self.outputs])
        if w is not None and not w.closed and w.settable:
            try:
                self._run_settable(until, steps, callback, every, w.poll)
            finally:
                self._write_sidecars()
            return self
        try:
            out = self.solver.run(steps, callback, every, record=record, cache=cache)
        finally:                                   # also after a stop or a NumericalError:
            self._write_sidecars()                 # the sidecar lists the rows written
        return self if out is None else out

    def _run_arguments(self, until: float | None = None, *, steps: int | None = None,
                       callback: Callable | None = None, every: int | None = None,
                       record: Any = None, cache: bool = False, on_stop: str = "raise") -> int:
        """The checks of :meth:`run` on its arguments, without side effects; returns the number
        of steps. Shared by ``run`` and by ``python -m conatus agent check`` (earlier versions: the dry run stopped at ``run(steps=-1)`` and reported ok)."""
        if on_stop not in ("raise", "return"):
            from ..errors import did_you_mean
            raise OptionError(f"run(on_stop={on_stop!r})",
                              fix="on_stop='raise' (cn.StopRun out of run) or 'return'"
                                  + did_you_mean(str(on_stop), ("raise", "return")), doc=_DOC)
        from ..timestep.integrator import check_callback, check_steps, is_whole
        check_callback(callback, every, "run()")   # callback used to reach IVPSolver.run
        if (until is None) == (steps is None):     # without every and always raise there
            # a fix line with the clock of this simulation
            raise OptionError(
                "run(): give until= or steps=" + ("" if until is None else ", not both"),
                why="the length of a run is yours: an end time (a whole number of steps of dt) "
                    "or a number of steps",
                fix=f"run(until={self.t + 100 * self.dt:.12g}) (t = {self.t:.12g} now, dt = "
                    f"{self.dt:.12g}) or run(steps=100)", doc=_DOC)
        if steps is None:
            # a finite real time (audit inputs: nan and inf failed in int(round(...))
            # with a bare ValueError/OverflowError, True was read as t = 1, "1" was a TypeError)
            from ..errors import check_real
            until = check_real(until, "until", "run()")
            n = (until - self.t) / self.dt
            steps = int(round(n))
            if steps < 0:                          # before the clock: no step count fixes it
                # (earlier versions: the fix line suggested run(steps=-2))
                raise ResolutionError(
                    f"run(until={until}): the simulation is already at t = {self.t:g}, after "
                    f"until", why="a run goes forward in time; until is an absolute time, not a "
                    "duration", fix=f"run(until=t_end) with t_end > {self.t:g}, e.g. "
                    f"run(until={self.t + abs(until - self.t):g}) for the same duration, or "
                    "restart from an earlier checkpoint (sim.restart(...))", doc=_DOC)
            if not is_whole(n):
                raise ResolutionError(
                    f"run(until={until}): {until - self.t:g} is not a whole number of steps "
                    f"dt = {self.dt:g} ({n:.6f})",
                    why="the stop time must be a whole number of time steps from the current time",
                    fix=f"run(steps={math.floor(n)}) or choose until = t + k*dt", doc=_DOC)
        else:                                      # a float is refused, never truncated
            steps = check_steps(steps, "run()", "steps")
        w = self.watching
        if w is not None and not w.closed and w.settable and (record is not None or cache):
            raise OptionError("run(record=...): this run's dt may be changed from the Studio "
                              "(watch(settable=...)); the registry records one constant dt",
                              fix="drop record=/cache=, or watch without settable=", doc=_DOC)
        return steps

    def _write_sidecars(self) -> None:
        """XDMF sidecars of the analysis files: rewritten at the end of each run call;
        a geometry the sidecar does not cover is declared by the output's explain()."""
        from ..io.analysis import AnalysisFile
        for o in self.outputs:
            if isinstance(o, AnalysisFile) and o.writes:
                o.write_xdmf()

    def _run_settable(self, until: float | None, steps: int, callback: Callable, every: int,
                      poll: int) -> None:
        """The run of a watch with a settable dt: the same steps and callbacks as one
        ``solver.run(steps)`` (callbacks fire on iteration multiples, whatever the chunks), in
        chunks ending where the watch reads the commands, so that after a ``set dt`` the steps
        left are recounted: ``until`` stays exact (the setter refuses a dt that would not land
        on it), ``steps=n`` keeps n."""
        from ..timestep.integrator import is_whole
        self._until = until
        try:
            done = 0
            while True:
                if until is not None:
                    n = (until - self.t) / self.dt
                    left = int(round(n))
                    if not is_whole(n):          # never: the setter checks it
                        raise ResolutionError(f"run(until={until}): {n:.6f} steps left after a "
                                              "change of dt", doc=_DOC)
                else:
                    left = steps - done
                if left <= 0:
                    return
                chunk = min(left, poll - self.iteration % poll)
                self.solver.run(chunk, callback, every)
                done += chunk
        finally:
            self._until = None

    @renames("conatus.model.run:Simulation.watch")
    def watch(self, run_id: str, *, film: Any = None, stats: Mapping[str, Any] | None = None,
              every_s: float = 0.5, scalars_every_s: float = 0.25, poll: int = 10,
              film_mb: float = 256.0, max_px: int = 384, max_px3d: int = 48,
              spool: str | Path | None = None, checkpoint_dir: str | Path | None = None,
              meta: Mapping[str, Any] | None = None, controls_every_s: float = 1.0,
              settable: Mapping[str, Any] | None = None):
        """Make the run visible live in the Studio, with its film.

        ``film``: the views, a list of field names (``["T", "u.x"]``) or ``{name: source}``
        with a source = a field name, a function ``f(sim)`` or ``cn.view(...)`` (scale, slice,
        3D mode); the first is the default view. ``stats``: ``{name: f(sim) | cn.series(...) |
        cn.spectrum(...) | cn.profile(...)}``. ``every_s``: wall seconds between two frames of a
        view; ``scalars_every_s``: between two evaluations of the numbers; ``poll``: steps between
        two calls of the watch; ``controls_every_s``: wall seconds between two reads of the Studio
        commands. ``settable={"dt": (lo, hi)}``: the time step may be changed from the Studio
        or an agent (command ``set``, bounds checked, Runge-Kutta schemes only: a multistep scheme, SBDF* or CNAB*, is refused
        with ``settable`` because its history was built with one dt; recorded as an event).
        Note the units: ``every_s``, ``scalars_every_s`` and ``controls_every_s`` are wall-clock
        seconds, ``poll`` is in steps (``output(every=)`` and ``checkpoints(every=)`` count
        steps). The Studio shows nothing else (no hidden physics); the numbers of
        the run are unchanged (tests/studio/test_watch.py). Defaults are declared by
        ``explain()``. Returns the :class:`~conatus.model.watch.Watch`."""
        from .watch import Watch
        if self.watching is not None and not self.watching.closed:
            raise OptionError(f"watch({run_id!r}): already watched as {self.watching.run_id!r}",
                              fix="sim.unwatch() first", doc=_DOC)
        self.watching = Watch(self, run_id, film=film, stats=stats, every_s=every_s,
                              scalars_every_s=scalars_every_s, poll=poll, film_mb=film_mb,
                              max_px=max_px, max_px3d=max_px3d, spool=spool,
                              checkpoint_dir=checkpoint_dir, meta=meta,
                              controls_every_s=controls_every_s, settable=settable)
        return self.watching

    # ------------------------------------------------------------------ files (cn.io)
    def output(self, path: str | Path, tasks: Any, *, every: int, mode: str | None = None,
               initial: bool = True):
        """Write ``tasks`` to the HDF5 analysis file ``path`` every ``every`` steps during
        :meth:`run` (``cn.io.AnalysisFile``): ``["T", "u.x"]`` (grid values of unknowns or
        components) or ``{name: name | f(sim)}``. ``initial`` (default True): the
        state at the start of the next :meth:`run` is written too (the row of t = 0 on a fresh
        start; never twice: a resumed chain already holds it), declared by explain();
        ``initial=False``: the first row is at iteration ``every``. ``mode`` is required if the file exists
        (``"w"`` replaces, ``"a"`` appends, ``"resume"`` appends after a restart from an older
        checkpoint, dropping the rows after the current iteration: declared by explain()). Rows hold ``sim_time``, ``iteration`` and the
        provenance of the run; read back with ``cn.io.read(path)``."""
        from ..io import AnalysisFile
        mode, note = self._chain_mode(path, mode, "output")
        out = AnalysisFile(self, path, tasks, every=every, mode=mode, initial=initial)
        out.chain_note = note
        self.outputs.append(out)
        return out

    def _chain_mode(self, path: str | Path | None, mode: str | None,
                    what: str) -> tuple[str | None, str | None]:
        """The file mode chosen by :meth:`resume_or_start` when ``mode`` is not given (fresh
        start: ``"w"``, an existing file refused; restart: ``"resume"``), with its explain()
        note; ``(mode, None)`` outside a chain."""
        chain = getattr(self, "_chain", None)
        if mode is not None or chain is None or path is None:
            return mode, None
        if chain[0] == "fresh" and Path(path).exists():
            # a fresh start never replaces an existing file silently (outputs
            # left without a checkpoint: a first job killed before its first checkpoint,
            # an emptied or misplaced checkpoint directory, another run's file)
            raise OptionError(
                f"{what}({str(path)!r}): the file exists but resume_or_start found no "
                "checkpoint, so this run starts again from its initial state",
                why="replacing the file would delete rows that no checkpoint can restore; "
                    "a file is never replaced silently",
                fix="mode='w' to replace it (e.g. a first job killed before its first "
                    "checkpoint), or check the checkpoint directory given to "
                    "resume_or_start", doc="docs/site/guide/io.md")
        mode = "w" if chain[0] == "fresh" else "resume"
        return mode, (f"mode {mode!r} chosen by resume_or_start ("
                      + ("fresh start" if chain[0] == "fresh" else "restart") + ")")

    def particles(self, positions: Any, *, path: str | Path | None = None,
                  every: int | None = None, velocity: str = "u", at_walls: str = "refuse",
                  mode: str | None = None, initial: bool = True):
        """Passive tracer particles advected during :meth:`run` (``conatus.model.particles``). ``positions``: (P, d) in coordinate order or
        ``{coord: array}``. The velocity at a particle is the exact spectral series of
        ``velocity`` at its position; each flow step advances the particles by one RK4 step, the
        midpoint velocity interpolated in time (cubic, last four flow states): order 4 in a
        steady flow, 3 in an unsteady one. Periodic axes: unwrapped positions (also written wrapped). ``at_walls``:
        ``"refuse"`` (a step ending outside the walls raises) or ``"clip"`` (put back on the
        wall, counted). ``path`` + ``every``: an HDF5 file of the ``cn.io`` layout (position,
        position_unwrapped, velocity, wall_clips; dims t, particle, component; ``mode`` and
        ``initial`` as :meth:`output`). Cartesian boxes, CPU and GPU. Returns the
        :class:`~conatus.model.particles.Tracers` (``positions()``, ``velocities()``,
        ``explain()``)."""
        from .particles import Tracers
        mode, _note = self._chain_mode(path, mode, "particles")
        out = Tracers(self, positions, path=path, every=every, velocity=velocity,
                        at_walls=at_walls, mode=mode, initial=initial)
        self.outputs.append(out)
        return out

    def dye(self, name: str, path: str | Path, *, every: int, mode: str | None = None,
            initial: bool = True, field: bool = True):
        """Write the dye ``name`` (declared by ``m.dye``, or any scalar unknown) and its
        conservation diagnostics every ``every`` steps during :meth:`run` (``conatus.model.dye.DyeFile``): grid values (``field=True``), ``mass`` = integ(c),
        ``variance``, ``minimum``, ``maximum`` and the spectral ``tail``; ``mode`` and
        ``initial`` as :meth:`output`. Read back with ``cn.io.read(path)``."""
        from .dye import DyeFile
        mode, note = self._chain_mode(path, mode, "dye")
        out = DyeFile(self, name, path, every=every, mode=mode, initial=initial, field=field)
        out.chain_note = note
        self.outputs.append(out)
        return out

    def checkpoints(self, directory: str | Path, *, every: int, keep: int = 2,
                    grids: bool = False):
        """Exact restart files ``directory/checkpoint_<iteration>.h5`` every ``every`` steps
        during :meth:`run`, the ``keep`` most recent kept (``cn.io.CheckpointSeries``; default
        ``conatus.io.checkpoint.CHECKPOINT_KEEP`` = 2: the latest and the one before it). The next job resumes with ``sim.restart(cn.io.latest(directory))``."""
        from ..io import CheckpointSeries
        out = CheckpointSeries(self.solver, directory, every=every, keep=keep, grids=grids)
        self.outputs.append(out)
        return out

    def properties(self, *, every: int, **props: Any):
        """Named scalar properties of the flow, evaluated every ``every`` steps during :meth:`run`
        and reduced by ``props.max("Re")``, ``min``, ``grid_average``,
        ``volume_average``, ``integral``. A property is an unknown or component name or a
        function ``f(sim)`` of grid values (``conatus.diagnostics.properties``)."""
        from ..diagnostics.properties import FlowProperties
        out = FlowProperties(self, every, **props)
        self.outputs.append(out)
        return out

    def save(self, path: str | Path, *, grids: bool = False) -> Path:
        """Checkpoint now: state, multistep history and clock (``.h5``: HDF5 with provenance,
        readable by ``cn.io.read``; ``.npz``: NumPy). A restart continues bit-for-bit. A suffix
        naming another format (``.hdf``, ``.nc``, ...) is refused: it used to be written as a
        NumPy file with ``.npz`` appended (review of the inputs audit)."""
        from ..timestep.integrator import check_checkpoint_suffix
        check_checkpoint_suffix(Path(path), "sim.save")
        return self.solver.save_checkpoint(path, grids=grids)

    def restart(self, path: str | Path | None, *, allow: Any = ()) -> Simulation:
        """Continue from a checkpoint (``save`` / ``checkpoints``); the scheme, dt, resolution
        and options must be those of the file (checked; a difference is refused), and so must the model,
        its equations, its parameters and its box: a difference is
        refused with the list of what differs. ``allow`` names what may differ deliberately,
        e.g. ``allow=("kappa",)`` to continue this state at a new kappa (also ``"parameters"``,
        ``"model"``, ``"equations"``, ``"geometry"``); it is printed by :meth:`explain` and
        carried by every later checkpoint. ``None`` (what ``cn.io.latest`` returns for an empty
        directory) is refused with an explanation."""
        if path is None:
            raise OptionError("restart(None): no checkpoint found",
                              why="cn.io.latest(directory) found no complete checkpoint file",
                              fix="start with sim.initial(...) on the first job", doc="docs/site/guide/io.md")
        self.solver.load_checkpoint(path, allow=allow)
        self._restarted = f"{path} (iteration {self.solver.iteration}, t = {self.solver.t:g})"
        self._chain = None
        return self

    def resume_or_start(self, directory: str | Path, *, allow: Any = (), check: bool = True,
                        t: Any = None, **initial: Any) -> Simulation:
        """One script for every job of a chain: continue from the
        latest checkpoint of ``directory`` (``cn.io.latest``) if it holds one, else start from
        ``initial`` (as :meth:`initial`; the directory is created). Afterwards an
        :meth:`output` without ``mode`` creates its file on a fresh start (``"w"``; an existing
        file is refused, never replaced silently: ``mode="w"`` replaces it explicitly) and continues it after a restart (``"resume"``: rows written
        after the checkpoint are dropped, the exact restart writes them again). Which branch was
        taken, and each output's mode, are declared by :meth:`explain`. Call it before
        :meth:`output` and :meth:`checkpoints`. The checkpoint must be a state of this run
        (model, equations, parameters, resolution, box: :meth:`restart`; ``allow=`` names what
        may differ); ``check`` is passed to :meth:`initial` on a fresh start.
        Both checkpoint formats are found (``checkpoint_*.h5`` and ``checkpoint_*.npz``).
        ``t``: the start time of a fresh start (:meth:`initial`); on a resume it must be the
        start time recorded by the chain's checkpoints (refused otherwise: the script would
        describe another run)."""
        from ..io import latest
        t0 = self._initial_time(t, initial, "resume_or_start")
        d = Path(directory)
        if not initial:
            raise OptionError("resume_or_start(directory): no initial state given",
                              why="the first job of the chain starts from it",
                              fix="sim.resume_or_start(directory, T=lambda x, z: ...)",
                              doc="docs/site/guide/io.md")
        created = not d.exists()
        d.mkdir(parents=True, exist_ok=True)
        ck = latest(d)
        if ck is None:
            self.initial(check=check, t=t0, **initial)
            self._chain = ("fresh", f"resume_or_start({str(directory)!r}): no checkpoint there "
                           + ("(directory created)" if created else "(empty directory)")
                           + ": started from the initial state; outputs without mode= create "
                             "their file (an existing one is refused)")
        else:
            self.restart(ck, allow=allow)
            self._check_chain_start(ck, t0)
            self._chain = ("resumed", f"resume_or_start({str(directory)!r}): continued from "
                           f"{ck.name} (iteration {self.solver.iteration}, t = "
                           f"{self.solver.t:g}); outputs without mode= continue their file "
                           "(mode='resume')")
        return self

    def _check_chain_start(self, ck: Path, t0: float | None) -> None:
        """The start time of the resumed chain (recorded by its checkpoints) against the ``t`` of
        this script (``resume_or_start(..., t=...)``): a difference is refused."""
        rec = getattr(self.solver, "_t_initial", None)
        want = 0.0 if t0 is None else t0
        if rec is None:
            if want != 0.0:
                raise OptionError(
                    f"resume_or_start(t={want:g}): {ck.name} does not record the start time of its "
                    "chain (written before checkpoints recorded it)",
                    why="the start time given here cannot be checked against the chain; before "
                        "the record, sim.initial() always started native runs at t = 0",
                    fix="continue this chain without t= (its clock is kept: t = "
                        f"{self.solver.t:g}), or start a new chain in a new directory", doc=_DOC)
            return
        if rec != want:
            raise OptionError(
                f"resume_or_start(t={want:g}): the chain of {ck.name} started at t = {rec:g}",
                why="the script gives another start time than the run it would continue: the "
                    "forcing, boundary data and outputs of the two would not be on one clock",
                fix=f"resume_or_start(..., t={rec!r}) for this chain, or a new directory for a run "
                    f"starting at t = {want:g}", doc=_DOC)

    def unwatch(self) -> None:
        """Stop watching: the final status is sent and the telemetry writer stops."""
        if self.watching is not None:
            self.watching.close()

    def _steps_of(self, T: float, what: str) -> int:
        """Whole number of steps in the horizon T (refused otherwise: dt is never adjusted)."""
        from ..errors import check_real
        from ..timestep.integrator import is_whole
        T = check_real(T, "T", f"{what}()", positive=True)   # audit inputs: nan/inf/str were bare
        n = T / self.dt
        steps = int(round(n))
        if steps < 1 or not is_whole(n):
            raise ResolutionError(f"{what}(T={T}): T is not a whole number of steps dt = {self.dt:g} "
                                  f"({n:.6f})", why="the horizon is part of the problem and is "
                                  "never rounded", fix=f"T = {max(steps, 1) * self.dt!r}", doc=_DOC)
        return steps

    def flowmap(self, T: float, *, t0: float | None = None, checkpoints: int | None = None):
        """Exact discrete tangent and adjoint of this simulation over the horizon ``T``
        (``conatus.adjoint.FlowMap``: ``jvp``, ``vjp``, Revolve with ``checkpoints`` slots;
        ``t0`` required when the equations depend on time). The state vectors are those of
        ``sim.solver`` (``sim.solver.X``); the tau unknowns are carried, never shown. Calling the
        map, ``Φ(X0)``, leaves this simulation where it was (state, time, iteration, multistep
        history): the map runs the solver and puts them back (so
        ``sim.flowmap(T)(X0)`` does not move the simulation to X_N at time t0 + T)."""
        from ..adjoint import FlowMap
        fm = FlowMap(self.solver, self._steps_of(T, "flowmap"), t0=t0, checkpoints=checkpoints)
        fm.restore = True
        return fm

    def nlop(self, *, T: float, E0: float, fields: tuple[str, ...] | None = None,
             t0: float | None = None, checkpoints: int | None = None, **kw: Any):
        """Nonlinear optimal perturbation of this simulation (``NLOP.from_solver``): maximise the
        energy ½∫|u|² dV at time T over seeds of energy ``E0``. ``fields``: the velocity unknowns
        by default (declared with ``m.velocity``); pressure and taus never enter the energy."""
        from ..adjoint.nlop import NLOP
        if fields is None:
            fields = tuple(u.name for u in self.compiled.model.unknowns if u.role == "velocity")
            if not fields:
                raise OptionError("nlop(): no velocity unknown declared",
                                  fix="m.velocity('u') in the model, or nlop(fields=('u',))",
                                  doc=_DOC)
        self._steps_of(T, "nlop")
        return NLOP.from_solver(self.solver, T=T, E0=E0, fields=tuple(fields), t0=t0,
                                checkpoints=checkpoints, **kw)

    def lyapunov(self, n_vectors: int, *, transient: float, align: float, T: float,
                 reortho_every: int, inner: Any, n_blocks: int, seed: int, starts: Any = None,
                 zero_mean: Any = (), project: Callable | None = None, covariant: bool = False,
                 clv_window: float | None = None, clv_backward: float | None = None,
                 clv_every: int = 1, clv_checkpoints: int | None = None,
                 progress: Callable | None = None):
        """Lyapunov exponents (and covariant Lyapunov vectors) of this simulation's discrete map
        (``conatus.chaos.lyapunov_spectrum``): the ``n_vectors`` leading
        exponents with their standard errors, from the exact tangent of the step (same scheme,
        same dt), on one or several trajectories integrated as one batch (CPU or GPU, the backend
        of the simulation). Durations are times, each a whole number of steps (``transient``,
        ``align``) or of intervals ``reortho_every * dt`` (``T``, ``clv_window``,
        ``clv_backward``), never rounded:

        * ``transient``: trajectory only; ``align``: trajectory and tangent vectors (alignment,
          no averaging); ``T``: averaging time per trajectory, in ``n_blocks`` blocks;
        * ``inner``: ``"euclidean"`` (on the state vector of coefficients) or a positive diagonal
          weight; ``seed``: the random initial tangent vectors;
        * ``starts``: None (the current state), an array (B, n) of states, or a list of
          ``sim.initial`` keyword dicts (B trajectories); the simulation's state is not advanced;
        * ``zero_mean=("u",)``: tangent vectors restricted to zero mean for these unknowns (all
          axes periodic: the invariant subspace of conserved means, e.g. Kuramoto–Sivashinsky),
          or ``project=f`` (V -> P V) for another invariant subspace;
        * ``covariant=True``: the CLVs of Ginelli et al. (2007) over a window of length
          ``clv_window`` ending ``clv_backward`` before the end (backward transient), kept every
          ``clv_every`` interval; ``clv_checkpoints=s`` bounds the memory (s checkpoints and one
          recomputation, see ``lyapunov_spectrum``).

        The result (``LyapunovResult``) has ``explain()`` (exponents ± errors, Kaplan–Yorke
        dimension, CLV memory, covariance residual, principal angles), ``angles(j)``,
        ``pair_angle(i, j)``, ``domination(tau)`` (domination of the Oseledets splitting,
        entangled clusters, physical-mode count) and ``clv_grid(name, i)`` (a CLV as grid
        values)."""
        from .lyapunov_native import simulation_lyapunov
        return simulation_lyapunov(self, n_vectors, transient=transient, align=align, T=T,
                                   reortho_every=reortho_every, inner=inner, n_blocks=n_blocks,
                                   seed=seed, starts=starts, zero_mean=zero_mean,
                                   project=project, covariant=covariant, clv_window=clv_window,
                                   clv_backward=clv_backward, clv_every=clv_every,
                                   clv_checkpoints=clv_checkpoints, progress=progress)

    def grid(self, name: str) -> np.ndarray:
        """Values of an unknown on the grid (``coordinate(c)`` gives the points). On a
        sine/cosine axis only the physical interval [a, b] is returned (N + 1 points, both
        walls included), not the doubled period."""
        self._check_unknown(name, "grid")
        g = self.solver.grid(name)
        if self.compiled.symmetry is not None:
            from .sincos import physical_slice
            ax, sl = physical_slice(self.compiled)
            rank = self.compiled.fields[name].rank
            g = np.take(g, np.arange(g.shape[rank + ax])[sl], axis=rank + ax)
        return _remember_grid(g)

    def evaluate(self, expr: str) -> Any:
        """Grid values of an expression of the current state written as in the equations
        (``"dz(T)"``, ``"0.5*u@u"``, ``"Ra*T - p"``): unknowns, given fields, parameters and
        the time ``t``; on the grid of :meth:`grid` (constant directions broadcast). An
        expression constant in space (a full integral ``"integ(u@u)/2"``, a parameter) is
        returned as a number (Python float or complex; a vector: its components), not as a grid
        array. The
        pseudospectral products are those of the right-hand side (dealiased grid, then the
        first N modes kept). Cartesian geometries (the curvilinear evaluators keep their own
        layouts: refused there)."""
        from ..errors import UnsupportedError
        from ..ir.parse import Namespace, parse_expr
        from ..reference import transforms as T
        comp, sv = self.compiled, self.solver
        if comp.coordsys.curvilinear:
            raise UnsupportedError(f"evaluate({expr!r}) on {comp.model.geometry!r}",
                                   why="only Cartesian layouts are evaluated here",
                                   fix="a function f(sim) of sim.grid(...) values", doc=_DOC)
        if not isinstance(expr, str):
            raise InputTypeError(f"evaluate({expr!r}): an expression as text", doc=_DOC)
        values: dict = dict(comp.params)
        values.update({n: f for n, f in comp.fields.items() if not n.startswith("tau_")})
        tname = None
        if comp.time is not None:
            values[comp.time.name] = comp.time
        elif "t" not in values:
            # the time t is a name of the native layer even when no equation uses it
            # (earlier versions: evaluate("t") was refused although t is valid in equations)
            tname, values["t"] = "t", Parameter("t", 0.0)
        e = parse_expr(expr, Namespace.of(values, layer="native"))
        data = dict(sv._ncc)
        data.update(zip(sv.system.variables, sv.system.scatter(sv.X), strict=True))
        params = dict(sv.plan.params)
        if sv.time_parameter:
            params[sv.time_parameter] = sv.t
        if tname is not None:
            params[tname] = sv.t
        ev = sv.evaluator
        if comp.symmetry is not None and comp.model.geometry.sincos_coord is not None:
            # integrals over the physical interval [a, b], not the doubled period (finding 7)
            ev = getattr(self, "_physical_ev", None)
            if ev is None or ev.domain is not sv.plan.domain:
                from .sincos import physical_evaluator
                ev = self._physical_ev = physical_evaluator(comp, sv)
        c = ev.evaluate(e, data, params)
        if sv.backend != "cpu":
            c = sv.xp.asnumpy(c)
        g, rank = np.asarray(c), e.rank
        coords = comp.coordsys.coords
        shape = []
        for i, cc in enumerate(coords):
            b = sv.plan.domain.basis(cc)
            ax = rank + i
            if b is not None and g.ndim > ax and g.shape[ax] == b.size:
                g = T.backward(g, b, axis=ax, dealias=False)
            shape.append(T.grid_size(b, dealias=False) if b is not None else 1)
        if np.iscomplexobj(g) and not sv.plan.layout.complex:
            # a real problem: imaginary parts are round-off of the transforms, checked, not dropped
            # blindly
            if float(np.abs(g.imag).max(initial=0.0)) > 1e-12 * max(1.0, float(np.abs(g.real).max(initial=0.0))):
                raise UnsupportedError(f"evaluate({expr!r}): complex values in a real problem", doc=_DOC)
            g = g.real
        g = np.broadcast_to(g, g.shape[:rank] + tuple(shape)).copy()
        dom = getattr(e, "domain", None)
        if dom is None or all(dom.basis(cc) is None for cc in coords):
            # constant in space (full integral, parameters): the number itself
            v = g[(Ellipsis,) + (0,) * len(coords)]
            if rank == 0:
                v = v.item()
                return float(v) if not isinstance(v, complex) else complex(v)
            return np.asarray(v)
        if comp.symmetry is not None:
            from .sincos import physical_slice
            ax, sl = physical_slice(comp)
            g = np.take(g, np.arange(g.shape[rank + ax])[sl], axis=rank + ax)
        return g

    def coordinate(self, name: str) -> np.ndarray:
        """Grid points of coordinate ``name`` matching ``grid()`` (a sine/cosine axis: the N + 1
        points of [a, b]; disk, annulus, pipe, sphere, ball, shell: the solver's own grids)."""
        names = list(self.compiled.model.geometry.coord_names)
        if name not in names:
            raise InputTypeError(f"coordinate({name!r}): not a coordinate of this geometry",
                                 fix="coordinates: " + ", ".join(names), doc=_DOC)
        space = getattr(self.plan, "space", None)
        if space is not None and hasattr(space, "grids"):
            # curvilinear plans (the reference grids of the Cartesian bases
            # were used here, NotImplementedError for DiskBasis and the other curved bases)
            # a copy: the sphere's theta is the plan's cached grid (an in-place edit of the
            # returned array changed every later coordinate(), audit state)
            return np.array(space.grids(False)[names.index(name)], copy=True)
        from ..reference.transforms import grid as _grid
        g = np.asarray(_grid(self.compiled.bases[name], dealias=False))
        if name == self.compiled.model.geometry.sincos_coord:
            g = g[: self.compiled.N[name] + 1]
        return g

    def coeffs(self, name: str) -> np.ndarray:
        """Spectral coefficients (expert layout; a sine/cosine axis: the RealFourier(2N)
        coefficients of the doubled period, cos/-sin pairs, half of them zero by parity)."""
        self._check_unknown(name, "coeffs")
        return self.solver.coeffs(name)

    def cfl(self, velocity: str = "u", safety: float = REQUIRED) -> float:
        """CFL time step of the current state (a stability indicator, not an error control).
        ``safety`` has no default (``REQUIRED`` in ``help()``): it scales the returned step
        (0.5 before v0.1)."""
        from ..timestep import CFL
        if safety is None or safety is REQUIRED:
            raise OptionError("sim.cfl(): safety must be given explicitly (no default since v0.1)",
                              why="the returned step is safety / max(|u_i| / dx_i): the safety "
                                  "factor is your margin on the advective limit, a numerical "
                                  "choice (it is not an error control)",
                              fix=f"sim.cfl({velocity!r}, safety=0.5)", doc=_DOC)
        return CFL(self.solver, velocity, safety=safety).compute()

    def methods(self):
        """The Methods paragraph (conatus.methods) with the native-layer decisions."""
        from ..methods import MethodsText, methods
        mt = methods(self.plan, self.solver)
        return MethodsText(mt.text + " " + _native_sentence(self.compiled), mt.citations,
                           dict(mt.facts, native=True, dealias_default=self.compiled.dealias_default))

    def _parameter_lines(self) -> list[str]:
        """The values compiled into this simulation: a later
        ``m.<name> = value`` reaches the next simulate(), not this one (said here)."""
        comp = self.compiled
        if not comp.params:
            return []
        now = comp.model.params
        used = []
        for k, p in comp.params.items():
            v = p.value
            note = ("" if k not in now or now[k] == v else
                    f" (the model has {k} = {now[k]:g} since: rebuild with m.simulate(...))")
            used.append(f"{k} = {v:g}{note}")
        return ["Parameters in use (compiled into this simulation): " + ", ".join(used)]

    def explain(self, *, full: bool = False) -> str:
        """The run in a few lines: resolution, scheme, time step, backend, the
        summary of the model's decisions, the initial state, the checks, outputs and warnings.
        ``full=True``: every automatic decision of the compiled model (term-by-term split, taus
        and their modes, gauge, bases) and the whole solver layout."""
        comp = self.compiled
        sol = self.solver
        if not full:
            lines = [f"Simulation of '{comp.problem.name}' at N = "
                     + ", ".join(f"{c}: {n}" for c, n in comp.N.items())
                     + f": {sol.scheme}, dt = {sol.dt:g}, backend {sol.backend}"]
            if getattr(self, "operators_note", None):
                lines.append("  " + self.operators_note)
            lines += self._parameter_lines()
            lines += comp.summary()
            lines += self._state_lines()
            lines.append(sol.explain())
            lines += [o.explain().splitlines()[0] + (f" [{o.chain_note}]" if getattr(
                o, "chain_note", None) else "") for o in self.outputs]
            if self.watching is not None and not self.watching.closed:
                lines.append(self.watching.explain().splitlines()[0])
            lines.append("(compiler decisions, tau by tau and term by term, and the solver layout: "
                         "sim.explain(full=True))")
            return "\n".join(lines)
        lines = [f"Simulation of '{comp.problem.name}' (native model) at N = "
                 + ", ".join(f"{c}: {n}" for c, n in comp.N.items())]
        lines += self._parameter_lines()
        lines += comp.decisions()
        if getattr(self, "operators_note", None):
            lines.append("  " + self.operators_note)
        lines += self._state_lines()
        lines.append(self.solver.explain(full=True))
        for o in self.outputs:
            lines.append(o.explain() + (f" [{o.chain_note}]" if getattr(o, "chain_note", None)
                                        else ""))
        if self.watching is not None and not self.watching.closed:
            lines.append(self.watching.explain())
        return "\n".join(lines)

    def _state_lines(self) -> list[str]:
        """explain() lines on where the state comes from (chain, restart, initial values)."""
        comp = self.compiled
        lines = []
        given = getattr(self, "_given", None)
        chain = getattr(self, "_chain", None)
        if chain is not None:
            lines.append(chain[1])
        if getattr(self, "_restarted", None):
            t_init = getattr(self.solver, "_t_initial", None)     # recorded by the checkpoint
            lines.append("State restarted from " + self._restarted
                         + (f" (its chain started at t = {t_init:g}: initial(t=...))" if t_init else ""))
            given = None
        if given is not None:
            rest = [u.name for u in comp.model.unknowns if u.name not in given]
            t0 = getattr(self, "_t_given", None)
            when = (f" at t = {t0:g} (initial(t=...): forcing, boundary data and outputs on this "
                    "clock, recorded by the checkpoints)" if t0 is not None else "")
            lines.append("Initial state: " + ", ".join(given) + (f" given; {', '.join(rest)} start "
                                                                  "at zero" if rest else " given")
                         + when)
            lines += getattr(self, "_initial_notes", None) or []
        return lines


def _combine(watch_cb: Callable, poll: int, cb: Callable | None,
             every: int | None) -> tuple[Callable, int]:
    """One callback for the solver: the watch every ``poll`` steps and the user's every
    ``every`` steps (called on their gcd, each one only on its own multiples)."""
    if cb is None:
        return watch_cb, poll
    g = math.gcd(int(poll), int(every))

    def both(solver: Any) -> None:
        if solver.iteration % every == 0:
            cb(solver)
        if solver.iteration % poll == 0:
            watch_cb(solver)
    return both, g
