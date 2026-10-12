"""Watching a native simulation in a monitor: ``sim.watch(...)``.

    import conatus as cn
    sim = m.simulate(N=(256, 128), scheme="RK443", dt=2e-3)
    w = sim.watch("kh_demo",
                  film={"vorticity": cn.view(vorticity, scale="signed"),   # first = default view
                        "dye": cn.view("c", scale="dye")},
                  stats={"energy": energy,                                 # a scalar per step
                         "E(k)": cn.spectrum("u.x", axis="x"),             # a curve per frame
                         "<c>(z)": cn.profile("c", along="z")},
                  every=0.5)                                               # s of wall time per frame
    print(w.explain())             # everything that will be sent, losses included
    sim.run(until=20.0)            # the run is watched; numbers identical to an unwatched run

What a monitor shows is **only what is declared here** (no hidden physics): a view is an unknown
(``"T"``), a component of a vector unknown (``"u.x"``), or a function of the simulation you write;
a statistic is a function returning a number, or one of the two declared reductions
:func:`spectrum` and :func:`profile`, whose formulas :meth:`Watch.explain` prints. Every lossy step
(stride, 8 bits, linear resampling of non-uniform nodes, the film's thinning) is declared in the
frames and in ``explain()``. The computation is never changed: the watch only reads the state, in a
callback, and the telemetry writer runs in its own thread (``conatus.telemetry``).
"""
from __future__ import annotations

import atexit
import math
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._deprecation import renames
from ..errors import InputTypeError, OptionError

_DOC = "docs/site/tutorials/09_studio.md"
# What a ``set`` command may change on a native run (with bounds declared by the user).
SETTABLE = {"dt": "time step"}
MODES_3D = ("isosurface", "volume", "slices")
SCALES = ("smoke", "signed", "dye")

__all__ = ["view", "spectrum", "profile", "series", "View", "Stat", "Watch"]


# ------------------------------------------------------------------------------ declarations
@dataclass(frozen=True)
class View:
    """A picture of the flow: what (``source``), how (``scale``, ``mode``), where (``at``)."""
    source: str | Callable[[Any], Any]
    scale: str = "smoke"
    label: str | None = None
    at: Mapping[str, float] | None = None
    mode: str | None = None          # None: a 2D film; isosurface | volume | slices: a 3D view

    def __repr__(self) -> str:
        # a function source by its name, not its address
        extra = "".join(f", {k}={v!r}" for k, v in (("label", self.label), ("at", self.at),
                                                    ("mode", self.mode)) if v is not None)
        return f"cn.view({_source_name(self.source)}, scale={self.scale!r}{extra})"

    def explain(self) -> str:
        """What a monitor shows for this view."""
        what = (f"the field {self.source}" if isinstance(self.source, str)
                else f"the grid values returned by {_source_name(self.source)}(sim)")
        how = {"smoke": "sequential colour scale", "signed": "signed colour scale (zero black)",
               "dye": "tracer (dye) colour scale"}[self.scale]
        where = (f", 3D {self.mode}" if self.mode else
                 (", the 2D slice at the grid node nearest to " + ", ".join(
                     f"{k} = {v:g}" for k, v in self.at.items())) if self.at else ", a 2D film")
        return f"view of {what}: {how}{where}" + (f", label {self.label!r}" if self.label else "")


def view(source: str | Callable[[Any], Any], *, scale: str = "smoke", label: str | None = None,
         at: Mapping[str, float] | None = None, mode: str | None = None) -> View:
    """Declare a view for :meth:`Simulation.watch`.

    ``source``: an unknown (``"T"``), a component of a vector unknown (``"u.x"``) or a function
    ``f(sim)`` returning grid values. ``scale``: ``"smoke"`` (sequential), ``"signed"`` (zero is
    black: vorticity, a perturbation), ``"dye"`` (fluorescein: a tracer). ``at={"y": 0.0}``: the
    2D slice of a 3D field at the grid node nearest to y = 0 (declared). ``mode``:
    ``"isosurface"``, ``"volume"`` or ``"slices"`` makes it a 3D view of a 3D field."""
    if scale not in SCALES:
        raise OptionError(f"view(scale={scale!r}): a monitor scales are {SCALES}",
                          fix='scale="signed" for a field of both signs, "dye" for a tracer',
                          doc=_DOC)
    if mode is not None and mode not in MODES_3D:
        raise OptionError(f"view(mode={mode!r}): 3D modes are {MODES_3D}", doc=_DOC)
    if mode is not None and at is not None:
        raise OptionError("view(): give at= (a 2D slice) or mode= (a 3D view), not both", doc=_DOC)
    if not (isinstance(source, str) or callable(source)):
        raise InputTypeError(f"view({source!r}): an unknown name, 'u.x', or a function of sim",
                             doc=_DOC)
    return View(source, scale, label, dict(at) if at else None, mode)


@dataclass(frozen=True)
class Stat:
    """A declared statistic: ``series`` (a number per step), ``spectrum`` or ``profile``."""
    kind: str
    source: str | Callable[[Any], Any]
    axis: str | None = None
    label: str | None = None
    logy: bool = False
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        # the call that makes it, a function source by its name
        lab = f", label={self.label!r}" if self.label is not None else ""
        if self.kind == "series":
            return f"cn.series({_source_name(self.source)}{lab}" + (", logy=True)" if self.logy else ")")
        if self.kind == "profile":
            return f"cn.profile({self.source!r}, {self.axis!r}{lab})"
        if self.kind == "spectrum":
            return f"cn.spectrum({self.source!r}, {self.axis!r}{lab})"
        return f"Stat({self.kind!r}, {_source_name(self.source)}, axis={self.axis!r}{lab})"

    def explain(self) -> str:
        """What a monitor plots for this statistic."""
        src = (self.source if isinstance(self.source, str)
               else f"{_source_name(self.source)}(sim)")
        if self.kind == "series":
            text = f"time series of {src}, one number per callback" + (
                ", log axis" if self.logy else "")
        elif self.kind == "profile":
            text = f"profile of {self.source} against {self.axis}: mean over the other coordinates"
        elif self.kind == "spectrum":
            text = (f"one-sided power spectrum of {self.source} along the periodic axis "
                    f"{self.axis}, averaged over the others (sum_k E(k) = mean(f^2)), log-log")
        else:
            text = f"statistic {self.kind} of {src}"
        return text + (f", label {self.label!r}" if self.label else "")


def _source_name(source: Any) -> str:
    """A view or statistic source as text: the field name, or the function's name."""
    if isinstance(source, str):
        return repr(source)
    return getattr(source, "__name__", None) or type(source).__name__


def spectrum(source: str, axis: str, *, label: str | None = None) -> Stat:
    """One-sided power spectrum of a field along a **periodic** axis, averaged over the others:
    with ``f_k`` the discrete Fourier coefficients of the grid values (``rfft / N``),
    ``E(k) = c_k |f_k|^2`` (``c_k = 2`` except ``k = 0`` and the Nyquist mode), so that
    ``sum_k E(k) = mean(f^2)`` (Parseval, checked by test). Shown against the wavenumber
    ``2 pi k / L`` on log-log axes; ``k = 0`` (the mean) is left out of the curve."""
    if not isinstance(source, str):
        raise InputTypeError("spectrum(): the field is a name ('T', 'u.x')", doc=_DOC)
    return Stat("spectrum", source, axis, label, True)


def profile(source: str, along: str, *, label: str | None = None) -> Stat:
    """Mean of a field over every coordinate but ``along``, against the grid nodes of ``along``
    (their true positions, Chebyshev nodes included). On a 1D domain: the field itself."""
    if not isinstance(source, str):
        raise InputTypeError("profile(): the field is a name ('T', 'u.x')", doc=_DOC)
    return Stat("profile", source, along, label, False)


def series(f: Callable[[Any], float], *, label: str | None = None, logy: bool = False) -> Stat:
    """A number per callback, ``f(sim)`` (energy, a Nusselt number...); ``logy`` for its plot.
    A plain function in ``stats={...}`` means ``series(f)``."""
    if not callable(f):
        raise InputTypeError("series(f): f is a function of the simulation", doc=_DOC)
    return Stat("series", f, None, label, bool(logy))


# ------------------------------------------------------------------------------ the watch
class Watch:
    """A watched simulation: telemetry, film and declared views. Made by :meth:`Simulation.watch`;
    closed by :meth:`close` (``sim.unwatch()``) or at interpreter exit (declared)."""

    @renames("conatus.model.watch:Watch.__init__")
    def __init__(self, sim: Any, run_id: str, *, film: Any = None,
                 stats: Mapping[str, Any] | None = None, every_s: float = 0.5,
                 scalars_every_s: float = 0.25, poll: int = 10, film_mb: float = 256.0,
                 max_px: int = 384, max_px3d: int = 48, spool: str | Path | None = None,
                 checkpoint_dir: str | Path | None = None,
                 meta: Mapping[str, Any] | None = None, controls_every_s: float = 1.0,
                 settable: Mapping[str, Any] | None = None) -> None:
        from ..telemetry import Telemetry, default_spool
        from ..telemetry.protocol import RUN_ID_RE
        if int(poll) < 1:
            raise OptionError(f"watch(poll={poll}): the callback runs every poll >= 1 steps",
                              doc=_DOC)
        if not every_s > 0 or not scalars_every_s > 0:
            raise OptionError("watch(): every_s and scalars_every_s are positive wall times (s)",
                              doc=_DOC)
        self.sim, self.run_id, self.poll = sim, run_id, int(poll)
        self.every, self.scalars_every = float(every_s), float(scalars_every_s)
        comp = sim.compiled
        self._names = comp.model.geometry.coord_names
        self._geo = comp.model.geometry
        self._nodes = _nodes(comp)
        self._unknowns = {u.name: u.rank for u in comp.model.unknowns}
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
            raise InputTypeError(f"watch({run_id!r}): a run id is 1-200 letters, digits, '.', '_' "
                                 "or '-', starting with a letter or digit (a spool file name and "
                                 "a key on the relay and the hub)", fix='sim.watch("kh_demo_2", ...)',
                                 doc=_DOC)
        self.views = self._parse_film(film)
        self.stats = self._parse_stats(stats)
        self._check_names()
        spool = Path(spool) if spool is not None else default_spool()
        if checkpoint_dir is None:          # declared default: next to the spool
            checkpoint_dir = (spool.parent / "checkpoints" if spool == default_spool()
                              else spool / "checkpoints")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.film_mb = float(film_mb)
        if int(max_px) < 2 or int(max_px3d) < 2:
            raise OptionError(f"watch(max_px={max_px}, max_px3d={max_px3d}): at least 2 points "
                              "per axis", doc=_DOC)
        self.max_px, self.max_px3d = int(max_px), int(max_px3d)
        self.settable = self._parse_settable(settable)
        views, stats = self._declare()
        sv = sim.solver
        info = dict(model=comp.model.name, N=dict(comp.N), scheme=str(getattr(sv, "scheme", "")),
                    dt=float(sv.dt), backend=getattr(sv, "backend", None),
                    parameters={k: _plain(v) for k, v in comp.model.params.items()},
                    checkpoint_dir=str(self.checkpoint_dir))     # where checkpoint/fork write (the app says so)
        self.controls_every = float(controls_every_s)
        self.tel = Telemetry(run_id, spool=spool, capacity=4096, scalar_interval=self.scalars_every,
                             control_interval=self.controls_every,
                             frame_interval=self.every, max_px=self.max_px,
                             max_px3d=self.max_px3d, film_max_bytes=int(self.film_mb * 2**20),
                             views=views, stats=stats, meta={**info, **dict(meta or {})},
                             header_extra={"watch_explain": self.describe()},
                             controls=[{"name": n, "min": lo, "max": hi, "label": SETTABLE[n]}
                                       for n, (lo, hi) in self.settable.items()] or None)
        self._cb = self.tel.ivp_callback(
            scalars={n: _on_sim(sim, s.source) for n, s in self.stats.items() if s.kind == "series"},
            frames={n: (self._frame_fn(v), self._frame_opts(v)) for n, v in self.views.items()
                    if v.mode is None},
            frames3d={n: (self._frame_fn(v), self._frame3d_opts(v)) for n, v in self.views.items()
                      if v.mode is not None},
            curves={n: (self._curve_fn(s), self._curve_opts(s)) for n, s in self.stats.items()
                    if s.kind != "series"},
            checkpoint_dir=self.checkpoint_dir,
            setters={"dt": self._set_dt} if "dt" in self.settable else None)
        self.closed = False
        ref = weakref.ref(self)
        atexit.register(lambda: (lambda w: w.close() if w is not None else None)(ref()))

    # ------------------------------------------------------------------ parsing
    def _parse_settable(self, settable: Mapping[str, Any] | None) -> dict[str, tuple[float, float]]:
        """``settable={"dt": (lo, hi)}``: what a ``set`` command may change, with explicit
        bounds. Only the time step for now, and only for a scheme whose step can change."""
        out: dict[str, tuple[float, float]] = {}
        for name, b in dict(settable or {}).items():
            if name not in SETTABLE:
                raise OptionError(f"watch(settable={{{name!r}: ...}}): only {sorted(SETTABLE)} can be "
                                  "changed by a monitor",
                                  why="a parameter of the equations is compiled into the operators; "
                                      "changing it is a new run (fork from a checkpoint)",
                                  fix='settable={"dt": (1e-4, 1e-2)}', doc=_DOC)
            try:
                lo, hi = (float(x) for x in b)
            except (TypeError, ValueError):
                raise OptionError(f"watch(settable={{{name!r}: {b!r}}}): give the bounds (lo, hi)",
                                  why="a parameter changed from outside the run is checked against "
                                      "bounds you declare; none is assumed",
                                  fix=f'settable={{"{name}": (1e-4, 1e-2)}}', doc=_DOC) from None
            if not (0 < lo < hi and math.isfinite(hi)):
                raise OptionError(f"watch(settable={{{name!r}: ({lo:g}, {hi:g})}}): need 0 < lo < hi",
                                  doc=_DOC)
            sv = self.sim.solver
            if name == "dt" and (getattr(sv, "order", 0) or getattr(sv, "family", None) == "cn"):
                raise OptionError(f"watch(settable={{'dt': ...}}): {getattr(sv, 'scheme', '?')} is "
                                  "constant-step (multistep history built with one dt)",
                                  fix="an IMEX Runge-Kutta scheme (RK222, RK443, ...) or no "
                                      "settable dt", doc=_DOC)
            out[str(name)] = (lo, hi)
        return out

    def _set_dt(self, solver: Any, value: float) -> dict:
        """Setter of the declared ``dt`` (called between two steps by a checked ``set``). With
        ``run(until=T)``, the time left must stay a whole number of steps: else refused, with
        the nearest step that keeps it whole (the code suggests, it never adjusts)."""
        until = getattr(self.sim, "_until", None)
        if until is not None:
            left = until - solver.t
            n = left / value
            k = max(1, int(round(n)))
            from ..timestep.integrator import is_whole
            if not is_whole(n) or n < 0.5:
                raise OptionError(f"dt = {value:g} does not divide the time left before until = "
                                  f"{until:g} ({left:g}) into whole steps; the nearest one that "
                                  f"does: dt = {left / k:.12g} ({k} steps)",
                                  fix="set a dt that divides the time left (the nearest one is in the "
                                      "message), or run with steps=")
        old = float(solver.dt)
        solver.set_dt(value)                        # IVPSolver.set_dt: Runge-Kutta only
        return {"old": old}

    def _parse_film(self, film: Any) -> dict[str, View]:
        if film is None:
            return {}
        if isinstance(film, (list, tuple)):
            film = {str(s): s for s in film}
        if not isinstance(film, Mapping):
            raise InputTypeError("watch(film=...): a list of field names or {name: view(...)}",
                                 fix='film={"vorticity": cn.view(omega, scale="signed"), '
                                     '"T": cn.view("T")}', doc=_DOC)
        out = {}
        for name, v in film.items():
            if isinstance(v, str) or callable(v) and not isinstance(v, View):
                v = View(v)
            if not isinstance(v, View):
                raise InputTypeError(f"watch(film): {name!r} must be a field name, a function "
                                     "of sim or cn.view(...)", doc=_DOC)
            if isinstance(v.source, str):
                self._check_field(v.source, f"view {name!r}")
            for c in dict(v.at or {}):
                if c not in self._names:
                    raise InputTypeError(f"view {name!r}: at={{{c!r}: ...}} is not a coordinate",
                                         fix="coordinates: " + ", ".join(self._names), doc=_DOC)
            if isinstance(v.source, str):     # dimensions known now: refused here, not per frame
                if v.mode is not None and len(self._names) != 3:
                    raise OptionError(f"view {name!r}: mode={v.mode!r} is a 3D view; this model "
                                      f"has {len(self._names)} coordinate(s)",
                                      fix=f'cn.view("{v.source}") (a 2D film)', doc=_DOC)
                if v.mode is None and len(self._names) - len(v.at or {}) != 2:
                    left = len(self._names) - len(v.at or {})
                    raise OptionError(f"view {name!r}: a film is 2D; {v.source!r} with "
                                      f"at={v.at or {}} has {left} dimension(s)",
                                      fix=('slice one coordinate: at={"' + self._names[-1]
                                           + '": ...}') if left == 3 else
                                      "a profile: cn.profile(...) in stats=", doc=_DOC)
            out[str(name)] = v
        return out

    def _check_names(self) -> None:
        """View and statistic names: spool-file safe, not a built-in number (t, iteration, dt,
        ...), and one namespace for both (frames and curves share it)."""
        from ..telemetry.sink import BUILTIN_SCALARS, check_name
        for n in [*self.views, *self.stats]:
            try:
                check_name(n, "view/statistic")
            except ValueError as err:
                raise InputTypeError(str(err), doc=_DOC) from None
        for n in self.stats:
            if n in BUILTIN_SCALARS:
                raise InputTypeError(f"watch(stats): {n!r} is always sent by the run (built-in: "
                                     f"{', '.join(BUILTIN_SCALARS)}); a statistic of that name "
                                     "would replace it", fix=f'stats={{"my_{n}": ...}}', doc=_DOC)
        both = sorted(set(self.views) & set(self.stats))
        if both:
            raise InputTypeError(f"watch(): {both} name both a view and a statistic",
                                 why="frames and curves of a run share their names",
                                 fix="rename one of them", doc=_DOC)

    def _parse_stats(self, stats: Mapping[str, Any] | None) -> dict[str, Stat]:
        out = {}
        for name, s in dict(stats or {}).items():
            if isinstance(s, str):
                raise InputTypeError(
                    f"watch(stats): {name!r} = {s!r} is a name, not a declared quantity",
                    why="a monitor never computes physics you did not declare",
                    fix=f'stats={{"{name}": lambda sim: ...}} (a number), or cn.spectrum("{s}", '
                        f'axis=...) / cn.profile("{s}", along=...)', doc=_DOC)
            if callable(s) and not isinstance(s, Stat):
                s = series(s)
            if not isinstance(s, Stat):
                raise InputTypeError(f"watch(stats): {name!r} must be a function of sim, "
                                     "cn.series, cn.spectrum or cn.profile", doc=_DOC)
            if s.kind != "series":
                self._check_field(str(s.source), f"stat {name!r}")
                if s.axis not in self._names:
                    raise InputTypeError(f"stat {name!r}: {s.axis!r} is not a coordinate",
                                         fix="coordinates: " + ", ".join(self._names), doc=_DOC)
                if s.kind == "spectrum" and not self._comp_basis(s.axis).periodic:
                    raise OptionError(f"stat {name!r}: spectrum along {s.axis!r}, which is not "
                                      "periodic", why="E(k) is defined here by the Fourier "
                                      "series of a periodic coordinate",
                                      fix=f'cn.profile("{s.source}", along="{s.axis}")', doc=_DOC)
            out[str(name)] = s
        return out

    def _comp_basis(self, c: str):
        return self.sim.compiled.bases[c]

    def _check_field(self, src: str, what: str) -> None:
        base, _, comp = src.partition(".")
        if base not in self._unknowns:
            raise InputTypeError(f"{what}: {base!r} is not an unknown of the model",
                                 fix="unknowns: " + ", ".join(self._unknowns) +
                                     "; or give a function of sim", doc=_DOC)
        rank = self._unknowns[base]
        if rank == 0 and comp:
            raise InputTypeError(f"{what}: {base!r} is a scalar (no component {comp!r})", doc=_DOC)
        if rank == 1 and comp not in self._names:
            raise InputTypeError(f"{what}: {base!r} is a vector; name a component",
                                 fix=", ".join(f"'{base}.{c}'" for c in self._names), doc=_DOC)
        if rank > 1:
            raise InputTypeError(f"{what}: tensors are shown through a function of sim", doc=_DOC)

    # ------------------------------------------------------------------ values
    def values(self, src: str | Callable[[Any], Any]) -> np.ndarray:
        """Grid values of a declared source (host array, one axis per coordinate)."""
        if callable(src):
            a = src(self.sim)
            return np.asarray(a.get() if hasattr(a, "get") and not isinstance(a, np.ndarray) else a,
                              dtype=float)
        base, _, comp = src.partition(".")
        g = np.asarray(self.sim.grid(base), dtype=float)
        return g[self._names.index(comp)] if comp else g

    def _frame_fn(self, v: View) -> Callable[[Any], Any]:
        def f(_solver: Any) -> np.ndarray:
            a = self.values(v.source)
            if v.mode is not None:
                if a.ndim != 3:
                    raise OptionError(f"3D view: the field has {a.ndim} dimension(s), not 3",
                                      fix="a 3D view (mode=...) needs a 3D field; use a film "
                                          "for a 2D field", doc="docs/site/guide/studio.md")
                return a
            if v.at:
                for c, x0 in v.at.items():
                    k = self._names.index(c) - (len(self._names) - a.ndim)
                    j = int(np.argmin(np.abs(self._nodes[c] - float(x0))))
                    a = np.take(a, j, axis=k)
            if a.ndim == 1:
                raise OptionError("a film needs 2D values; this field is 1D",
                                  fix="watch it as a profile, or build a space-time diagram from "
                                      "a function of sim", doc="docs/site/guide/studio.md")
            if a.ndim != 2:
                raise OptionError(f"a film needs 2D values, got {a.ndim}D",
                                  fix="slice it with at={'z': 0.5} (one coordinate per dropped "
                                      "dimension)", doc="docs/site/guide/studio.md")
            return self._uniform(a, v)
        return f

    def _plane(self, v: View) -> list[str]:
        """The two coordinates of a 2D film (a 2D model, or a 3D one sliced with ``at=``); [] for
        a function of sim on a 1D or 3D model (its own 2D array, shown in index coordinates)."""
        if len(self._names) - len(v.at or {}) == 2:
            return [c for c in self._names if c not in (v.at or {})]
        return []

    def _uniform(self, a: np.ndarray, v: View) -> np.ndarray:
        """Linear resampling of non-uniform node axes (Chebyshev) onto as many uniform points
        between the first and last node, so the picture is not distorted (declared). On a 2D
        plane the values must sit on the grid nodes (a function of sim included)."""
        plane = self._plane(v)
        if not plane:
            return a
        want = tuple(len(self._nodes[c]) for c in plane)
        if a.shape != want:
            raise OptionError(f"values of shape {a.shape}, not on the grid nodes {want} of "
                              f"({', '.join(plane)})",
                              fix="return grid values, e.g. from sim.grid(...)", doc="docs/site/guide/studio.md")
        for ax, c in enumerate(plane):
            x = self._nodes[c]
            if _is_uniform(x):
                continue
            xu = np.linspace(x[0], x[-1], x.size)
            a = np.apply_along_axis(lambda col, x=x, xu=xu: np.interp(xu, x, col), ax, a)
        return a

    def _frame_opts(self, v: View) -> dict:
        plane = self._plane(v)
        opts: dict[str, Any] = {"scale": v.scale}
        if plane:
            opts["x"], opts["y"] = self._nodes[plane[0]], self._nodes[plane[1]]
            bad = [c for c in plane if not _is_uniform(self._nodes[c])]
            if bad:
                opts["resample"] = (f"linear interpolation of the nodes along {', '.join(bad)} onto "
                                    "uniform points between the first and last node")
        return opts

    def _frame3d_opts(self, v: View) -> dict:
        if isinstance(v.source, str) and len(self._names) == 3:
            return {"extent": [[float(self._nodes[c][0]), float(self._nodes[c][-1])]
                               for c in self._names],
                    "scale": v.scale, "axes": tuple(self._names),
                    "coords": [self._nodes[c] for c in self._names]}
        return {"extent": [[0.0, 1.0]] * 3, "scale": v.scale}

    def _curve_fn(self, s: Stat) -> Callable[[Any], tuple[np.ndarray, np.ndarray]]:
        names = self._names

        def f(_solver: Any) -> tuple[np.ndarray, np.ndarray]:
            a = self.values(s.source)
            ax = names.index(s.axis) - (len(names) - a.ndim)
            if s.kind == "profile":
                y = a.mean(axis=tuple(i for i in range(a.ndim) if i != ax)) if a.ndim > 1 else a
                return self._nodes[s.axis], y
            return power_spectrum(a, ax, self._period(s.axis))
        return f

    def _curve_opts(self, s: Stat) -> dict:
        if s.kind == "spectrum":
            return {"xlabel": f"k_{s.axis}", "ylabel": f"E({s.source})", "logx": True, "logy": True}
        return {"xlabel": str(s.axis), "ylabel": f"<{s.source}>", "logx": False, "logy": False}

    def _period(self, c: str) -> float:
        a, b = self._geo._axes[c].bounds
        return float(b - a)

    # ------------------------------------------------------------------ declarations
    def _declare(self) -> tuple[list[dict], list[dict]]:
        views = []
        for k, (n, v) in enumerate(self.views.items()):
            d = {"name": n, "kind": "film2d" if v.mode is None else "volume3d",
                 "scale": v.scale, "label": v.label or n, "default": k == 0}
            if v.mode is not None:
                d["mode"] = v.mode
            if isinstance(v.source, str):
                d["field"] = v.source
            if v.at:
                d["at"] = {c: float(self._nodes[c][int(np.argmin(np.abs(self._nodes[c] - x)))])
                           for c, x in v.at.items()}
            views.append(d)
        stats = []
        for n, s in self.stats.items():
            d = {"name": n, "kind": s.kind, "label": s.label or n, "logy": s.logy,
                 "logx": s.kind == "spectrum"}
            if s.kind != "series":
                d["field"], d["axis"] = s.source, s.axis
            stats.append(d)
        return views, stats

    def describe(self) -> str:
        """What is watched and how (the part of :meth:`explain` written in the run's header)."""
        lines = [f"Watch '{self.run_id}': the callback reads the state every {self.poll} step(s) "
                 f"(never changes it); frames at most every {self.every:g} s of wall time per "
                 f"name, declared numbers at most every {self.scalars_every:g} s; t, iteration, "
                 "dt, wall time per step are always sent."]
        for k, (n, v) in enumerate(self.views.items()):
            src = v.source if isinstance(v.source, str) else f"{_fname(v.source)}(sim)"
            what = (f"3D view ({v.mode}) of {src}" if v.mode else f"film of {src}")
            if v.at:
                at = ", ".join(f"{c} = {self._nodes[c][int(np.argmin(np.abs(self._nodes[c] - x)))]:.6g}"
                               f" (node nearest {x:g})" for c, x in v.at.items())
                what += f" at {at}"
            lines.append(f"  view '{n}'{' (default)' if k == 0 else ''}: {what}, scale {v.scale}")
            if v.mode is None and self._plane(v):
                grid = [len(self._nodes[c]) for c in self._plane(v)]
                sent = [-(-n // -(-n // self.max_px)) for n in grid]       # ceil(n / ceil(n / max_px))
                lines.append(f"    sent {'x'.join(map(str, sent))} of the {'x'.join(map(str, grid))} "
                             f"grid nodes (stride {'x'.join(str(-(-n // self.max_px)) for n in grid)}, "
                             f"max_px={self.max_px}), 8 bits: {sent[0] * sent[1] / 1024:.0f} KiB "
                             "per frame before deflate")
                bad = [c for c in self._plane(v) if not _is_uniform(self._nodes[c])]
                if bad:
                    lines.append(f"    non-uniform nodes along {', '.join(bad)}: linearly "
                                 "resampled onto uniform points (declared in each frame)")
            elif v.mode is None:
                lines.append("    a function of sim on a non-2D model: shown in array-index "
                             "coordinates")
        for n, s in self.stats.items():
            if s.kind == "series":
                lines.append(f"  number '{n}': {_fname(s.source)}(sim)")
            elif s.kind == "spectrum":
                lines.append(f"  curve '{n}': power spectrum of {s.source} along {s.axis}, "
                             "E(k) = c_k |rfft(f)_k / N|^2 (c_k = 2 but at k = 0 and Nyquist), "
                             "averaged over the other coordinates; k = 0 not drawn")
            else:
                lines.append(f"  curve '{n}': mean of {s.source} over the coordinates other "
                             f"than {s.axis}, at the grid nodes of {s.axis}")
        lines.append(f"  film: up to {self.film_mb:g} MiB, thinned geometrically (every 4^level-th "
                     "frame; the level goes up each time half of the remaining budget is used)")
        lines.append(f"  monitor commands, read at most every {self.controls_every:g} s: pause/resume "
                     f"(the run waits), checkpoint/fork (to {self.checkpoint_dir}), stop (raises "
                     "StopRun out of sim.run)")
        for n, (lo, hi) in self.settable.items():
            lines.append(f"  settable by a monitor: {n} in [{lo:g}, {hi:g}] (command 'set', "
                         "applied between two steps, recorded as an event param_set; with "
                         "run(until=T) the time left must stay a whole number of steps, else "
                         "refused with the nearest valid value; with run(steps=n) n is kept)")
        return "\n".join(lines)

    def explain(self) -> str:
        """Everything that is sent, how often, and what is lost on the way (telemetry included)."""
        return self.describe() + "\n  " + self.tel.explain()

    # ------------------------------------------------------------------ running
    def callback(self, solver: Any) -> None:
        """The callback the simulation calls every ``poll`` steps (telemetry + monitor commands)."""
        self._cb(solver)

    def close(self) -> None:
        """Send the final status and stop the telemetry writer (idempotent)."""
        if not getattr(self, "closed", True):
            self.closed = True
            self.tel.close()


def power_spectrum(a: np.ndarray, axis: int, length: float) -> tuple[np.ndarray, np.ndarray]:
    """(wavenumbers 2 pi k / L for k >= 1, E(k)) of :func:`spectrum` (one-sided, Parseval:
    ``E.sum() + E(0) == mean(a**2)``)."""
    n = a.shape[axis]
    f = np.fft.rfft(a, axis=axis) / n
    p = np.abs(f) ** 2
    p = np.moveaxis(p, axis, -1)
    p[..., 1:] *= 2.0
    if n % 2 == 0:
        p[..., -1] /= 2.0                           # the Nyquist mode appears once
    e = p.reshape(-1, p.shape[-1]).mean(axis=0)
    k = 2 * math.pi * np.arange(e.size) / length
    return k[1:], e[1:]


def _nodes(comp: Any) -> dict[str, np.ndarray]:
    """Grid nodes of each coordinate (non-dealiased grid: the one ``sim.grid`` returns).

    A sine/cosine axis keeps the N + 1 nodes of [a, b] that ``sim.grid`` returns, not the 2N
    nodes of the doubled period (films were refused and profiles
    paired 2N nodes with N + 1 values)."""
    from ..reference.transforms import grid as _grid
    geo = comp.model.geometry
    out = {}
    for c in geo.coord_names:
        try:
            out[c] = np.asarray(_grid(comp.bases[c], dealias=False), dtype=float).reshape(-1)
        except NotImplementedError:             # no reference grid (curvilinear): index positions
            out[c] = np.arange(comp.N[c], dtype=float)
        if c == geo.sincos_coord:
            out[c] = out[c][: comp.N[c] + 1]
    return out


def _is_uniform(x: np.ndarray) -> bool:
    d = np.diff(x)
    return x.size < 3 or bool(np.allclose(d, d[0], rtol=1e-9, atol=0))


def _on_sim(sim: Any, f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda _solver: f(sim)


def _fname(f: Any) -> str:
    return getattr(f, "__name__", type(f).__name__)


def _plain(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)
