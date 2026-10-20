"""Lagrangian tracer particles of a native simulation (``sim.particles``).

Passive tracers dx/dt = u(x, t) advected by the spectral velocity of the simulation.

* **Velocity at a particle**: the spectral series of the velocity evaluated exactly at the
  particle position (sum over every resolved mode of the basis functions at that point:
  cos/sin along periodic axes, Chebyshev polynomials T_n along the wall axis), never
  a low-order interpolation of grid values. Cost: P x (number of modes) per evaluation, done as
  one contraction per axis in chunks of particles (CPU NumPy or GPU CuPy, the backend of the
  simulation).
* **Time integration**: one classical RK4 step per flow step. RK4 needs the velocity at
  t_n, t_n + dt/2 and t_n+1: the two ends are the flow states before and after the step (exact
  in space); the midpoint is the Lagrange interpolant in time through the last four flow states
  of the run (t_n-2 .. t_n+1; two and three states on the first two steps of a run), formed on
  the coefficients (the evaluation is linear in them). Order: 4 in a steady flow (the RK4
  error). In an unsteady flow the tracers see the flow states u_n, which carry the time error of
  the flow scheme (RK443: third order), and the first step of each run interpolates the midpoint
  linearly between two states (one local error of order dt^3; the second step, three states,
  dt^4): order 3 with RK443. A run split into several ``sim.run`` calls continues the same
  history (it is reset only when the state changed between the calls). In decaying Taylor-Green both parts are of order 3; with the start
  given exact past states the error drops (3.4-4.7x) but the order stays 3, that of the flow
  (tests/model/test_particles_dye.py). The flow stepper is unchanged.
* **Periodic axes**: positions are kept unwrapped (the dispersion is read from them) and also
  written wrapped into [a, a + L).
* **Walls** (``at_walls``): a step that ends outside [a, b] on the wall axis is refused
  (``"refuse"``, a ``NumericalError`` naming the particle) or the particle is put back on the
  wall it crossed and counted (``"clip"``; counts per particle in the file and in explain()).
  RK4 stage positions are not clipped: the velocity there is the polynomial series itself (its
  continuation just outside the interval). A particle on a no-slip wall stays on it.
* **Files**: an HDF5 analysis file of the self-describing layout (``cn.io.read(path)``,
  ``.to_xarray()``): ``position`` and ``position_unwrapped`` (t, particle, component),
  ``velocity`` (t, particle, component) and ``wall_clips`` (t, particle).

Not covered, refused with the reason: curved or mapped geometries, sine/cosine walls, half lines
and lines (Laguerre, Hermite), compound intervals, complex-valued problems. The particle positions
are not part of the simulation's checkpoints: a restarted run declares new particles (from the
last row of the file if it should continue them).

This is the **in-simulation** interface; :class:`conatus.diagnostics.Particles` is the standalone
one (any velocity, caller's dt and scheme, boundary "clip"/"raise", no file). The two contracts
are compared in :mod:`conatus.diagnostics.lagrangian` (finding 63).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import InputTypeError, NumericalError, OptionError, UnsupportedError
from ..io import _describe as dsc
from ..io._h5 import json_attr, open_h5
from ..io.analysis import AnalysisFile

__all__ = ["SpectralProbe", "Tracers", "ParticleFile"]

_DOC = "docs/site/guide/particles.md"
#: Particles evaluated together (memory of one contraction: components x modes x chunk).
CHUNK = 4096


class SpectralProbe:
    """Exact evaluation of one unknown's spectral series at arbitrary points of a Cartesian box
    (RealFourier, Chebyshev T). ``points``: (P, d) in coordinate order."""

    def __init__(self, sim: Any, name: str) -> None:
        from ..ir.bases import Chebyshev, RealFourier
        from .geometry import Box
        comp = sim.compiled
        geo = comp.model.geometry
        if not isinstance(geo, Box) or type(geo).__name__ == "MappedBox":
            raise UnsupportedError(f"particles on {geo!r}",
                                   why="the exact evaluation at a point is written for the "
                                       "Cartesian box (Fourier and Chebyshev series)",
                                   fix="a cn.Box geometry", doc=_DOC)
        if comp.symmetry is not None or geo.interfaces:
            raise UnsupportedError(f"particles on {geo!r}: sine/cosine walls or a compound "
                                   "interval", fix="walls(a, b) with the Chebyshev basis",
                                   doc=_DOC)
        if getattr(sim.solver.plan.layout, "complex", False):
            raise UnsupportedError("particles in a complex-valued problem (a periodic(..., "
                                   "complex=True) axis or complex data)",
                                   why="a tracer moves with a real velocity; the real part of a "
                                       "complex field is not taken for you",
                                   fix="periodic(L) (the real cos/sin series) for a real flow",
                                   doc=_DOC)
        self.sim, self.name = sim, name
        self.coords = list(geo.coord_names)
        self.axes = []
        for c in self.coords:
            b = comp.bases[c]
            if isinstance(b, RealFourier):
                kind = "real_fourier"
            elif isinstance(b, Chebyshev) and getattr(b, "map", None) is None:
                kind = "chebyshev"
            else:
                raise UnsupportedError(f"particles: basis {b!r} along {c}",
                                       why="exact point evaluation covers RealFourier and "
                                           "unmapped Chebyshev",
                                       fix="periodic(...) or walls(a, b) axes", doc=_DOC)
            self.axes.append((kind, float(b.bounds[0]), float(b.length), int(b.size)))
        sv = sim.solver
        self.xp = sv.xp
        names = [v.name for v in sv.system.variables]
        self.index = names.index(name)
        self.rank = comp.fields[name].rank

    def explain(self) -> str:
        """What is evaluated and how: the unknown, each axis's series and its size."""
        axes = ", ".join(f"{c}: {kind.replace('_', ' ')} N={N} on [{a:g}, {a + L:g}]"
                         for c, (kind, a, L, N) in zip(self.coords, self.axes, strict=True))
        return (f"SpectralProbe of {self.name!r} (rank {self.rank}): exact evaluation of its "
                f"spectral series at arbitrary points, from the coefficients of the state "
                f"(no grid interpolation); {axes}")

    __str__ = explain

    def coefficients(self):
        """Coefficient array of the unknown in the current state (device array on the GPU)."""
        sv = self.sim.solver
        c = sv.system.scatter(sv.X, only=frozenset({self.name}))[self.index]
        return c.copy()

    def _basis(self, axis: int, x):
        """Basis functions of axis ``axis`` at the points x (P,): (P, N)."""
        xp = self.xp
        kind, a, L, N = self.axes[axis]
        if kind == "chebyshev":
            t = 2.0 * (x - a) / L - 1.0
            T = xp.empty((x.shape[0], N), dtype=float)
            T[:, 0] = 1.0
            if N > 1:
                T[:, 1] = t
            for n in range(2, N):                      # T_n+1 = 2 t T_n - T_n-1 (any t)
                T[:, n] = 2.0 * t * T[:, n - 1] - T[:, n - 2]
            return T
        k = (2.0 * np.pi / L) * xp.arange(N // 2)       # RealFourier: pairs (cos, -sin)
        ph = (x - a)[:, None] * k[None, :]
        E = xp.empty((x.shape[0], N), dtype=float)
        E[:, 0::2] = xp.cos(ph)
        E[:, 1::2] = -xp.sin(ph)                       # the (-sin) coefficient convention
        return E

    def evaluate(self, coeffs, points):
        """Values (components, P) of the series with coefficients ``coeffs`` (the layout of
        :meth:`coefficients`, components first) at ``points`` (P, d)."""
        xp = self.xp
        c = coeffs if self.rank else coeffs[None]
        ncomp, d = c.shape[0], len(self.axes)
        P = points.shape[0]
        out = xp.empty((ncomp, P), dtype=float)
        for p0 in range(0, P, CHUNK):
            pts = points[p0:p0 + CHUNK]
            # contract the last axis first: (ncomp, N1..N_d-1, p), then each earlier axis
            acc = xp.tensordot(c, self._basis(d - 1, pts[:, d - 1]), axes=([d], [1]))
            for ax in range(d - 2, -1, -1):
                E = self._basis(ax, pts[:, ax])        # (p, N_ax)
                acc = (acc * _bshape(acc, E, ax)).sum(axis=ax + 1)
            out[:, p0:p0 + CHUNK] = acc
        return out if self.rank else out[0]


def _lagrange(hist: list, tau: float):
    """Coefficients at time ``tau`` of the Lagrange interpolant through ``hist`` [(t_j, c_j)]."""
    out = None
    for j, (tj, cj) in enumerate(hist):
        w = 1.0
        for m, (tm, _c) in enumerate(hist):
            if m != j:
                w *= (tau - tm) / (tj - tm)
        out = w * cj if out is None else out + w * cj
    return out


def _bshape(acc, E, ax):
    """E (p, N_ax) laid out to broadcast against acc (ncomp, N_0..N_ax, p)."""
    Et = E.T                                           # (N_ax, p)
    shape = [1] * acc.ndim
    shape[ax + 1] = Et.shape[0]
    shape[-1] = Et.shape[1]
    return Et.reshape(shape)


class Tracers:
    """Tracer particles advected during :meth:`Simulation.run` (see the module docstring)."""

    def __init__(self, sim: Any, positions: Any, *, path: str | Path | None, every: int | None,
                 velocity: str, at_walls: str, mode: str | None, initial: bool) -> None:
        from ..io._h5 import positive
        if at_walls not in ("refuse", "clip"):
            from ..errors import did_you_mean
            raise OptionError(f"particles(at_walls={at_walls!r})",
                              fix="at_walls='refuse' (a step ending outside the walls raises) "
                                  "or 'clip' (put back on the wall, counted)"
                                  + did_you_mean(str(at_walls), ("refuse", "clip")), doc=_DOC)
        sim._check_unknown(velocity, "particles(velocity=...)")
        if sim.compiled.fields[velocity].rank != 1:
            raise InputTypeError(f"particles(velocity={velocity!r}): a vector unknown",
                                 doc=_DOC)
        self.sim, self.velocity_name, self.at_walls = sim, velocity, at_walls
        self.probe = SpectralProbe(sim, velocity)
        self.coords = self.probe.coords
        xp = self.xp = self.probe.xp
        X = self._positions(positions)
        self.P, self.d = X.shape
        self._x = xp.asarray(X)                       # unwrapped positions (P, d)
        self.clips = xp.zeros(self.P, dtype=np.int64)
        self.steps_taken = 0
        self._hist: list = []                         # (t, coefficients) of this run, last 4
        self.every = 1                                # stepped after every flow step
        self.file = None
        if (path is None) != (every is None):
            raise OptionError("particles(path=..., every=...): both or neither",
                              why="the file is written every `every` steps; without a file the "
                                  "positions are read with particles.positions()", doc=_DOC)
        if path is not None:
            self.write_every = positive(every, "particles()", "every")
            self.file = ParticleFile(sim, self, path, every=self.write_every, mode=mode,
                                     initial=initial)
        self.begin_run()

    # ---------------------------------------------------------------- input
    def _positions(self, positions: Any) -> np.ndarray:
        d = len(self.coords)
        if isinstance(positions, Mapping):
            miss = [c for c in self.coords if c not in positions]
            extra = [k for k in positions if k not in self.coords]
            if miss or extra:
                raise InputTypeError("particles(positions={...}): one array per coordinate "
                                     f"{tuple(self.coords)}" + (f"; missing {miss}" if miss else "")
                                     + (f"; not coordinates {extra}" if extra else ""), doc=_DOC)
            cols = [np.asarray(positions[c], float).ravel() for c in self.coords]
            if len({c.size for c in cols}) != 1:
                raise InputTypeError("particles(positions={...}): arrays of one length",
                                     doc=_DOC)
            X = np.stack(cols, axis=1)
        else:
            X = np.asarray(positions.get() if hasattr(positions, "get")
                           and not isinstance(positions, np.ndarray) else positions, float)
            if X.ndim == 1 and d == 1:
                X = X[:, None]
            if X.ndim != 2 or X.shape[1] != d or X.shape[0] < 1:
                raise InputTypeError(f"particles(positions=<shape {X.shape}>): (P, {d}) in the "
                                     f"order {tuple(self.coords)}, or a dict of arrays", doc=_DOC)
        if not np.all(np.isfinite(X)):
            raise InputTypeError("particles(positions=...): non-finite positions", doc=_DOC)
        for ax, (kind, a, L, _N) in enumerate(self.probe.axes):
            if kind == "chebyshev":
                bad = np.nonzero((X[:, ax] < a) | (X[:, ax] > a + L))[0]
                if bad.size:
                    raise InputTypeError(
                        f"particles: {bad.size} position(s) outside the walls [{a:g}, {a + L:g}] "
                        f"of {self.coords[ax]} (first: particle {int(bad[0])} at "
                        f"{X[bad[0], ax]:g})", fix="positions inside the box", doc=_DOC)
        return X

    # ---------------------------------------------------------------- stepping
    def begin_run(self) -> None:
        """The velocity at the start of a run (``Simulation.run`` calls it). The past flow
        states are kept when the run continues the last one (same time, same velocity
        coefficients): splitting a run into one-step calls used to reset them every call and
        drop the convergence order from 3 to 2 (finding 14). They are
        dropped only when the state really changed (``sim.initial``, ``sim.restart``)."""
        c = self.probe.coefficients()
        if self._hist:
            t_last, c_last = self._hist[-1]
            xp = self.xp
            if t_last == self.sim.t and c_last.shape == c.shape and bool(xp.array_equal(c_last, c)):
                return
        self._hist = [(self.sim.t, c)]

    def _vel(self, c, x):
        return self.probe.evaluate(c, x).T          # (P, d)

    def __call__(self, solver: Any) -> None:         # callback of every flow step
        xp = self.xp
        c1 = self.probe.coefficients()
        t0, c0 = self._hist[-1]
        h = self.sim.t - t0
        self._hist = (self._hist + [(self.sim.t, c1)])[-4:]
        cm = _lagrange(self._hist, t0 + 0.5 * h)
        x = self._x
        k1 = self._vel(c0, x)
        k2 = self._vel(cm, x + 0.5 * h * k1)
        k3 = self._vel(cm, x + 0.5 * h * k2)
        k4 = self._vel(c1, x + h * k3)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        for ax, (kind, a, L, _N) in enumerate(self.probe.axes):
            if kind != "chebyshev":
                continue
            lo, hi = x[:, ax] < a, x[:, ax] > a + L
            out = lo | hi
            if bool(out.any()):
                if self.at_walls == "refuse":
                    i = int(xp.nonzero(out)[0][0])
                    v = float(x[i, ax])
                    raise NumericalError(
                        f"particles: {int(out.sum())} particle(s) left [{a:g}, {a + L:g}] along "
                        f"{self.coords[ax]} at t = {self.sim.t:g} (first: particle {i} at {v:.6g})",
                        why="the time step moved a tracer through a wall (O(dt^p) error near a "
                            "wall, or a velocity that crosses it)",
                        fix="a smaller dt, or particles(..., at_walls='clip') to put such "
                            "particles back on the wall (counted, declared)", doc=_DOC)
                x[:, ax] = xp.where(lo, a, xp.where(hi, a + L, x[:, ax]))
                self.clips += out.astype(np.int64)
        self._x = x
        self.steps_taken += 1
        if self.file is not None and self.sim.iteration % self.write_every == 0:
            self.file.write()

    # ---------------------------------------------------------------- output
    @property
    def initial_pending(self) -> bool:
        """True while the start row of the file (``initial=True``) is still to be written."""
        return self.file is not None and self.file.initial_pending

    def write_initial(self) -> None:
        """The row of the start (``initial=True``)."""
        if self.file is not None:
            self.file.write_initial()

    def _host(self, a) -> np.ndarray:
        return np.asarray(a if self.xp is np else self.xp.asnumpy(a))

    def positions(self, unwrapped: bool = False) -> np.ndarray:
        """Positions (P, d) in coordinate order: wrapped into [a, a + L) along periodic axes, or
        ``unwrapped=True`` (continuous paths, for dispersion)."""
        X = self._host(self._x).copy()
        if not unwrapped:
            for ax, (kind, a, L, _N) in enumerate(self.probe.axes):
                if kind != "chebyshev":
                    X[:, ax] = a + np.mod(X[:, ax] - a, L)
        return X

    def velocities(self) -> np.ndarray:
        """Velocity (P, d) at the current positions (exact series of the current state)."""
        return self._host(self._vel(self.probe.coefficients(), self._x))

    def wall_clips(self) -> np.ndarray:
        """Number of times each particle was put back on a wall (``at_walls='clip'``)."""
        return self._host(self.clips).copy()

    def explain(self) -> str:
        """What is integrated, how, and what was written."""
        kinds = {"real_fourier": "cos/sin series", "chebyshev": "Chebyshev T series"}
        axes = ", ".join(f"{c}: {kinds[k]} ({N} modes)" for c, (k, _a, _L, N)
                         in zip(self.coords, self.probe.axes, strict=True))
        f = (f"file {self.file.path} every {self.write_every} steps ({self.file.writes} row(s))"
             if self.file is not None else "no file (particles.positions())")
        n_clip = int(self._host(self.clips).sum())
        lines = [f"Particles: {self.P} passive tracer(s) advected by {self.velocity_name}, "
                 f"backend {self.sim.solver.backend}, {self.steps_taken} step(s) taken; {f}",
                 f"  velocity at a particle: exact evaluation of the spectral series ({axes})",
                 "  time integration: classical RK4 per flow step; the midpoint velocity is the "
                 "cubic interpolant in time of the last four flow states (2 and 3 states on the "
                 "first two steps of a run): order 4 in a steady flow, 3 in an unsteady flow "
                 "(the time error of the flow states themselves, third order for RK443, and the "
                 "linear midpoint of the first step of a run)",
                 "  periodic axes: positions unwrapped (written also wrapped into [a, a + L))",
                 f"  walls: at_walls={self.at_walls!r}"
                 + (f" ({n_clip} clip(s) so far)" if self.at_walls == "clip" else
                    " (a step ending outside the walls raises)")
                 + "; stage positions are not clipped (the series continues there)",
                 "  not in the simulation's checkpoints: after a restart, declare the particles "
                 "again (e.g. from the last row of the file)"]
        return "\n".join(lines)

    __str__ = explain


class ParticleFile(AnalysisFile):
    """HDF5 analysis file of particle data (``sim.particles(path, ...)``): the self-describing
    layout of ``cn.io`` with dimensions (t, particle, component) instead of grid coordinates."""

    def __init__(self, sim: Any, particles: Tracers, path: str | Path, *, every: int,
                 mode: str | None, initial: bool) -> None:
        self.particles = particles
        tasks = {"position": lambda s: particles.positions(),
                 "position_unwrapped": lambda s: particles.positions(unwrapped=True),
                 "velocity": lambda s: particles.velocities(),
                 "wall_clips": lambda s: particles.wall_clips()}
        super().__init__(sim, path, tasks, every=every, mode=mode, initial=initial)

    def _scales(self) -> dict:
        return {}                                     # no grid: the particle index instead

    def _create(self) -> None:
        super()._create()
        p = self.particles
        with open_h5(self.path, "a") as f:
            sc = f["scales"]
            dsc.make_scale(sc.create_dataset("particle", data=np.arange(p.P)), "particle")
            comp = sc.create_dataset(dsc.COMPONENT, data=np.array(p.coords, dtype="S"))
            dsc.make_scale(comp, dsc.COMPONENT)
            f.attrs["content"] = "particles"
            f.attrs["particles"] = json_attr({
                "count": p.P, "velocity": p.velocity_name, "at_walls": p.at_walls,
                "time_integration": "RK4 per flow step, midpoint velocity cubic in time "
                                    "(order 4 steady, 3 unsteady)",
                "evaluation": "exact spectral series at the positions"})

    @staticmethod
    def _describe(src: Any) -> str:
        return "particle data (sim.particles)"

    def _label(self, f: Any, dset: Any, name: str, shape: tuple) -> None:
        sc = f["scales"]
        vec = len(shape) == 2
        dims = ["t", "particle"] + ([dsc.COMPONENT] if vec else [])
        scales = {"t": sc["sim_time"], "particle": sc["particle"],
                  dsc.COMPONENT: sc[dsc.COMPONENT]}
        dsc.label(dset, dims, scales, rank=1 if vec else None, coords=list(self.particles.coords))

    def write_xdmf(self) -> None:
        """No XDMF sidecar for particle data (declared in explain())."""
        self.xdmf_note = "no XDMF sidecar: particle data (no grid)"
        return None
