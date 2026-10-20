"""Lagrangian diagnostics: interpolation to points, tracer and inertial particles,
pathlines, streaklines, timelines (numerical smoke wire / hydrogen bubbles), FTLE.

Read-only: particles read a velocity (an analytic callable, or grid snapshots of a run) and never
touch the computation. Everything numerical is declared and shown by ``explain()``:

* **Interpolation** (:class:`Interpolator`): ``"spectral"`` evaluates the Fourier / Chebyshev
  series of the grid values *exactly* at the points (cost O(Np * prod(n)) per component: for
  tests, small grids, few particles); ``"lagrange"`` with ``order=p`` uses a local tensor-product
  Lagrange polynomial through ``p`` nodes per axis (degree p - 1, error O(h^p)); periodic stencils
  wrap around, Chebyshev stencils are shifted inside the node range near a wall (between the wall
  and the first Gauss node the stencil polynomial is *extrapolated*: declared).
* **Time integration** (:class:`Particles`): explicit RK4 (``"rk4"``) or Heun RK2 (``"rk2"``),
  step ``dt`` given by the caller. Grid velocity snapshots (u_n at t_n, u_{n+1} at t_{n+1}) are
  interpolated **linearly in time** between them (2nd order in time, declared); with one snapshot
  the velocity is frozen (declared).
* **Boundaries**: periodic axes wrap (the unwrapped positions are kept, for FTLE and pathlines);
  on a bounded axis a position outside [a, b] is either clipped back (``boundary="clip"``: every
  clipped coordinate is **counted** in ``n_clipped`` and reported) or refused (``"raise"``).
* **Inertial particles** (:class:`MaxeyRiley`): simplified Maxey–Riley equation, Stokes drag
  only, dv/dt = (u(x, t) - v) / tau_p + (1 - beta) g. Neglected, and listed by ``explain()``:
  added mass, fluid acceleration (pressure gradient) term, Basset history force, Faxén
  corrections, Saffman lift, particle-particle and two-way coupling.

Two particle interfaces, two contracts (finding 63): this module is the
**standalone** one: any velocity (an analytic callable, or grid snapshots interpolated by a
declared method), a caller-chosen ``dt`` and scheme (RK4 / RK2), boundaries ``"clip"`` /
``"raise"``, positions in memory. ``sim.particles(...)`` (:mod:`conatus.model.particles`) is the
**in-simulation** one: the exact spectral series of the running simulation, one RK4 step per flow
step with past flow states for the midpoint, walls ``"refuse"`` / ``"clip"``, an HDF5 file. Use
this module for a velocity you own or for data after a run; use ``sim.particles`` while a native
simulation runs. The same split holds for the dye (:mod:`conatus.diagnostics.dye` vs
``sim.dye``).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import ConvergenceError, CoordinateError, OptionError
from .grid import Chebyshev, Grid, Periodic

__all__ = ["Interpolator", "AnalyticVelocity", "GridVelocity", "MaxeyRiley", "Particles",
           "PathRecorder", "Emitter", "Streaklines", "Timelines", "ftle"]


# ====================================================================== interpolation
class Interpolator:
    """Values of grid fields at arbitrary points.

    ``method="spectral"``: exact evaluation of the grid interpolant (Fourier series on periodic
    axes, Chebyshev series on Chebyshev axes). ``method="lagrange"``: local Lagrange polynomial
    through ``order`` nodes per axis (``order`` >= 2 required, no default)."""

    def __init__(self, grid: Grid, method: str, order: int | None = None) -> None:
        if method not in ("spectral", "lagrange"):
            raise OptionError(f"interpolation method {method!r}: 'spectral' or 'lagrange'",
                              fix="declare method='spectral' (exact) or method='lagrange', order=p")
        if method == "lagrange":
            # an integer, never truncated (order=2.7 used to run as 2)
            if order is None or isinstance(order, bool) or not isinstance(order, (int, np.integer)) \
                    or order < 2:
                raise OptionError(f"lagrange interpolation needs an integer order >= 2 (got {order!r})",
                                  fix="declare order=p: p nodes per axis, error O(h^p)")
            for ax in grid.axes:
                if int(order) > ax.n:
                    raise OptionError(f"lagrange order {order} > {ax.n} nodes on axis {ax!r}")
        elif order is not None:
            raise OptionError("order= only applies to method='lagrange' (spectral is exact)")
        self.grid, self.method = grid, method
        self.order = None if order is None else int(order)
        self.xp = grid.xp

    def explain(self) -> str:
        """Method, cost and boundary treatment."""
        if self.method == "spectral":
            return (f"Interpolator(spectral: exact evaluation of the Fourier/Chebyshev series of the "
                    f"grid values; even-n Nyquist mode evaluated as a cosine; cost Np*{int(np.prod(self.grid.shape))} "
                    f"per component; {self.grid.explain()})")
        return (f"Interpolator(lagrange, {self.order} nodes per axis (degree {self.order - 1}, "
                f"error O(h^{self.order})), periodic stencils wrap, Chebyshev stencils shifted inside "
                f"near walls (extrapolation between wall and first node); {self.grid.explain()})")

    def __call__(self, f, pts):
        """Values at ``pts`` (shape (Np, d)) of ``f`` (grid shape, or (ncomp, *grid shape)).
        Returns (Np,) or (ncomp, Np). Bounded coordinates must lie in [a, b]."""
        xp = self.xp
        g = self.grid
        pts = xp.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != g.ndim:
            raise CoordinateError(f"points of shape {pts.shape}: (Np, {g.ndim}) expected")
        f = xp.asarray(f)
        single = f.ndim == g.ndim
        F = f[None] if single else f
        if tuple(F.shape[1:]) != g.shape:
            raise CoordinateError(f"field of shape {f.shape} on a grid of shape {g.shape}")
        if pts.shape[0] == 0:                            # no point: empty result, not an error
            out = xp.zeros((F.shape[0], 0))
            return out[0] if single else out
        n_bad = int(pts.size - xp.count_nonzero(xp.isfinite(pts)))
        if n_bad:                                        # never dropped silently
            raise CoordinateError(f"{n_bad} non-finite point coordinates",
                                  why="a NaN position cannot be interpolated; it usually means "
                                      "the velocity field itself is not finite",
                                  fix="check the flow field (and the particle time step)")
        for i, ax in enumerate(g.axes):
            if isinstance(ax, Chebyshev):
                x = pts[:, i]
                bad = int(xp.count_nonzero((x < ax.a) | (x > ax.b)))
                if bad:
                    raise CoordinateError(f"{bad} points outside [{ax.a:g}, {ax.b:g}] on bounded "
                                          f"axis {i}", fix="apply a boundary policy first "
                                          "(Particles(boundary='clip' | 'raise'))")
        out = self._spectral(F, pts) if self.method == "spectral" else self._lagrange(F, pts)
        return out[0] if single else out

    # ---------------------------------------------------------------- spectral (exact)
    def _basis(self, i: int, x):
        """(Np, n) matrix: value of every basis function of axis i at the coordinates x."""
        xp = self.xp
        ax = self.grid.axes[i]
        if isinstance(ax, Periodic):
            n = ax.n
            k = xp.asarray(2 * np.pi * np.fft.fftfreq(n, 1.0 / n) / ax.L)
            B = xp.exp(1j * (x - ax.a)[:, None] * k[None, :])
            if n % 2 == 0:                      # Nyquist mode: the real interpolant is a cosine
                B[:, n // 2] = xp.cos((x - ax.a) * (np.pi * n / ax.L))
            return B
        t = xp.clip((2 * x - ax.a - ax.b) / (ax.b - ax.a), -1.0, 1.0)
        return xp.cos(xp.arccos(t)[:, None] * xp.arange(ax.n)[None, :])

    def _spectral(self, F, pts):
        xp = self.xp
        g = self.grid
        C = F
        periodic = [i for i, ax in enumerate(g.axes) if isinstance(ax, Periodic)]
        if periodic:
            C = xp.fft.fftn(C, axes=[i + 1 for i in periodic])
            for i in periodic:
                C = C / g.axes[i].n
        for i, ax in enumerate(g.axes):
            if isinstance(ax, Chebyshev):
                C = g.cheb_coeffs(C.real, i + 1, ax) + (1j * g.cheb_coeffs(C.imag, i + 1, ax)
                                                        if xp.iscomplexobj(C) else 0)
        # contract the last grid axis first, keeping the particle index p
        B = self._basis(g.ndim - 1, pts[:, g.ndim - 1])
        T = xp.tensordot(C, B.T, axes=([C.ndim - 1], [0]))          # (ncomp, n0, ..., n_{d-2}, Np)
        for i in range(g.ndim - 2, -1, -1):
            B = self._basis(i, pts[:, i])                              # (Np, n_i)
            T = (T * xp.moveaxis(B, 0, 1).reshape((1,) * (T.ndim - 2) + B.T.shape)).sum(axis=T.ndim - 2)
        out = T                                                        # (ncomp, Np)
        return out if xp.iscomplexobj(F) else out.real

    # ---------------------------------------------------------------- lagrange (order p)
    def _stencil(self, i: int, x):
        """(idx (Np, p) int, w (Np, p)) Lagrange stencil of axis i at coordinates x."""
        xp = self.xp
        ax = self.grid.axes[i]
        p = self.order
        off = xp.arange(p) - (p - 1) // 2               # offsets relative to the base node
        if isinstance(ax, Periodic):
            h = ax.L / ax.n
            s = (x - ax.a) / h
            base = xp.floor(s) if p % 2 == 0 else xp.floor(s + 0.5)
            node = base[:, None] + off[None, :]          # unwrapped node numbers
            pos = node * h                               # node coordinates (relative to a)
            xr = (s * h)[:, None]
            idx = xp.mod(node, ax.n).astype(xp.int64)
        else:
            nodes = self.grid.nodes(i)
            j = xp.searchsorted(nodes, x) - 1            # node just left of x (-1 before the first)
            if p % 2 == 0:                               # p/2 nodes on each side of x
                start = j - (p // 2 - 1)
            else:                                        # centred on the nearest node
                jl, jr = xp.clip(j, 0, ax.n - 1), xp.clip(j + 1, 0, ax.n - 1)
                right = xp.abs(nodes[jr] - x) < xp.abs(x - nodes[jl])
                start = j + right.astype(j.dtype) - (p - 1) // 2
            start = xp.clip(start, 0, ax.n - p)          # shifted inside near a wall (declared)
            idx = (start[:, None] + xp.arange(p)[None, :]).astype(xp.int64)
            pos = nodes[idx]
            xr = x[:, None]
        # Lagrange weights  w_m = prod_{l != m} (x - x_l) / (x_m - x_l)
        w = xp.ones(pos.shape)
        for m in range(p):
            for l_ in range(p):
                if l_ != m:
                    w[:, m] *= (xr[:, 0] - pos[:, l_]) / (pos[:, m] - pos[:, l_])
        return idx, w

    def _lagrange(self, F, pts):
        xp = self.xp
        g = self.grid
        d, p, Np = g.ndim, self.order, pts.shape[0]
        idx, w = zip(*[self._stencil(i, pts[:, i]) for i in range(d)], strict=True)
        # tensor-product gather: (ncomp, Np, p, ..., p)
        index = []
        W = xp.ones((Np,) + (p,) * d)
        for i in range(d):
            shp = [Np] + [1] * d
            shp[i + 1] = p
            index.append(idx[i].reshape(shp))
            W = W * w[i].reshape(shp)
        vals = F[(slice(None),) + tuple(index)]                        # (ncomp, Np, p, ..., p)
        return (vals * W[None]).reshape(F.shape[0], Np, -1).sum(axis=2)


# ====================================================================== velocities
class AnalyticVelocity:
    """Velocity given by a callable ``u(x, t)``: x of shape (Np, d) -> (Np, d)."""

    def __init__(self, fn: Callable[[Any, float], Any], d: int, xp: Any = np) -> None:
        from ..errors import check_count
        self.fn, self.d, self.xp = fn, check_count(d, "d", "AnalyticVelocity"), xp

    def __call__(self, x, t: float):
        """Velocity at the points x (Np, d) at time t."""
        return self.xp.asarray(self.fn(x, t), dtype=float)

    def explain(self) -> str:
        """What the velocity is."""
        return f"AnalyticVelocity({getattr(self.fn, '__name__', 'callable')}, d={self.d}, exact in space and time)"


class GridVelocity:
    """Velocity from grid snapshots of a run, interpolated by a declared :class:`Interpolator`.

    ``set(u, t)`` stores a snapshot (``u``: sequence of d component arrays, or an array of shape
    (d, *grid shape)). With two snapshots the velocity is **linear in time** between them (2nd
    order in time, declared); with one it is frozen; outside [t_n, t_{n+1}] it is refused."""

    def __init__(self, interp: Interpolator) -> None:
        self.interp = interp
        self.grid, self.xp, self.d = interp.grid, interp.xp, interp.grid.ndim
        self._snaps: list[tuple[float, Any]] = []

    def set(self, u, t: float) -> None:
        """Add the snapshot (u at time t); keeps the last two."""
        # a copy: the snapshot must not follow later in-place changes of the caller's array
        # (finding 68: asarray kept a view of a float array)
        U = self.xp.stack([self.xp.asarray(c, dtype=float) for c in u]) if not hasattr(u, "shape") \
            else self.xp.array(u, dtype=float, copy=True)
        if U.shape != (self.d,) + self.grid.shape:
            raise CoordinateError(f"velocity snapshot of shape {U.shape}: ({self.d}, *{self.grid.shape}) expected")
        if self._snaps and t <= self._snaps[-1][0]:
            raise OptionError(f"snapshot time {t} is not after the previous one ({self._snaps[-1][0]})")
        self._snaps = (self._snaps + [(float(t), U)])[-2:]

    def __call__(self, x, t: float):
        """Velocity at the points x (Np, d) at time t, shape (Np, d)."""
        if not self._snaps:
            raise OptionError("GridVelocity: no snapshot set (call set(u, t) first)")
        if len(self._snaps) == 1:
            return self.interp(self._snaps[0][1], x).T
        (t0, U0), (t1, U1) = self._snaps
        # round-off of the particle clock (a sum of dt) grows with |t|: tolerance relative to
        # the time magnitude (1e-12 at t = 229 was refused), clamped weight
        tol = 1e-12 * max(1.0, abs(t0), abs(t1), abs(t1 - t0))
        if t < t0 - tol or t > t1 + tol:
            raise OptionError(f"velocity requested at t = {t}, outside the snapshots [{t0}, {t1}]",
                              why="no extrapolation in time", fix="set the next snapshot first")
        a = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        return ((1 - a) * self.interp(U0, x) + a * self.interp(U1, x)).T

    def explain(self) -> str:
        """Snapshots held and the time interpolation."""
        times = [s[0] for s in self._snaps]
        mode = "frozen" if len(times) < 2 else "linear in time between snapshots (2nd order)"
        return f"GridVelocity({mode}, snapshots at t = {times}; {self.interp.explain()})"


class MaxeyRiley:
    """Simplified Maxey–Riley model: dv/dt = (u(x, t) - v) / tau_p + (1 - beta) g.

    ``tau_p`` > 0: Stokes response time; ``beta`` = rho_f / rho_p and ``g`` (vector) are optional
    together (settling). Neglected terms are listed by :meth:`explain`."""

    NEGLECTED = ("added mass", "fluid-acceleration (pressure-gradient) term", "Basset history force",
                 "Faxén corrections", "Saffman lift", "particle-particle collisions",
                 "two-way coupling (the flow does not feel the particles)")

    def __init__(self, tau_p: float, *, beta: float | None = None, g: Sequence[float] | None = None) -> None:
        if not tau_p > 0:
            raise OptionError(f"tau_p = {tau_p}: a positive response time is required")
        if (beta is None) != (g is None):
            raise OptionError("declare beta and g together (settling (1 - beta) g) or neither")
        self.tau_p, self.beta = float(tau_p), beta
        self.g = None if g is None else np.asarray(g, dtype=float)

    def explain(self) -> str:
        """Equation solved and the terms neglected."""
        grav = "" if self.g is None else f" + (1 - {self.beta:g}) g, g = {self.g.tolist()}"
        return (f"MaxeyRiley(dv/dt = (u(x, t) - v) / {self.tau_p:g}{grav}; neglected: "
                + ", ".join(self.NEGLECTED) + ")")


# ====================================================================== integration core
def _bounds_policy(grid_axes, x, boundary: str, counter: list, xp):
    """Apply the boundary policy to positions x (Np, d); returns the positions to evaluate
    the velocity at (periodic coordinates wrapped, bounded ones clipped or refused)."""
    y = x.copy()
    for i, ax in enumerate(grid_axes):
        if isinstance(ax, Periodic):
            y[:, i] = ax.a + xp.mod(x[:, i] - ax.a, ax.L)
        else:
            out = (x[:, i] < ax.a) | (x[:, i] > ax.b)
            n = int(xp.count_nonzero(out))
            if n:
                if boundary == "raise":
                    raise CoordinateError(f"{n} particles left [{ax.a:g}, {ax.b:g}] on bounded axis {i}",
                                          fix="smaller dt, or boundary='clip' (counted)")
                counter[0] += n
                y[:, i] = xp.clip(x[:, i], ax.a, ax.b)
    return y


class Particles:
    """A set of tracer (``inertia=None``) or inertial particles advected by a velocity.

    ``x0`` (Np, d) initial positions; ``velocity``: :class:`AnalyticVelocity`,
    :class:`GridVelocity` or any callable (x, t) -> (Np, d); ``axes``: the domain axes
    (``grid.axes``; periodic axes wrap, bounded ones follow ``boundary``); ``scheme`` 'rk4' or
    'rk2'; ``v0`` for inertial particles: 'fluid' (v = u(x0, t0), declared) or an array."""

    def __init__(self, x0, velocity: Callable, *, axes: Sequence[Periodic | Chebyshev],
                 scheme: str, boundary: str, t0: float = 0.0, inertia: MaxeyRiley | None = None,
                 v0: Any = "fluid", xp: Any = None) -> None:
        if scheme not in ("rk4", "rk2"):
            raise OptionError(f"scheme {scheme!r}: 'rk4' or 'rk2'")
        if boundary not in ("clip", "raise"):
            raise OptionError(f"boundary {boundary!r}: 'clip' (counted) or 'raise'")
        self.xp = xp if xp is not None else getattr(velocity, "xp", np)
        self.axes, self.scheme, self.boundary = tuple(axes), scheme, boundary
        self.velocity, self.inertia = velocity, inertia
        self.x = self.xp.array(x0, dtype=float)          # unwrapped positions
        if self.x.ndim != 2 or self.x.shape[1] != len(self.axes):
            raise CoordinateError(f"x0 of shape {self.x.shape}: (Np, {len(self.axes)}) expected")
        self.t, self.nsteps = float(t0), 0
        self._clip = [0]
        self.v = None
        if inertia is not None:
            if isinstance(v0, str):
                if v0 != "fluid":
                    raise OptionError(f"v0={v0!r}: 'fluid' or an array")
                self.v = self._u(self.x, self.t)
            else:
                self.v = self.xp.array(v0, dtype=float)

    @property
    def n_clipped(self) -> int:
        """Number of bounded coordinates clipped back into the domain so far (all RK stages)."""
        return self._clip[0]

    @property
    def positions(self):
        """Positions with periodic coordinates wrapped into the domain."""
        return _bounds_policy(self.axes, self.x, "clip", [0], self.xp)

    def _u(self, x, t):
        return self.xp.asarray(self.velocity(_bounds_policy(self.axes, x, self.boundary, self._clip, self.xp), t))

    def _rhs(self, s, t):
        """Right-hand side of the particle ODE on the state s = x or (x, v)."""
        if self.inertia is None:
            return self._u(s, t)
        x, v = s
        a = (self._u(x, t) - v) / self.inertia.tau_p
        if self.inertia.g is not None:
            a = a + (1.0 - self.inertia.beta) * self.xp.asarray(self.inertia.g)[None, :]
        return (v, a)

    def advance(self, dt: float, nsteps: int = 1) -> None:
        """Take ``nsteps`` steps of size ``dt`` (a negative dt integrates backwards)."""
        def ax(s, k, c):
            return s + c * k if self.inertia is None else (s[0] + c * k[0], s[1] + c * k[1])

        def comb(s, ks, cs):
            if self.inertia is None:
                return s + sum(c * k for c, k in zip(cs, ks, strict=True))
            return (s[0] + sum(c * k[0] for c, k in zip(cs, ks, strict=True)), s[1] + sum(c * k[1] for c, k in zip(cs, ks, strict=True)))

        from ..timestep.integrator import check_steps
        for _ in range(check_steps(nsteps, "Particles.advance")):
            s, t = (self.x if self.inertia is None else (self.x, self.v)), self.t
            if self.scheme == "rk4":
                k1 = self._rhs(s, t)
                k2 = self._rhs(ax(s, k1, dt / 2), t + dt / 2)
                k3 = self._rhs(ax(s, k2, dt / 2), t + dt / 2)
                k4 = self._rhs(ax(s, k3, dt), t + dt)
                s = comb(s, (k1, k2, k3, k4), (dt / 6, dt / 3, dt / 3, dt / 6))
            else:
                k1 = self._rhs(s, t)
                k2 = self._rhs(ax(s, k1, dt), t + dt)
                s = comb(s, (k1, k2), (dt / 2, dt / 2))
            xn = s if self.inertia is None else s[0]
            n_bad = int(xn.size - self.xp.count_nonzero(self.xp.isfinite(xn)))
            if n_bad:                                    # never kept or dropped silently
                raise ConvergenceError(f"{n_bad} non-finite particle coordinates at t = {self.t + dt:g}",
                                       why="the velocity returned non-finite values",
                                       fix="check the flow field; the particles were not advanced")
            if self.inertia is None:
                self.x = s
            else:
                self.x, self.v = s
            if self.boundary == "clip":              # keep bounded coordinates in the domain
                for i, a in enumerate(self.axes):
                    if isinstance(a, Chebyshev):
                        out = (self.x[:, i] < a.a) | (self.x[:, i] > a.b)
                        n = int(self.xp.count_nonzero(out))
                        if n:
                            self._clip[0] += n
                            self.x[:, i] = self.xp.clip(self.x[:, i], a.a, a.b)
            self.t += dt
            self.nsteps += 1

    def explain(self) -> str:
        """Scheme, boundary policy, counters, velocity and inertia model."""
        kind = "tracers" if self.inertia is None else self.inertia.explain()
        vel = self.velocity.explain() if hasattr(self.velocity, "explain") else repr(self.velocity)
        return (f"Particles({self.x.shape[0]} {kind if self.inertia is None else 'inertial'} in "
                f"{len(self.axes)}D, {self.scheme.upper()}, t = {self.t:g} after {self.nsteps} steps, "
                f"boundary '{self.boundary}' ({self.n_clipped} coordinates clipped); velocity: {vel}"
                + ("" if self.inertia is None else f"; {kind}") + ")")


# ====================================================================== pathlines / emitters
class PathRecorder:
    """Bounded pathline history of a :class:`Particles`: ``record()`` every ``every`` calls keeps
    at most ``max_points`` positions per particle (oldest dropped, counted in ``dropped``)."""

    def __init__(self, particles: Particles, *, every: int, max_points: int) -> None:
        from ..errors import check_count
        # integers, never truncated (finding 66)
        self.p = particles
        self.every = check_count(every, "every", "PathRecorder")
        self.max_points = check_count(max_points, "max_points", "PathRecorder", minimum=2)
        self._calls, self.dropped = 0, 0
        self._hist: list = []
        self.times: list[float] = []

    def record(self) -> None:
        """Call after each advance; stores the unwrapped positions every ``every`` calls."""
        if self._calls % self.every == 0:
            self._hist.append(self.p.x.copy())
            self.times.append(self.p.t)
            if len(self._hist) > self.max_points:
                self._hist.pop(0)
                self.times.pop(0)
                self.dropped += 1
        self._calls += 1

    def polylines(self, *, wrap: bool = True):
        """(Np, n_pts, d) array; ``wrap``: periodic coordinates wrapped into the domain and the
        point where a line crosses a periodic boundary replaced by NaN (display break, declared)."""
        xp = self.p.xp
        H = xp.stack(self._hist, axis=1)                 # (Np, n, d)
        if not wrap:
            return H
        W = H.copy()
        for i, ax in enumerate(self.p.axes):
            if isinstance(ax, Periodic):
                W[..., i] = ax.a + xp.mod(H[..., i] - ax.a, ax.L)
        jump = xp.zeros(H.shape[:2], dtype=bool)
        for i, ax in enumerate(self.p.axes):
            if isinstance(ax, Periodic):
                jump[:, 1:] |= xp.abs(xp.diff(W[..., i], axis=1)) > ax.L / 2
        W[jump] = xp.nan
        return W


class Emitter:
    """Particles released continuously from fixed ``seeds`` (ns, d) every ``emit_every`` steps,
    at most ``capacity`` per seed (the oldest is dropped when full, counted in ``dropped``).

    ``mode="streak"``: :meth:`polylines` gives one line per seed, newest point first (numerical
    smoke wire: streaklines). ``mode="timeline"``: one line per emission, across the seeds
    (hydrogen-bubble wire: timelines). Unemitted slots are NaN."""

    def __init__(self, seeds, velocity: Callable, *, axes, emit_every: int, capacity: int,
                 scheme: str, boundary: str, mode: str, t0: float = 0.0, xp: Any = None) -> None:
        if mode not in ("streak", "timeline"):
            raise OptionError(f"mode {mode!r}: 'streak' or 'timeline'")
        from ..errors import check_count
        # integers, never truncated (finding 66)
        emit_every = check_count(emit_every, "emit_every", "Emitter")
        capacity = check_count(capacity, "capacity", "Emitter")
        self.xp = xp if xp is not None else getattr(velocity, "xp", np)
        self.seeds = self.xp.array(seeds, dtype=float)
        self.mode, self.emit_every, self.capacity = mode, int(emit_every), int(capacity)
        ns, d = self.seeds.shape
        self.X = self.xp.full((ns, self.capacity, d), self.xp.nan)
        self._p = Particles(self.seeds[:1], velocity, axes=axes, scheme=scheme, boundary=boundary,
                            t0=t0, xp=self.xp)
        self.dropped, self.emitted, self._step = 0, 0, 0
        self._emit()

    @property
    def t(self) -> float:
        """Current time."""
        return self._p.t

    @property
    def n_clipped(self) -> int:
        """Clipped bounded coordinates (see :class:`Particles`)."""
        return self._p.n_clipped

    def _emit(self) -> None:
        xp = self.xp
        if not bool(xp.all(xp.isnan(self.X[:, -1, 0]))):
            self.dropped += self.X.shape[0]
        self.X = xp.roll(self.X, 1, axis=1)
        self.X[:, 0] = self.seeds
        self.emitted += 1

    def advance(self, dt: float, nsteps: int = 1) -> None:
        """Advect every emitted particle, then emit when due (every ``emit_every`` steps)."""
        xp = self.xp
        from ..timestep.integrator import check_steps
        for _ in range(check_steps(nsteps, "Emitter.advance")):
            live = ~xp.isnan(self.X[..., 0])
            self._p.x = self.X[live]
            self._p.advance(dt)
            self.X[live] = self._p.x
            self._step += 1
            if self._step % self.emit_every == 0:
                self._emit()

    def polylines(self):
        """streak: (ns, capacity, d); timeline: (capacity, ns, d) — periodic coordinates wrapped,
        NaN where nothing was emitted yet."""
        xp = self.xp
        W = self.X.copy()
        for i, ax in enumerate(self._p.axes):
            if isinstance(ax, Periodic):
                W[..., i] = ax.a + xp.mod(W[..., i] - ax.a, ax.L)
        return W if self.mode == "streak" else xp.moveaxis(W, 1, 0)

    def explain(self) -> str:
        """Seeds, emission, capacity and counters."""
        what = ("streaklines (numerical smoke wire)" if self.mode == "streak"
                else "timelines (numerical hydrogen-bubble wire)")
        return (f"Emitter({what}: {self.seeds.shape[0]} seeds, one particle per seed every "
                f"{self.emit_every} steps, capacity {self.capacity} per seed (oldest dropped: "
                f"{self.dropped}), {self.emitted} emissions; {self._p.explain()})")


class Streaklines(Emitter):
    """:class:`Emitter` with ``mode='streak'`` (one polyline per seed)."""

    def __init__(self, seeds, velocity: Callable, **kw: Any) -> None:
        super().__init__(seeds, velocity, mode="streak", **kw)


class Timelines(Emitter):
    """:class:`Emitter` with ``mode='timeline'`` (one polyline per emission)."""

    def __init__(self, seeds, velocity: Callable, **kw: Any) -> None:
        super().__init__(seeds, velocity, mode="timeline", **kw)


# ====================================================================== FTLE
def ftle(velocity: Callable, points: Sequence, *, axes, t0: float, T: float, dt: float,
         delta: float, scheme: str = "rk4", boundary: str = "raise", xp: Any = None):
    """Finite-time Lyapunov exponent on the lattice ``points`` (one 1D coordinate array per axis).

    For each lattice point x, 2d auxiliary neighbours x +- delta e_j are advected over [t0, t0 + T]
    (T < 0: backward FTLE) with steps of |dt| (T / dt must be an integer: no silent rounding);
    F_ij = (phi_i(x + delta e_j) - phi_i(x - delta e_j)) / (2 delta) (centred differences,
    unwrapped periodic coordinates), C = F^T F, FTLE = ln(sqrt(lambda_max(C))) / |T|.
    Returns (ftle array of lattice shape, info dict)."""
    xp = xp if xp is not None else getattr(velocity, "xp", np)
    d = len(points)
    nst = abs(T / dt)
    from ..timestep.integrator import is_whole        # the one whole-steps rule (WHOLE_STEPS_TOL)
    if not is_whole(nst) or round(nst) < 1:
        raise OptionError(f"T / dt = {T / dt}: an integer number of steps is required")
    nst = int(round(nst))
    step = abs(dt) * (1 if T > 0 else -1)
    mesh = xp.meshgrid(*[xp.asarray(p, dtype=float) for p in points], indexing="ij")
    shape = mesh[0].shape
    X0 = xp.stack([m.ravel() for m in mesh], axis=1)                  # (M, d)
    M = X0.shape[0]
    E = xp.eye(d) * delta
    aux = xp.concatenate([X0 + E[j][None, :] * s for j in range(d) for s in (1.0, -1.0)])
    p = Particles(aux, velocity, axes=axes, scheme=scheme, boundary=boundary, t0=t0, xp=xp)
    p.advance(step, nst)
    Y = p.x.reshape(d, 2, M, d)                                        # (j, +/-, M, i)
    F = xp.moveaxis((Y[:, 0] - Y[:, 1]) / (2 * delta), 0, 2)            # (M, i, j)
    C = xp.einsum("mki,mkj->mij", F, F)
    lam = xp.linalg.eigvalsh(C)[:, -1]
    out = xp.log(xp.sqrt(lam)) / abs(T)
    return out.reshape(shape), dict(T=T, dt=abs(dt), steps=nst, delta=delta, scheme=scheme,
                                    n_clipped=p.n_clipped)
