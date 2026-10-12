"""Poincaré sections and first-return maps with declared dense interpolation.
Notes: §2.

A section is the zero set of a scalar function g(x) crossed in a declared direction
(+1: g increasing, −1: decreasing, 0: both). Hyperplanes g(x) = ⟨n, x − x*⟩ are built by
:meth:`PoincareSection.hyperplane`. Crossings are bracketed by a sign change of g between two
samples of a :class:`~conatus.manifolds.trajectory.Trajectory` and located on its dense output
by Brent's method (``xtol`` declared). For a discrete map (``SolverIntegrator``) the crossing can be
``refine``-d on the exact partial step (secant on θ in X_n → Φ_θh(X_n)), and the distance between
the interpolated and the re-stepped point is always reported (``interp_defect``): it is the error
of the dense output at that point, not hidden.

Missed crossings: two crossings inside one segment (g returns to its sign) are invisible to a
sign test; the largest |Δg| between samples relative to the section scale is reported so the
caller can judge the sampling.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as so

from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["PoincareSection", "Crossings", "poincare_crossings", "first_return_map"]


class PoincareSection:
    """g(x) = 0 crossed in ``direction`` (+1, −1 or 0), with an optional in-section chart
    ``coords(x)`` (default: the raw state)."""

    def __init__(self, g: Callable[[np.ndarray], float], *, direction: int, name: str,
                 coords: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        if direction not in (-1, 0, 1):
            raise OptionError("PoincareSection: direction must be -1, 0 or +1")
        self.g, self.direction, self.name = g, int(direction), name
        self.coords = coords if coords is not None else (lambda x: np.asarray(x))

    @classmethod
    def hyperplane(cls, normal, point, *, direction: int, basis=None, name: str = "hyperplane"):
        """g(x) = ⟨n, x − x*⟩; ``basis`` (k, n): in-section coordinates ⟨e_j, x − x*⟩."""
        nrm = np.asarray(normal, float)
        x0 = np.asarray(point, float)
        if not np.linalg.norm(nrm) > 0:
            raise OptionError("PoincareSection.hyperplane: zero normal")
        E = None if basis is None else np.atleast_2d(np.asarray(basis, float))
        sec = cls(lambda x: float(nrm @ (np.asarray(x) - x0)), direction=direction, name=name,
                  coords=None if E is None else (lambda x: E @ (np.asarray(x) - x0)))
        sec.normal, sec.point = nrm, x0
        return sec

    def explain(self) -> str:
        d = {1: "g increasing", -1: "g decreasing", 0: "both directions"}[self.direction]
        return f"PoincareSection '{self.name}': g(x) = 0 crossed with {d}"


@dataclass
class Crossings:
    """Crossing times, states (refined if requested), directions and their evidence."""
    __repr__ = compact_repr   # one line, arrays by shape
    t: np.ndarray
    X: np.ndarray
    direction: np.ndarray
    g_residual: np.ndarray             # |g| at the returned point
    interp_defect: np.ndarray          # ‖dense(t*) − restep(t*)‖ (discrete maps), nan otherwise
    method: str
    max_sample_jump: float             # max |Δg| between samples / max |g| (sampling check)
    notes: list[str] = field(default_factory=list)

    def coords(self, section: PoincareSection) -> np.ndarray:
        return np.array([section.coords(x) for x in self.X])

    def explain(self) -> str:
        finite = self.interp_defect[np.isfinite(self.interp_defect)]
        dmax = f"{finite.max():.2e}" if finite.size else "n/a (ODE dense output)"
        lines = [f"{len(self.t)} crossings ({self.method}); max |g| at the points "
                 f"{np.max(self.g_residual, initial=0):.2e}; max interpolation defect "
                 f"{dmax}; max |Δg| between samples / "
                 f"max |g| = {self.max_sample_jump:.2f}"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def poincare_crossings(traj, section: PoincareSection, *, xtol: float,
                       refine: bool = False, t_min: float | None = None) -> Crossings:
    """All crossings of ``section`` by ``traj`` after ``t_min``, located on the dense output
    (Brent, ``xtol`` in time, declared). ``refine=True`` (discrete maps): secant on the exact partial
    step until |g| stops decreasing; the returned state is then a point of the discrete map."""
    g = np.array([section.g(x) for x in traj.X])
    scale = max(float(np.max(np.abs(g))), 1e-300)
    jumps = np.abs(np.diff(g)) / scale
    ts, Xs, ds, res, dfs = [], [], [], [], []
    for i in range(len(g) - 1):
        if t_min is not None and traj.t[i + 1] < t_min:
            continue
        a, b = g[i], g[i + 1]
        if a == 0.0 or (a < 0) == (b < 0) and b != 0.0:
            continue
        d = 1 if b > a else -1
        if section.direction and d != section.direction:
            continue
        f = lambda tt, i=i: section.g(traj._dense(i, tt))      # noqa: E731
        fa, fb = f(traj.t[i]), f(traj.t[i + 1])
        if (fa < 0) == (fb < 0) and fb != 0 and fa != 0:        # dense output disagrees with samples
            continue
        tc = so.brentq(f, traj.t[i], traj.t[i + 1], xtol=xtol) if fa != 0 else traj.t[i]
        xd = traj._dense(i, tc)
        defect = np.nan
        if traj._restep is not None:
            xr = traj._restep(i, tc)
            defect = float(np.linalg.norm(xr - xd))
            if refine:
                tc, xd = _secant_restep(traj, section, i, tc, xr)
        ts.append(tc)
        Xs.append(xd)
        ds.append(d)
        res.append(abs(section.g(xd)))
        dfs.append(defect)
    cr = Crossings(np.array(ts), np.array(Xs).reshape(len(ts), traj.X.shape[1]), np.array(ds, int),
                   np.array(res), np.array(dfs),
                   f"{traj.dense_kind}, Brent xtol = {xtol:g}"
                   + (" + secant on the exact partial step" if refine else ""),
                   float(jumps.max(initial=0.0)))
    cr.notes.append("interp_defect = ‖dense(t*) − exact partial step(t*)‖ (nan for ODE trajectories: "
                    "their dense output is the integrator's own continuous extension)")
    return cr


def _secant_restep(traj, section, i, t0, x0, max_iterations: int = 8):
    """Secant on t of g(restep(t)) from the dense estimate (exact discrete point)."""
    t1 = t0 + 1e-3 * (traj.t[i + 1] - traj.t[i])
    g0, x_best, t_best = section.g(x0), x0, t0
    for _ in range(max_iterations):
        x1 = traj._restep(i, min(max(t1, traj.t[i]), traj.t[i + 1]))
        g1 = section.g(x1)
        if abs(g1) < abs(section.g(x_best)):
            x_best, t_best = x1, t1
        if g1 == g0 or abs(g1) == 0.0:
            break
        t0, t1, g0 = t1, t1 - g1 * (t1 - t0) / (g1 - g0), g1
    return t_best, x_best


def first_return_map(crossings: Crossings, section: PoincareSection) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (P_k, P_{k+1}) of successive in-section coordinates and return times τ_k."""
    P = crossings.coords(section)
    if len(P) < 2:
        raise OptionError("first_return_map: fewer than two crossings")
    return np.stack([P[:-1], P[1:]], axis=1), np.diff(crossings.t)
