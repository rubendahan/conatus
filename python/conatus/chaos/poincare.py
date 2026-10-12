"""Poincaré sections and first-return maps with declared event location.

A section is the zero set of a user function g(X) (evaluated on a batch of states: (B, n) -> (B,)),
crossed in a declared direction (+1: g goes from < 0 to >= 0, -1: the reverse, 0: both). The
trajectories are advanced as an :class:`Ensemble` (one batch for all initial states / parameter
values). When g changes sign over a step [X_n, X_{n+1}], the crossing is located on the **scheme
itself**: τ* in (0, dt] with g(Φ_τ*(X_n)) = 0, Φ_τ one step of the same IMEX Runge–Kutta scheme of
size τ from X_n (Illinois regula falsi, bracket kept). So the section point carries the scheme's
local error at step τ* ≤ dt, and no interpolant (dense output, Hermite) is assumed — the latter
would need M⁻¹, undefined with algebraic rows (taus, gauges). ``g_tol`` (on |g|) and
``max_iter`` are mandatory; a crossing not located within them is **refused** (ConvergenceError),
never accepted approximately. The refinement is done member by member with the member's own
single solver (crossings are rare events), and the step factors created for τ are discarded after
each location (the solver's factor cache keeps only its own dt).

Multistep schemes are refused (Φ_τ from a single state is not their step).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..ensembles import Ensemble
from ..errors import ConvergenceError, OptionError
from .tangent import require_rk
from .._deprecation import renames
from .._repr import compact_repr

#: Iteration cap of the regula-falsi location of one crossing (rule a
#: budget has a declared default; the accuracy g_tol stays required; a crossing not located
#: within it is refused, never accepted).
SECTION_MAX_ITERATIONS = 50

__all__ = ["poincare_section", "PoincareResult"]


@dataclass
class PoincareResult:
    __repr__ = compact_repr   # one line, arrays by shape (as)
    points: list                  # per member: array (m_b, n) of section points
    times: list                   # per member: array (m_b,) of crossing times
    residuals: list               # per member: |g| at the located points
    iterations: list              # per member: regula falsi iterations per crossing
    settings: dict
    notes: list = field(default_factory=list)

    def return_pairs(self, observable: Callable) -> tuple[np.ndarray, np.ndarray]:
        """(s_k, s_{k+1}) pairs of a scalar observable of the section points, all members
        (the first-return map in that coordinate)."""
        a, b = [], []
        for P in self.points:
            if len(P) >= 2:
                s = np.asarray([observable(p) for p in P])
                a.append(s[:-1])
                b.append(s[1:])
        return (np.concatenate(a), np.concatenate(b)) if a else (np.empty(0), np.empty(0))

    def return_times(self) -> np.ndarray:
        """Times between successive crossings, all members."""
        return np.concatenate([np.diff(t) for t in self.times if len(t) >= 2]) \
            if any(len(t) >= 2 for t in self.times) else np.empty(0)

    def explain(self) -> str:
        s = self.settings
        n = sum(len(p) for p in self.points)
        its = np.concatenate([np.asarray(i) for i in self.iterations if len(i)]) if n else np.zeros(0)
        res = np.concatenate([np.asarray(r) for r in self.residuals if len(r)]) if n else np.zeros(0)
        return (f"Poincaré section ({s['direction_text']}), {len(self.points)} member(s), {n} "
                f"crossing(s) over {s['steps']} steps of {s['scheme']} dt = {s['dt']:g}\n"
                f"  location: τ* in (0, dt] with g(Φ_τ*(X_n)) = 0 on the scheme (Illinois regula "
                f"falsi), |g| <= {s['g_tol']:g}, max {s['max_iterations']} iterations"
                + (f"; observed max |g| {res.max():.1e}, mean iterations {its.mean():.1f}" if n else ""))


def _single_step(solver, X, tau: float, tab):
    return solver.rk_stages(X, 0.0, tau, tab)[-1]


@renames("conatus.chaos.poincare:poincare_section")
def poincare_section(solver, x0, g: Callable, *, direction: int, steps: int, g_tol: float,
                     max_iterations: int = SECTION_MAX_ITERATIONS, rhs: str = "batched",
                     solvers=None) -> PoincareResult:
    """Section points of the trajectories from ``x0`` (n,) or (B, n) over ``steps`` steps.
    ``g``: batch function (B, n) -> (B,) (array module of the solver); ``direction`` in (+1, -1, 0).
    ``solvers``: optional list (parameter ensemble, one member per solver, same numerics)."""
    if direction not in (1, -1, 0):
        raise OptionError("direction must be +1, -1 or 0 (both)")
    if not g_tol > 0 or max_iterations < 1:
        raise OptionError("g_tol > 0 and max_iterations >= 1 are required")
    xp = solver.xp
    x0 = xp.asarray(x0)
    if x0.ndim == 1:
        x0 = x0[None]
    B = x0.shape[0]
    if solvers is None:
        ens, members = Ensemble(solver, B, rhs=rhs), [solver] * B
    else:
        if len(solvers) != B:
            raise OptionError("one solver per initial state")
        ens, members = Ensemble.from_solvers(solvers, rhs=rhs), list(solvers)
    require_rk(ens)
    ens.set_states(x0)
    tab, dt = ens._rk, ens.dt

    def gval(X):
        v = g(X)
        return np.asarray(v if xp is np else xp.asnumpy(v), float)

    pts = [[] for _ in range(B)]
    tms = [[] for _ in range(B)]
    res = [[] for _ in range(B)]
    its = [[] for _ in range(B)]
    g0 = gval(ens.X)
    for n in range(steps):
        Xn = ens.X.copy()
        ens.step()
        g1 = gval(ens.X)
        up = (g0 < 0) & (g1 >= 0)
        down = (g0 >= 0) & (g1 < 0)
        hit = up if direction == 1 else (down if direction == -1 else (up | down))
        for b in np.nonzero(hit)[0]:
            s = members[b]
            keep = set(s.system._factors)
            a_, b_, ga, gb = 0.0, dt, g0[b], g1[b]
            side = 0
            for k in range(1, max_iterations + 1):  # noqa: B007 -- read after the loop
                tau = b_ - gb * (b_ - a_) / (gb - ga)
                Xt = _single_step(s, Xn[b], tau, tab)
                gt = float(gval(Xt[None])[0])
                if abs(gt) <= g_tol:
                    break
                if np.sign(gt) == np.sign(gb):
                    b_, gb = tau, gt
                    if side == -1:
                        ga /= 2                                   # Illinois modification
                    side = -1
                else:
                    a_, ga = tau, gt
                    if side == 1:
                        gb /= 2
                    side = 1
            else:
                raise ConvergenceError(
                    f"Poincaré crossing of member {b} at step {n} not located to |g| <= {g_tol:g} "
                    f"in {max_iterations} iterations (last |g| = {abs(gt):.2e})",
                    fix="raise max_iterations, loosen g_tol, or reduce dt (the crossing is refused, "
                        "not accepted approximately)")
            for key in set(s.system._factors) - keep:             # factors made for τ only
                del s.system._factors[key]
            pts[b].append(Xt if xp is np else xp.asnumpy(Xt))
            tms[b].append(ens._t0 + n * dt + tau)
            res[b].append(abs(gt))
            its[b].append(k)
        g0 = g1
    settings = dict(direction=direction, direction_text={1: "g increasing through 0",
                                                         -1: "g decreasing through 0",
                                                         0: "both directions"}[direction],
                    steps=steps, scheme=ens.scheme, dt=dt, g_tol=g_tol, max_iterations=max_iterations, rhs=rhs)
    return PoincareResult([np.asarray(p).reshape(-1, ens.n) for p in pts],
                          [np.asarray(t) for t in tms], res, its, settings)
