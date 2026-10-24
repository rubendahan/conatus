"""Homoclinic snaking of spatially localized states: restriction of a
:class:`~conatus.bifurcation.DynamicalSystem` to a declared invariant subspace (reflection-even
states, which removes the translation mode), localized seeds, and the pinning region read from
the folds of a snaking branch continued with ``conatus.bifurcation.continue_equilibria``.
Notes: Burke & Knobloch, PRE 73, 056211 (2006) and
Phys. Lett. A 360, 681 (2007).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["restrict_system", "even_subspace", "PinningRegion", "pinning_region"]


def even_subspace(solver) -> np.ndarray:
    """Indices of the reflection-even dofs f(x) = f(−x) of a 1D RealFourier solver (the cos
    coefficients, mean included; the sin coefficients are odd)."""
    from ..ecs.fourier_ops import FourierOps
    ops = FourierOps(solver)
    return np.flatnonzero(~ops.odd_mask(np.zeros(solver.system.n)))


def restrict_system(system, index, *, name: str, invariance_tol: float = 1e-12):
    """G_r(y) = Qᵀ G(Q y) with Q = the coordinate columns ``index`` (an invariant coordinate
    subspace, declared). Invariance is **measured** at a random point, ‖(I − QQᵀ) G(Q y)‖/‖G‖, and
    refused above ``invariance_tol``; J, B, C, ∂G/∂p are restricted the same way."""
    from ..bifurcation.system import DynamicalSystem
    idx = np.asarray(index, int)
    n, m = system.n, len(idx)

    def up(y):
        x = np.zeros(n, dtype=np.result_type(y, float))
        x[idx] = y
        return x

    rng = np.random.default_rng(3)
    y = 0.3 * rng.standard_normal(m)
    g = system.G(up(y))
    leak = np.delete(g, idx)
    inv = float(np.linalg.norm(leak) / max(np.linalg.norm(g), 1e-300))
    if inv > invariance_tol:
        raise OptionError(f"restrict_system: the subspace is not invariant (leak {inv:.1e} > "
                          f"invariance_tol {invariance_tol:g})")
    M = system.M
    Mr = M[idx][:, idx] if hasattr(M, "tocsr") else np.asarray(M)[np.ix_(idx, idx)]

    def Jv(x, v, par):
        return system.Jv(up(x), up(v), par)[idx]

    ds = DynamicalSystem(
        m, params=system.params, origin=f"{system.origin} restricted to {name}",
        G=lambda x, par: system.G(up(x), par)[idx],
        J=lambda x, par: np.column_stack([Jv(x, e, par) for e in np.eye(m)]),
        Gp=lambda x, par, nm, order=1: system.Gp(up(x), nm, par, order=order)[idx],
        JF_dir=Jv, B=lambda x, u, v, par: system.B(up(x), up(u), up(v), par)[idx],
        C=lambda x, u, v, w, par: system.C(up(x), up(u), up(v), up(w), par)[idx],
        M=Mr, Jp=lambda x, v, par, nm: system.Jp(up(x), up(v), nm, par)[idx],
        notes=list(system.notes) + [f"restricted to {name} ({m} of {n} dofs; measured "
                                    f"invariance leak {inv:.1e})"])
    ds.up, ds.index = up, idx
    return ds


@dataclass
class PinningRegion:
    """Folds of a snaking branch and the pinning interval they converge to."""
    __repr__ = compact_repr   # one line, arrays by shape
    folds: np.ndarray                  # parameter values of the folds, in branch order
    left: np.ndarray                   # folds at the left edge (every other fold)
    right: np.ndarray
    r_minus: float                     # mean of the last ``n_last`` left folds
    r_plus: float
    spread_minus: float                # max − min of those folds (convergence of the snake)
    spread_plus: float
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [f"PinningRegion: {len(self.folds)} folds; r- = {self.r_minus:.6f} (spread "
                 f"{self.spread_minus:.1e}), r+ = {self.r_plus:.6f} (spread {self.spread_plus:.1e})"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def pinning_region(curve, parameter: str, *, skip: int, n_last: int) -> PinningRegion:
    """Pinning interval from the fold events of ``curve`` (continue_equilibria): the first
    ``skip`` folds (the approach to the snake) are dropped, the remaining ones alternate between
    the two edges; the edges are the means of the last ``n_last`` folds on each side (declared),
    with their spread as the convergence evidence. Unconverged fold events are refused."""
    ev = [e for e in curve.events if e.kind == "fold"]
    bad = [e for e in ev if not e.converged or e.point is None]
    if bad:
        raise OptionError(f"pinning_region: {len(bad)} fold events not converged (see curve.events)")
    r = np.array([e.point.z[curve.names[parameter]] for e in ev])
    r = r[skip:]
    if len(r) < 2 * n_last:
        raise OptionError(f"pinning_region: {len(r)} folds after skip, need >= {2 * n_last}")
    a, b = r[0::2], r[1::2]
    left, right = (a, b) if a.mean() < b.mean() else (b, a)
    L, R = left[-n_last:], right[-n_last:]
    pr = PinningRegion(r, left, right, float(L.mean()), float(R.mean()), float(np.ptp(L)),
                       float(np.ptp(R)))
    pr.notes.append(f"{skip} first folds skipped; edges = mean of the last {n_last} folds per side")
    return pr
