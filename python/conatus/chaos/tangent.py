"""Batched exact tangent of the discrete step: k tangent vectors per trajectory, B
trajectories, advanced through the same IMEX Runge–Kutta stages as the trajectory.

For one RK step ("Tangent"), stage i of the tangent solves

    (M + dt H_ii L) dX_i = M dX_0 + dt sum_k A_ik F'(X_k) dX_k - dt sum_k H_ik L dX_k,

with the recorded stage states X_k of the trajectory. This is ``FlowMap.jvp`` (exact derivative of
the discrete map) evaluated for a whole block of directions at once: F'(X_k) V from the batched
polynomial/general form (``BatchedRHS.jvp``, base grids broadcast over the k directions) and one
stacked solve per stage for the B x k right-hand sides (``BatchedSystem.solve``). Agreement with
``FlowMap.jvp`` is measured (to rounding) in tests/chaos.

Multistep (SBDF) maps act on the augmented state (X_n, ..., X_{n-q+1}); their tangent spectrum
contains the parasitic roots of the scheme, so they are refused here (explicit) rather than
filtered silently.
"""
from __future__ import annotations

from ..errors import UnsupportedError

__all__ = ["rk_tangent_step", "require_rk"]


def require_rk(ens) -> None:
    """Refuse multistep schemes (see module docstring)."""
    if ens.order:
        raise UnsupportedError(
            f"tangent dynamics of {ens.scheme}: multistep map on an augmented state (parasitic "
            "roots in the tangent spectrum)",
            why="Lyapunov vectors of an SBDF map live in the augmented space (X_n, ..., X_{n-q+1})",
            fix="use an IMEX Runge-Kutta scheme (RK222, RK443) for the trajectory and its tangent")


def rk_tangent_step(ens, stages: list, V, dt: float, tableau):
    """dX_s for the directions V (B, k, n) given the stage states ``stages`` (each (B, n)) of
    one RK step of the ensemble ``ens`` (same tableau and dt)."""
    A, H, c = tableau
    sysm, rhs_ = ens.sys, ens.rhs
    dS = [V]
    MdX0 = sysm.M(V)
    for i in range(1, len(c)):
        rhs = MdX0.copy()
        for k in range(i):
            if A[i][k] != 0:
                rhs += (dt * A[i][k]) * rhs_.jvp(stages[k], dS[k])
            if H[i][k] != 0:
                rhs -= (dt * H[i][k]) * sysm.L(dS[k])
        dS.append(sysm.solve(1.0, dt * H[i][i], rhs))
    return dS[-1]
