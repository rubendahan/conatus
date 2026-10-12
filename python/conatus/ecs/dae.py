"""Equilibria and travelling waves of IVPs with a singular mass matrix (gauge / algebraic rows),
e.g. the streamfunction form of Kolmogorov flow.

A relative equilibrium u(x, t) = U(x − c t) along a RealFourier direction satisfies
R(U, c) = F(U) − L U + c M ∂x U = 0, with the phase condition ⟨∂x U*, δU⟩ = 0 (c = 0 for an
equilibrium in a translation-invariant system: the drift is then an unknown that must converge
to 0, and doing so is part of the evidence). No M⁻¹ is formed. Newton–Krylov–hookstep is run on
the **left-preconditioned** residual P R with P = (−L)⁻¹ (declared; the stiff linear part is
inverted exactly), and the **unpreconditioned** relative residual ‖R‖ / ‖L U‖ is what is reported.
Stability: generalized eigenvalues of (J_F − L + c M ∂x) v = σ M v, dense QZ, infinite ones
removed and counted.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from ..errors import OptionError, ResolutionError

from .fourier_ops import AxisOps
from .newton import NewtonKrylovHookstep
from .solutions import ECSSolution, TravellingWave

__all__ = ["find_relative_equilibrium_dae", "stability_dae"]


class _Lin:
    """Exact linearization of F at X (IR Fréchet derivative through FlowMap's evaluator)."""

    def __init__(self, solver) -> None:
        from ..adjoint import FlowMap
        self.fm = FlowMap(solver, 1)

    def __call__(self, X, V):
        return self.fm._JF(X, V)


def find_relative_equilibrium_dae(solver, X0: np.ndarray, c0: float, newton: NewtonKrylovHookstep, *,
                                  coord: str, log=None, c_zero_tol: float = 1e-10) -> ECSSolution:
    """Solve F(U) − L U + c M ∂x U = 0, unknowns (U, c), phase ⟨∂x U*, δU⟩ = 0.
    ``c_zero_tol``: |c| < c_zero_tol · max(1, max|U|) is reported as an equilibrium (note only;
    declared threshold, audit R7)."""
    sysm = solver.system
    ops = AxisOps(solver, coord)
    lin = _Lin(solver)
    M, L = sysm.M.tocsr(), sysm.L.tocsr()
    P = spla.splu((-L).tocsc())                     # declared left preconditioner (−L)⁻¹
    ref = ops.ddx(X0)
    if np.linalg.norm(ref) == 0:
        raise OptionError(f"initial guess is invariant along {coord}: the phase condition is void",
                          fix=f"start from a state that depends on {coord} (or solve for an "
                              "equilibrium instead)")

    def R(z):
        X, c = z[:-1], z[-1]
        body = solver.F(X, 0.0) - L @ X + c * (M @ ops.ddx(X))
        return np.concatenate([P.solve(body), [0.0]])

    def J(z, d):
        X, c = z[:-1], z[-1]
        dX, dc = d[:-1], d[-1]
        body = lin(X, dX) - L @ dX + c * (M @ ops.ddx(dX)) + dc * (M @ ops.ddx(X))
        return np.concatenate([P.solve(body), [ref @ dX]])

    scale = max(1.0, float(np.linalg.norm(P.solve(L @ X0))))
    res = newton.solve(R, J, np.concatenate([X0, [c0]]), scale=scale, log=log)
    X, c = res.z[:-1], float(res.z[-1])
    raw = solver.F(X, 0.0) - L @ X + c * (M @ ops.ddx(X))
    rel = float(np.linalg.norm(raw) / np.linalg.norm(L @ X))
    kind = "travelling_wave"
    sol = TravellingWave(kind, X, {"c": c, "coord": coord}, rel, res, float("nan"))
    sol.notes.append("residual = ‖F(U) − L U + c M ∂x U‖ / ‖L U‖ (unpreconditioned); Newton on "
                     "the (−L)⁻¹-preconditioned residual")
    if abs(c) < c_zero_tol * max(1.0, float(np.abs(X).max())):
        sol.notes.append(f"c ≈ 0 (|c| < c_zero_tol = {c_zero_tol:g} × max(1, max|U|)): equilibrium "
                         "(drift unknown converged to zero)")
    return sol


def stability_dae(solver, sol: ECSSolution, *, dense_max: int, finite_max: float,
                  translation_kernel_tol: float = 1e-8, translation_overlap_min: float = 0.99) -> dict:
    """Generalized eigenvalues of (J_F − L + c M ∂x) v = σ M v in the co-moving frame (dense QZ,
    size ≤ ``dense_max``); |σ| > ``finite_max`` (explicit) counted as infinite and removed.

    The marginal translation mode is excluded from the unstable count **only if** it exists: the
    translation vector t = ∂x U is non-zero, lies in the kernel (‖A t‖ ≤ ``translation_kernel_tol``
    · ‖A‖_F ‖t‖, A = J_F − L + c M ∂x) and one eigenvector overlaps it by at least
    ``translation_overlap_min``. Otherwise nothing is removed and ``out["translation"]`` says why
    (audit R3: on an x-invariant state the most unstable mode used to be removed)."""
    sysm = solver.system
    n = sysm.n
    if n > dense_max:
        raise ResolutionError(f"stability_dae: n = {n} > dense_max = {dense_max}",
                              why="dense QZ costs O(n^3) time and O(n^2) memory",
                              fix=f"raise dense_max to {n} if that is affordable, or lower N")
    coord, c = sol.params["coord"], sol.params["c"]
    ops = AxisOps(solver, coord)
    lin = _Lin(solver)
    M = sysm.M.toarray()
    cols = []
    for e in np.eye(n):
        cols.append(lin(sol.X, e) + c * (M @ ops.ddx(e)))
    A = np.column_stack(cols) - sysm.L.toarray()
    (a, b), V = sla.eig(A, M, right=True, homogeneous_eigvals=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = a / b
    ok = np.isfinite(sig) & (np.abs(sig) < finite_max)
    sig, V = sig[ok], V[:, ok]
    order = np.argsort(-sig.real)
    sig, V = sig[order], V[:, order]
    t = ops.ddx(sol.X)
    tn = float(np.linalg.norm(t))
    marginal, overlap_max = None, float("nan")
    if tn <= np.finfo(float).eps * max(1.0, float(np.linalg.norm(sol.X))):
        why = f"not excluded: ∂{coord} U = 0 (state invariant along {coord}, no translation mode)"
    else:
        t = t / tn
        kern = float(np.linalg.norm(A @ t) / np.linalg.norm(A))
        overlap = np.abs(t @ V) / np.linalg.norm(V, axis=0)
        i = int(np.argmax(overlap))
        overlap_max = float(overlap[i])
        if kern > translation_kernel_tol:
            why = (f"not excluded: ‖A t‖/‖A‖_F = {kern:.1e} > translation_kernel_tol = "
                   f"{translation_kernel_tol:g} (∂{coord} U is not in the kernel)")
        elif overlap_max < translation_overlap_min:
            why = (f"not excluded: best eigenvector overlap {overlap_max:.3f} < "
                   f"translation_overlap_min = {translation_overlap_min:g}")
        else:
            marginal = i
            why = (f"excluded: σ = {complex(sig[i]):.2e}, overlap {overlap_max:.6f}, "
                   f"‖A t‖/‖A‖_F = {kern:.1e}")
    unstable = [complex(s) for k, s in enumerate(sig) if s.real > 0 and k != marginal]
    return {"eigenvalues": sig, "n_infinite": int((~ok).sum()),
            "marginal": None if marginal is None else complex(sig[marginal]),
            "marginal_overlap": overlap_max, "translation": why,
            "translation_kernel_tol": float(translation_kernel_tol),
            "translation_overlap_min": float(translation_overlap_min),
            "n_unstable": len(unstable),
            "sum_re_unstable": float(sum(s.real for s in unstable)),
            "max_re": float(max((s.real for s in unstable), default=0.0))}
