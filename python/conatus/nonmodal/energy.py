"""Transient growth and abscissae in the declared energy norm.

All quantities are 2-norms of the generator Ã in energy-orthonormal coordinates
(``StateSpace``), so they are norms in the physical energy ½ Σ w_i ∫|s_i|² with the exact
Gram: ``exp_norm`` = ‖exp(t Ã)‖, ``energy_growth`` = G(t) = ‖exp(t Ã)‖², ``max_growth`` =
max over t of G(t), the numerical and spectral abscissae.
"""
from __future__ import annotations


import numpy as np
import scipy.linalg as sla
import scipy.optimize as so

from ..errors import ConvergenceError, OptionError
from .statespace import StateSpace
from .._deprecation import renames

__all__ = ["restrict", "galerkin_numerical_abscissa", "exp_norm", "energy_growth",
           "max_growth", "numerical_abscissa", "spectral_abscissa"]










# ---------------------------------------------------------------- exponential, growth, abscissae
def exp_norm(ss: StateSpace, t: float) -> float:
    """‖e^{tA}‖_E = ‖e^{tÃ}‖₂ (scaling and squaring on the Schur factor)."""
    T, _ = ss.schur()
    return float(np.linalg.norm(sla.expm(t * T), 2))


def energy_growth(ss: StateSpace, t: float) -> float:
    """G(t) = max over initial states of E(t)/E(0) = ‖e^{tA}‖²_E."""
    return exp_norm(ss, t) ** 2


#: Declared default of ``max_growth(n_scan=)``: points of the
#: uniform scan that brackets the maximum before Brent refines it to ``xtol``. An algorithm
#: control: the precision of t_max is ``xtol``; a scan too coarse for a narrow peak is a
#: resolution choice the caller states with ``n_scan=``.
MAX_GROWTH_N_SCAN = 64


@renames("conatus.nonmodal.energy:max_growth")
def max_growth(ss: StateSpace, t_bounds: tuple[float, float], *, xtol: float,
               n_scan: int = MAX_GROWTH_N_SCAN) -> tuple[float, float]:
    """(G_max, t_max) over t in ``t_bounds``: scan of ``n_scan`` points (declared default
    :data:`MAX_GROWTH_N_SCAN` = 64), then bounded Brent to ``xtol`` (required) around the best
    scan point. The answer is the best of the two interval ends, the best scan point and the
    refined candidate, so a maximum at an end of the interval is found; an optimiser failure
    raises."""
    t0, t1 = (float(t) for t in t_bounds)
    if not (np.isfinite(t0) and np.isfinite(t1) and 0 <= t0 < t1):
        raise OptionError(f"max_growth: t_bounds = {t_bounds!r} must satisfy 0 <= t0 < t1 < inf",
                          fix="t_bounds=(0.0, 10.0)")
    if isinstance(n_scan, bool) or not isinstance(n_scan, (int, np.integer)) or n_scan < 3:
        raise OptionError(f"max_growth: n_scan = {n_scan!r} must be an integer >= 3",
                          fix=f"n_scan={MAX_GROWTH_N_SCAN}")
    ts = np.linspace(t0, t1, int(n_scan))
    g = np.array([energy_growth(ss, t) for t in ts])
    i = int(np.argmax(g))
    lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, n_scan - 1)]
    r = so.minimize_scalar(lambda t: -energy_growth(ss, t), bounds=(lo, hi), method="bounded",
                           options={"xatol": xtol})
    if not r.success:
        raise ConvergenceError(f"max_growth: bounded Brent failed on [{lo:g}, {hi:g}]: {r.message}",
                               fix="raise n_scan or loosen xtol")
    # candidates: both ends (already in the scan), the best scan point, the refined point
    cands = [(float(g[0]), float(ts[0])), (float(g[-1]), float(ts[-1])),
             (float(g[i]), float(ts[i])), (float(-r.fun), float(r.x))]
    best = max(cands, key=lambda c: c[0])
    return best


def numerical_abscissa(ss: StateSpace) -> float:
    """ω(Ã) = max eig((Ã + Ãᴴ)/2) = d/dt ‖e^{tÃ}‖ at t = 0⁺, of *this matrix*.

    On the full tau generator this is dominated by the unresolved top modes (the tau rows are not
    energy-consistent: measured ω grows with N). Use
    ``restrict(ss, cutoff=...)`` (declared truncation) or ``galerkin_numerical_abscissa``
    (variational, converges) for the physical value."""
    return float(np.linalg.eigvalsh(0.5 * (ss.A + ss.A.conj().T))[-1])


def restrict(ss: StateSpace, *, cutoff: float) -> StateSpace:
    """Restriction to the invariant subspace of the eigenvalues with Re λ > ``cutoff``.

    Ordered Schur form of Ã: the kept Schur vectors are energy-orthonormal, so the restricted
    generator is T₁₁ and its 2-norms are energy norms on that subspace. This is the truncated
    eigenfunction expansion of Reddy, Schmid & Henningson (1993) done with unitary vectors;
    ``cutoff`` is mandatory and reported (``explain``)."""
    T, U, kept = sla.schur(ss.A, output="complex", sort=lambda lam: lam.real > cutoff)
    if kept == 0:
        raise OptionError(f"restrict: no eigenvalue with Re λ > {cutoff}")
    out = StateSpace(T[:kept, :kept], ss.to_state @ U[:, :kept], ss.descriptor, ss.n_infinite,
                     ss.gap, ss.cond_Re, ss.infinite_tol)
    out.restricted = (float(cutoff), int(kept), int(ss.k))
    return out


def galerkin_numerical_abscissa(desc, *, quad_points: int) -> float:
    """sup Re⟨u, A u⟩_E / ⟨u, u⟩_E over admissible polynomial states (continuity + BCs), with the
    operator applied *exactly* in weak form (Gauss–Legendre with ``quad_points`` nodes: exact for
    polynomial coefficients once quad_points ≥ N + deg c / 2 + 1). The multiplier (pressure) term
    is checked to vanish on admissible states (⟨u, G p⟩ = ⟨C u, p⟩ = 0), else refused. This is the
    energy-method rate (Reynolds–Orr) of the discretized mode, variational, converging in N."""
    import sympy

    from ..errors import UnsupportedError
    from .system import Y
    from .transient import steady_operator
    sysm, N = desc.system, desc.N
    a, b = sysm.bounds
    xg, wg = np.polynomial.legendre.leggauss(quad_points)
    yg = 0.5 * (b - a) * xg + 0.5 * (a + b)
    wq = 0.5 * (b - a) * wg
    I = np.eye(N)
    D = [np.polynomial.chebyshev.chebval(xg, np.polynomial.chebyshev.chebder(I, k)
                                         if k else I).T * (2.0 / (b - a)) ** k for k in range(5)]
    V = D[0]

    def gal(op, wrow):                 # ∫ T_m (op T_n) · ½ w_row
        G = np.zeros((N, N), complex)
        for k, c in op.items():
            cv = np.asarray(sympy.lambdify(Y, c, "numpy")(yg), dtype=complex) + 0 * yg
            G += V.T @ ((0.5 * wrow * wq * cv)[:, None] * D[k])
        return G

    ns = len(sysm.state)
    Q = np.zeros((ns * N, ns * N), complex)
    for (i, j), op in sysm.A.items():
        ii, jj = sysm.state.index(i), sysm.state.index(j)
        Q[ii * N:(ii + 1) * N, jj * N:(jj + 1) * N] += gal(op, sysm.weights[i])
    Z = steady_operator(desc).admissible_basis()
    for m in sysm.multipliers:         # pressure work on admissible states must vanish
        P = np.zeros((ns * N, N), complex)
        for (i, mm), op in sysm.G.items():
            if mm == m:
                ii = sysm.state.index(i)
                P[ii * N:(ii + 1) * N] += gal(op, sysm.weights[i])
        leak = np.linalg.norm(Z.conj().T @ P) / max(np.linalg.norm(P), 1e-300)
        if leak > 1e-8:
            raise UnsupportedError(f"multiplier '{m}' does work on admissible states "
                                   f"(relative {leak:.1e}): the energy method needs G* = C")
    H = 0.5 * (Q + Q.conj().T)
    B = Z.conj().T @ H @ Z
    Wz = Z.conj().T @ desc.W @ Z
    return float(sla.eigh(B, Wz, eigvals_only=True)[-1])


def spectral_abscissa(ss: StateSpace) -> float:
    return float(np.max(ss.eigenvalues().real))




