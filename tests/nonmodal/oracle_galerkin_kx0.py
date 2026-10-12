"""Independent oracle: componentwise H₂ norms at kx = 0 by a Legendre–Galerkin method (test only).

Written without conatus (review of the resolvent lot). Streamwise-constant
perturbations of a channel flow U(y), y in [−1, 1], energy ∫|·|² dy: stream function ψ (v = i kz ψ,
w = −ψ', clamped: ψ = ψ' = 0 at the walls) and u (Dirichlet), in Shen's Legendre bases; weak forms
integrated exactly by Gauss–Legendre. A Galerkin method projects the white body forcing orthogonally
in the energy (it is the Leray projection restricted to the trial space) and keeps the numerical
range of the operator, unlike a tau method. With time rescaled by Re, the coefficients of Theorem 1 of
Jovanović & Bamieh (2005) are Re-independent traces:

    g_u = tr(response u, forcing in (v, w)) / Re³,   f_x = tr(response u, forcing in u) / Re,
    f_y + f_z = tr(response (v, w), forcing in (v, w)) / Re.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from numpy.polynomial import legendre as L


def _shen(N: int, clamped: bool, y: np.ndarray) -> np.ndarray:
    """Values and first two derivatives of the N Shen basis functions: shape (N, 3, len(y))."""
    out = []
    for j in range(N):
        c = np.zeros(j + 5)
        if clamped:
            c[j], c[j + 2], c[j + 4] = 1, -2 * (2 * j + 5) / (2 * j + 7), (2 * j + 3) / (2 * j + 7)
        else:
            c[j], c[j + 2] = 1, -1
        out.append([L.legval(y, c), L.legval(y, L.legder(c, 1)), L.legval(y, L.legder(c, 2))])
    return np.array(out)


def h2_coefficients(N: int, kz: float, dU, Re: float = 1000.0) -> dict[str, float]:
    """{"g_u", "f_x", "f_y+f_z"} with N basis functions per unknown; ``dU`` = U'(y) (callable)."""
    y, wq = L.leggauss(2 * N + 40)
    P, C = _shen(N, True, y), _shen(N, False, y)

    def ip(a, b):
        return (a * wq) @ b.T

    k2 = kz * kz
    Kp = ip(P[:, 1], P[:, 1]) + k2 * ip(P[:, 0], P[:, 0])              # energy of (v, w)
    Sp = ip(P[:, 2], P[:, 2]) + 2 * k2 * ip(P[:, 1], P[:, 1]) + k2 * k2 * ip(P[:, 0], P[:, 0])
    Mu = ip(C[:, 0], C[:, 0])
    Su = ip(C[:, 1], C[:, 1]) + k2 * Mu
    n = N
    lift = np.zeros((2 * n, 2 * n), complex)
    lift[n:, :n] = -1j * kz * ip(C[:, 0], dU(y) * P[:, 0])              # u_t = -U' v + ...
    F = np.linalg.cholesky(sla.block_diag(Kp, Mu)).conj().T             # energy-orthonormal coords
    Fi = np.linalg.inv(F)
    A = Fi.conj().T @ (-sla.block_diag(Sp, Su) / Re + lift) @ Fi

    def cov(block):
        Q = np.zeros((2 * n, 2 * n))
        Q[block, block] = np.eye(n)
        return sla.solve_continuous_lyapunov(A, -Q)

    vw, u = slice(0, n), slice(n, 2 * n)
    Xvw, Xu = cov(vw), cov(u)
    return {"g_u": float(np.trace(Xvw[u, u]).real / Re ** 3),
            "f_x": float(np.trace(Xu[u, u]).real / Re),
            "f_y+f_z": float(np.trace(Xvw[vw, vw]).real / Re)}
