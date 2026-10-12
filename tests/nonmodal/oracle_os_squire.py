"""Independent oracle (tests only): Orr–Sommerfeld–Squire in (v, η), Chebyshev collocation.

Classical formulation of Reddy, Schmid & Henningson (1993) and Schmid & Henningson (2001, ch. 3–4):
Trefethen's clamped fourth derivative (Spectral Methods in MATLAB, 2000, program 40), energy
E = 1/(2k²) ∫ (|Dv|² + k²|v|² + |η|²) dy with Clenshaw–Curtis weights. Shares nothing with
conatus (no IR, no tau, no primitive variables, no exact Gram): it cross-checks the generator
Ã of ``conatus.nonmodal`` up to discretization error.
"""
from __future__ import annotations

import numpy as np


def cheb(N):
    """Trefethen's cheb.m: differentiation matrix and Chebyshev points x_j = cos(πj/N)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def clencurt(N):
    """Clenshaw–Curtis weights on the points of cheb(N) (Trefethen, program clencurt.m)."""
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1 / (N ** 2 - 1)
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
        v -= np.cos(N * theta[ii]) / (N ** 2 - 1)
    else:
        w[0] = w[N] = 1 / N ** 2
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
    w[ii] = 2 * v / N
    return w


def os_squire_generator(N, kx, kz, Re, U, dU, d2U):
    """Ã = F A F⁻¹ with F the energy factor: 2-norms of Ã are energy norms."""
    D, x = cheb(N)
    I = np.eye(N - 1)
    D2 = (D @ D)[1:N, 1:N]
    S = np.diag(np.hstack([0, 1 / (1 - x[1:N] ** 2), 0]))
    D3, D4f = D @ D @ D, D @ D @ D @ D
    D4 = ((np.diag(1 - x ** 2) @ D4f - 8 * np.diag(x) @ D3 - 12 * D @ D) @ S)[1:N, 1:N]
    k2 = kx ** 2 + kz ** 2
    xi = x[1:N]
    Ud, dUd, d2Ud = np.diag(U(xi)), np.diag(dU(xi)), np.diag(d2U(xi))
    Mos = D2 - k2 * I
    Los = (-1j * kx * Ud @ Mos + 1j * kx * d2Ud + (D4 - 2 * k2 * D2 + k2 ** 2 * I) / Re)
    Aos = np.linalg.solve(Mos, Los)
    Asq = -1j * kx * Ud + (D2 - k2 * I) / Re
    A = np.block([[Aos, np.zeros_like(Aos)], [-1j * kz * dUd, Asq]])
    # energy: v = (1 - x²) φ with φ(±1) = 0, so Dv at all points from interior v
    D1c = ((np.diag(1 - x ** 2) @ D - 2 * np.diag(x)) @ S)[:, 1:N]
    w = clencurt(N)
    W = np.diag(w)
    E = np.eye(N + 1)[:, 1:N]
    Qv = (D1c.T @ W @ D1c + k2 * E.T @ W @ E) / (2 * k2)
    Qe = (E.T @ W @ E) / (2 * k2)
    Q = np.block([[Qv, np.zeros_like(Qv)], [np.zeros_like(Qv), Qe]])
    F = np.linalg.cholesky(Q).T
    return F @ A @ np.linalg.inv(F)
