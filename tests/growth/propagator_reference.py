"""Test oracle: the step-by-step propagator that ``conatus.growth.propagator`` used before.

Kept verbatim (loop over time steps in Python, RK stages inside) as the reference the batched
implementation must reproduce to round-off, and for a before/after timing. Never imported by
``python/conatus/``.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla

from conatus.timestep.integrator import rk_tableau


def reference_propagator(op, t0: float, T: float, nsteps: int, scheme: str, xp=np) -> np.ndarray:
    """Φ(t0, T) by stepping the block of unit states step by step (pre-E92 code)."""
    nsteps = int(nsteps)
    dt = T / nsteps
    A, H, c = (np.asarray(m, dtype=float) for m in rk_tableau(scheme))
    s = len(c) - 1
    hii = H[1, 1]
    n, cols = op.n, op.state_cols
    Ainv = xp.asarray(sla.inv(op.M + dt * hii * op.L0))
    M, L0 = xp.asarray(op.M), xp.asarray(op.L0)
    Bp, Bm = xp.asarray(op.Bp), xp.asarray(op.Bm)
    X = xp.zeros((n, len(cols)), dtype=complex)
    X[xp.asarray(cols), xp.arange(len(cols))] = 1.0
    for k in range(nsteps):
        t = t0 + k * dt
        MX0 = M @ X
        Xs, Fs, LXs = [X], [], []
        for i in range(1, s + 1):
            tj = t + c[i - 1] * dt
            e = complex(math.cos(2 * math.pi * tj), math.sin(2 * math.pi * tj))
            Fs.append(-(e * (Bp @ Xs[-1]) + e.conjugate() * (Bm @ Xs[-1])))
            LXs.append(L0 @ Xs[-1] if H[i:, i - 1].any() else None)
            rhs = MX0.copy()
            for j in range(i):
                if A[i, j]:
                    rhs += (dt * A[i, j]) * Fs[j]
                if H[i, j] and LXs[j] is not None:
                    rhs -= (dt * H[i, j]) * LXs[j]
            Xs.append(Ainv @ rhs)
        X = Xs[-1]
    Phi = X[xp.asarray(cols), :]
    return Phi.get() if xp is not np else np.asarray(Phi)
