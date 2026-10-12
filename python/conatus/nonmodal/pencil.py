"""Matrix kernel of the descriptor path: :func:`reduce_pencil`, the generator Ã on the
admissible states of a pencil M x' + L x = 0 in energy-orthonormal coordinates (ordered QZ, no
eigenvectors; see ``statespace.py``).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from ..errors import OptionError, SpectralError

__all__ = ["reduce_pencil"]


def reduce_pencil(M: np.ndarray, L: np.ndarray, cols: np.ndarray, F: np.ndarray, *,
                  infinite_tol: float):
    """(Ã, to_state, n_infinite, gap, cond R_e): see ``statespace.state_space``."""
    if not infinite_tol > 0:
        raise OptionError("infinite_tol must be > 0 (mandatory: it decides the finite/infinite "
                          "split of the QZ pairs)")
    scale = max(np.abs(L).max(), np.abs(M).max())

    def finite(alpha, beta):
        return np.abs(beta) > infinite_tol * np.abs(alpha)

    S, T, alpha, beta, Q, Z = sla.ordqz(-L / scale, M / scale, sort=finite, output="complex")
    fin = finite(alpha, beta)
    k = int(fin.sum())
    if not fin[:k].all():
        raise SpectralError("ordered QZ did not put the finite eigenvalues first")
    lam_fin = np.abs(alpha[:k] / beta[:k])
    with np.errstate(divide="ignore", invalid="ignore"):    # β = 0 (and 0/0) are infinite
        lam_inf = np.abs(alpha[k:] / beta[k:])
    lam_inf = np.where(np.isnan(lam_inf), np.inf, lam_inf)
    gap = float(lam_inf.min() / lam_fin.max()) if k < len(alpha) else np.inf
    if gap < 1e4:
        raise SpectralError(f"finite/infinite split not clean: gap {gap:.2e} < 1e4",
                            fix="change infinite_tol or the resolution; the split must not "
                                "depend on the tolerance")
    Aq = sla.solve_triangular(T[:k, :k], S[:k, :k])            # T₁₁⁻¹ S₁₁ (upper triangular)
    X = F @ Z[cols, :k]
    Qe, Re = np.linalg.qr(X)
    sv = np.linalg.svd(Re, compute_uv=False)
    if sv[-1] <= 1e-13 * sv[0]:
        raise SpectralError("the state columns do not determine the finite deflating subspace "
                            f"(σ_min/σ_max of F C Z₁ = {sv[-1] / sv[0]:.1e})")
    Rinv = sla.solve_triangular(Re, np.eye(k))                 # R_e is upper triangular
    return Re @ Aq @ Rinv, Z[cols, :k] @ Rinv, len(alpha) - k, gap, float(sv[0] / sv[-1])




