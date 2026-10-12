"""Floquet–Hill slogdet and solves on the GPU, batched over many μ at once.

    S_k(μ_p) = D_k(μ_p) - L_k S_{k-1}(μ_p)⁻¹ U_{k-1}      (batched getrf/getrs, FP64)

and log|det| = Σ_k log|det S_k| (batched slogdet). The recurrence over k stays sequential; the
parallelism is over μ (contours, winding numbers, scans) and inside each LU. A single μ (P = 1)
runs the same code. Everything is declared in ``explain()``; the host engine
(``hill_general.HillBlocks``) remains the oracle and the tests compare both.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .hill_general import HillBlocks
from ..errors import OptionError

__all__ = ["HillGPU", ]


class HillGPU:
    """Device mirror of a μ-affine ``HillBlocks`` (optionally with μ-dependent patched rows).

    ``patch(mu) -> (rows, values)``: block rows ``rows`` of every harmonic's diagonal block are
    replaced by ``values[k]`` (shape (len(rows), size)) for harmonic index k; Without ``patch`` the operator must be exactly μ-affine,
    which is checked at construction (refused otherwise).
    """

    def __init__(self, hill: HillBlocks, *, patch: Callable[[complex], tuple] | None = None,
                 affine_check_tol: float = 1e-12):
        import cupy as cp
        if hill.mass is None:
            raise OptionError("HillGPU needs the μ-derivative block (HillBlocks(mass=...))",
                              fix="HillBlocks(..., mass=...)")
        self.cp, self.hill, self.patch = cp, hill, patch
        h = hill.harmonics
        D0 = np.stack([hill.diag(0.0, n) for n in h])
        if patch is None:                               # exact affinity in μ, checked once
            # two probe points (one real, one complex): a single probe at μ = 1 accepted
            # (1 + μ²) I with mass I (D(1) - D(0) = I), a silent acceptance of a wrong operator
            err = 0.0
            for probe in (1.0, 0.37 + 0.81j):
                Dp = np.stack([hill.diag(probe, n) for n in h])
                err = max(err, np.max(np.abs(Dp - D0 - probe * hill.mass[None])))
            if err > affine_check_tol * max(1.0, np.max(np.abs(D0))):
                raise OptionError(f"diagonal blocks are not affine in μ (deviation {err:.1e}); "
                                  "pass patch= for μ-dependent rows",
                                  fix="pass patch= for the mu-dependent rows")
        self.sup = hill._super()
        w = len(self.sup[0])
        self.w = w
        # super-block diagonal (without μ B), super-offdiagonals (constant), mass per super-block
        self._D0 = [cp.asarray(hill._block(0.0, r, r)) for r in self.sup]
        self._L = [cp.asarray(hill._block(0.0, self.sup[k], self.sup[k - 1]))
                   for k in range(1, len(self.sup))]
        self._U = [cp.asarray(hill._block(0.0, self.sup[k], self.sup[k + 1]))
                   for k in range(len(self.sup) - 1)]
        self._B = [cp.asarray(np.kron(np.eye(len(r)), hill.mass)) for r in self.sup]
        self._h = h

    # ------------------------------------------------------------------ batched evaluation
    def _patch_values(self, mus: np.ndarray):
        """Host-computed patched rows for every μ, uploaded once: (rows, device array (P, K, r, s))."""
        if self.patch is None:
            return None, None
        out = [self.patch(complex(mu)) for mu in mus]
        rows = out[0][0]
        return list(rows), self.cp.asarray(np.stack([np.asarray(v) for _, v in out]))

    def _diag_batch(self, k: int, mus: np.ndarray, prows, pvals):
        cp, s = self.cp, self.hill.size
        Dk = self._D0[k][None] + cp.asarray(mus)[:, None, None] * self._B[k][None]
        if prows is not None:
            for i, n in enumerate(self.sup[k]):
                kk = self._h.index(n)
                for j, r in enumerate(prows):
                    Dk[:, i * s + r, :] = 0
                    Dk[:, i * s + r, i * s:(i + 1) * s] = pvals[:, kk, j, :]
        return Dk

    def slogdet(self, mus) -> tuple[np.ndarray, np.ndarray]:
        """(phases, log|det|) of M(μ_p) for every μ_p in ``mus`` (batched on the device)."""
        cp = self.cp
        mus = np.atleast_1d(np.asarray(mus, complex))
        prows, pvals = self._patch_values(mus)
        phase = cp.ones(mus.size, complex)
        logabs = cp.zeros(mus.size)
        S = None
        for k in range(len(self.sup)):
            D = self._diag_batch(k, mus, prows, pvals)
            if k > 0:
                U = cp.broadcast_to(self._U[k - 1], (mus.size,) + self._U[k - 1].shape)
                D = D - self._L[k - 1][None] @ cp.linalg.solve(S, U)
            sgn, la = cp.linalg.slogdet(D)
            phase *= sgn
            logabs += la
            S = D
        return cp.asnumpy(phase), cp.asnumpy(logabs)

    def explain(self) -> str:
        return (f"Floquet–Hill on the GPU (CuPy, FP64): block-Thomas on {len(self.sup)} super-blocks "
                f"of {self.w} harmonics, batched over μ (cuSOLVER getrf/getrs); diagonal blocks "
                f"D(0) + μ B" + (" with μ-dependent patched rows computed on the host" if self.patch
                                 else " (μ-affinity checked)") + "; host engine = oracle. "
                + self.hill.explain())


