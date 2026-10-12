"""Bordered banded solvers (C++): P A = B + U V^T, B banded, rank-r border = boundary rows.

``col_border=True`` (opt-in) also takes the first r columns out of the band: every entry of a bulk
row in those columns goes to a sparse column border X, solved exactly around the band factors
(core/include/conatus/banded.hpp, "Column border"). Used when a few columns are dense, e.g. the
lifted gradient tau of the first-order spherical shell."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from . import _core
from ..errors import OptionError

__all__ = ["BorderedBanded", "BorderedBandedBatch", "csr_args", "solve_operand"]


def _kind(dtype) -> str:
    return "Z" if np.issubdtype(np.dtype(dtype), np.complexfloating) else "D"


def solve_operand(b, dtype, where: str, xp=np):
    """Right-hand side ``b`` of a solve with factors of ``dtype``, contiguous, on array module
    ``xp``. An exact widening (float32 -> float64, real -> complex) is done; a narrowing (complex
    data into real factors) is refused: it dropped the imaginary part with only a
    ComplexWarning on the CPU, and the GPU kernels, which read raw pointers, read complex memory
    as real numbers. The single check of every banded solve entry point."""
    b = xp.asarray(b)
    dtype = np.dtype(dtype)
    if b.dtype != dtype:
        if not np.can_cast(b.dtype, dtype, "safe"):
            from ..errors import DtypeError
            raise DtypeError(f"{where}: right-hand side of dtype {b.dtype} for {dtype.name} factors",
                             why="the conversion would lose information (e.g. the imaginary part)",
                             fix=f"pass a {dtype.name} right-hand side (take .real explicitly if "
                                 "the imaginary part is known to vanish)")
        b = b.astype(dtype)
    return xp.ascontiguousarray(b)


def csr_args(A, dtype=None) -> tuple:
    """(n, indptr, indices, data) contiguous arrays for the C++ side."""
    A = sp.csr_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise OptionError(f"square matrix required, got {A.shape}", fix='a square matrix')
    A.sum_duplicates()
    A.sort_indices()
    dt = np.complex128 if _kind(dtype or A.dtype) == "Z" else np.float64
    return (A.shape[0], np.ascontiguousarray(A.indptr, dtype=np.int32),
            np.ascontiguousarray(A.indices, dtype=np.int32), np.ascontiguousarray(A.data, dtype=dt))


class BorderedBanded:
    """Factor A once; ``border_rows`` are the rows that break the band (boundary conditions)."""

    def __init__(self, A, border_rows: Sequence[int], dtype=None, col_border: bool = False):
        self.dtype = np.complex128 if _kind(dtype or sp.csr_matrix(A).dtype) == "Z" else np.float64
        self.border_rows = [int(i) for i in border_rows]
        cls = _core.BorderedBandedZ if self.dtype == np.complex128 else _core.BorderedBandedD
        self._s = cls(*csr_args(A, self.dtype), self.border_rows, bool(col_border))

    n = property(lambda self: self._s.n)
    kl = property(lambda self: self._s.kl)
    ku = property(lambda self: self._s.ku)
    rank = property(lambda self: self._s.rank)
    capacitance_cond = property(lambda self: self._s.capacitance_cond)
    col_border = property(lambda self: self._s.col_border)

    def solve(self, b: np.ndarray, transpose: bool = False) -> np.ndarray:
        """Solve A x = b (or A^T x = b, no conjugation). b: (n,) or (n, nrhs)."""
        b = solve_operand(b, self.dtype, "BorderedBanded.solve")
        cols = b.reshape(self.n, -1).T.copy()          # (nrhs, n) C-order = column vectors
        x = self._s.solve(cols, transpose).T
        return x.reshape(b.shape)

    def explain(self) -> str:
        return (f"BorderedBanded(n={self.n}, band kl={self.kl} ku={self.ku}, border rank={self.rank}, "
                f"capacitance cond_1={self.capacitance_cond:.3g}, dtype={np.dtype(self.dtype).name}); "
                "LAPACK gbtrf + Woodbury/Schur on the boundary rows"
                + (f" and a column border on the first {self.rank} columns ({self._s.col_border_nnz} "
                   "entries out of the band, exact)" if self.col_border else ""))


class BorderedBandedBatch:
    """One factorization per mode/group (same size n), solves looped in C++."""

    def __init__(self, mats: Sequence, border_rows: Sequence[int], dtype=None, col_border: bool = False):
        dtype = dtype or sp.csr_matrix(mats[0]).dtype
        self.dtype = np.complex128 if _kind(dtype) == "Z" else np.float64
        self.col_border = bool(col_border)
        cls = _core.BorderedBandedBatchZ if self.dtype == np.complex128 else _core.BorderedBandedBatchD
        self._s = cls([csr_args(A, self.dtype) for A in mats], [int(i) for i in border_rows],
                      self.col_border)

    kl = property(lambda self: self._s.kl)        # common band (max over the systems)
    ku = property(lambda self: self._s.ku)

    def __len__(self) -> int:
        return len(self._s)

    def explain(self) -> str:
        kind = "complex" if self.dtype == np.complex128 else "real"
        return (f"BorderedBandedBatch: {len(self)} {kind} bordered-banded systems (one per mode/group), "
                "each factorized once: boundary rows first, LAPACK gbtrf on the band "
                f"(kl={self.kl}, ku={self.ku}), Woodbury/Schur on the border"
                + (" (rows and columns: column border, exact)" if self.col_border else "")
                + "; solves looped in C++")

    def solve(self, b: np.ndarray, transpose: bool = False) -> np.ndarray:
        """b: (batch, n) -> x: (batch, n); ``transpose=True`` solves A_k^T x = b_k (plain
        transpose, no conjugation, same factors: the discrete adjoint path)."""
        return self._s.solve(solve_operand(b, self.dtype, "BorderedBandedBatch.solve"), bool(transpose))
