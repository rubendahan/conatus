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
from ..errors import DtypeError, InputTypeError, NumericalError, OptionError

__all__ = ["BorderedBanded", "BorderedBandedBatch", "csr_args", "solve_operand"]


def _kind(dtype) -> str:
    return "Z" if np.issubdtype(np.dtype(dtype), np.complexfloating) else "D"


def _factor_dtype(mats: Sequence, dtype=None):
    """complex128 if ``dtype`` (when given) or any of ``mats`` is complex, else float64. Every
    matrix counts (review finding 38: the batch looked at its first matrix only)."""
    if dtype is not None:
        return np.complex128 if _kind(dtype) == "Z" else np.float64
    return np.complex128 if any(_kind(m.dtype) == "Z" for m in mats) else np.float64


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
            raise DtypeError(f"{where}: right-hand side of dtype {b.dtype} for {dtype.name} factors",
                             why="the conversion would lose information (e.g. the imaginary part)",
                             fix=f"pass a {dtype.name} right-hand side (take .real explicitly if "
                                 "the imaginary part is known to vanish)")
        b = b.astype(dtype)
    return xp.ascontiguousarray(b)


def csr_args(A, dtype=None, where: str = "csr_args") -> tuple:
    """(n, indptr, indices, data) contiguous arrays for the C++ side. Refused: a non-square or
    non-finite matrix (a NaN/Inf entry factored with cond_1 = 0, review finding 41) and complex
    entries with a non-zero imaginary part for real factors (``dtype`` real: the cast dropped it,
    finding 39). A complex-typed matrix whose entries are all real converts exactly."""
    A = sp.csr_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise OptionError(f"square matrix required, got {A.shape}", fix='a square matrix')
    A.sum_duplicates()
    A.sort_indices()
    if not np.all(np.isfinite(A.data)):
        k = int(np.flatnonzero(~np.isfinite(A.data))[0])
        row = int(np.searchsorted(A.indptr, k, side="right") - 1)
        raise NumericalError(f"{where}: non-finite matrix entry {A.data[k]!r} at "
                             f"({row}, {int(A.indices[k])})",
                             why="a factorization of a matrix with NaN/Inf has no meaning (it used "
                                 "to report a capacitance condition number of 0)",
                             fix="find where the NaN/Inf enters the operator (a coefficient, a "
                                 "parameter, a division by zero in a non-constant coefficient)")
    dt = np.complex128 if _kind(dtype or A.dtype) == "Z" else np.float64
    if dt == np.float64 and np.iscomplexobj(A.data) and np.any(A.data.imag != 0):
        raise DtypeError(f"{where}: complex matrix for real (float64) factors",
                         why="the conversion would drop the imaginary part of the entries",
                         fix="pass dtype=np.complex128 (or no dtype: it follows the matrices)")
    return (A.shape[0], np.ascontiguousarray(A.indptr, dtype=np.int32),
            np.ascontiguousarray(A.indices, dtype=np.int32),
            np.ascontiguousarray(A.data.real if dt == np.float64 else A.data, dtype=dt))


def _check_rhs_shape(b, expected: tuple, where: str, what: str) -> None:
    """Refuse a right-hand side whose shape is not ``expected`` (None = any extent); a reshape
    used to reinterpret a wrong shape as another valid one (review finding 40)."""
    ok = b.ndim == len(expected) and all(e is None or s == e for s, e in zip(b.shape, expected, strict=True))
    if not ok:
        raise InputTypeError(f"{where}: right-hand side of shape {tuple(b.shape)}, expected {what}",
                             why="any other shape would be reinterpreted, not solved",
                             fix=f"pass an array of shape {what}")


class BorderedBanded:
    """Factor A once; ``border_rows`` are the rows that break the band (boundary conditions)."""

    def __init__(self, A, border_rows: Sequence[int], dtype=None, col_border: bool = False):
        A = sp.csr_matrix(A)
        self.dtype = _factor_dtype([A], dtype)
        self.border_rows = [int(i) for i in border_rows]
        cls = _core.BorderedBandedZ if self.dtype == np.complex128 else _core.BorderedBandedD
        self._s = cls(*csr_args(A, self.dtype, "BorderedBanded"), self.border_rows, bool(col_border))

    n = property(lambda self: self._s.n)
    kl = property(lambda self: self._s.kl)
    ku = property(lambda self: self._s.ku)
    rank = property(lambda self: self._s.rank)
    capacitance_cond = property(lambda self: self._s.capacitance_cond)
    col_border = property(lambda self: self._s.col_border)

    def solve(self, b: np.ndarray, transpose: bool = False) -> np.ndarray:
        """Solve A x = b (or A^T x = b, no conjugation). b: (n,) or (n, nrhs)."""
        b = solve_operand(b, self.dtype, "BorderedBanded.solve")
        if b.ndim == 1:
            _check_rhs_shape(b, (self.n,), "BorderedBanded.solve", f"({self.n},) or ({self.n}, nrhs)")
        else:
            _check_rhs_shape(b, (self.n, None), "BorderedBanded.solve",
                             f"({self.n},) or ({self.n}, nrhs)")
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
        mats = [sp.csr_matrix(A) for A in mats]
        if not mats:
            raise InputTypeError("BorderedBandedBatch: no matrix given",
                                 fix="pass at least one matrix (one per mode/group)")
        self.dtype = _factor_dtype(mats, dtype)
        self.col_border = bool(col_border)
        cls = _core.BorderedBandedBatchZ if self.dtype == np.complex128 else _core.BorderedBandedBatchD
        self._s = cls([csr_args(A, self.dtype, f"BorderedBandedBatch (matrix {k})")
                       for k, A in enumerate(mats)], [int(i) for i in border_rows], self.col_border)
        self.n = mats[0].shape[0]

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
        b = solve_operand(b, self.dtype, "BorderedBandedBatch.solve")
        _check_rhs_shape(b, (len(self), self.n), "BorderedBandedBatch.solve", f"({len(self)}, {self.n})")
        return self._s.solve(b, bool(transpose))
