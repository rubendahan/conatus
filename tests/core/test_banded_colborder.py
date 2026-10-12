"""Column border of the bordered banded solver: the first r columns may be dense
(lifted gradient tau of the spherical shell); with ``col_border=True`` they leave the band and are
solved exactly around the band factors (core/include/conatus/banded.hpp, "Column border").
Checked against the dense solve and against the plain row-border layout (same matrix), forward
and transposed, real and complex, single and batched."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.runtime import BorderedBanded, BorderedBandedBatch

rng = np.random.default_rng(7)


def bordered_with_dense_columns(n, kl, ku, r, dtype, dense_cols=None):
    """Banded core (diagonally dominant), r dense border rows, and dense entries in the first r
    columns over the whole bulk (``dense_cols`` of them; default all r)."""
    cplx = np.dtype(dtype).kind == "c"
    rnd = (lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)) if cplx else \
        (lambda *s: rng.standard_normal(s))
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            if j == i or rng.random() < 0.5:
                A[i, j] = rnd(1)[0]
    A[np.arange(r, n), np.arange(r, n)] += 2.0 * (kl + ku + 2)
    A[:r] = rnd(r, n)
    for k in range(r if dense_cols is None else dense_cols):
        rows = r + np.nonzero(rng.random(n - r) < 0.5)[0]
        A[rows, k] = rnd(rows.size)
    return A


def rel(a, b):
    return float(np.max(abs(a - b)) / max(np.max(abs(b)), 1e-300))


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("r", [1, 3, 9])
def test_colborder_matches_dense_and_row_border(dtype, r):
    n, kl, ku = 160, 6, 9
    A = bordered_with_dense_columns(n, kl, ku, r, dtype)
    border = list(range(r))
    s_col = BorderedBanded(sp.csr_matrix(A), border, col_border=True)
    s_row = BorderedBanded(sp.csr_matrix(A), border)            # dense columns kept in the band
    assert s_col.col_border and not s_row.col_border
    assert s_col.kl <= kl and s_col.ku <= ku                     # band of the core only
    assert s_row.kl > n // 2                                     # what the column border removes
    b = rng.standard_normal((n, 4)) + (1j * rng.standard_normal((n, 4)) if dtype is complex else 0)
    x_ref = np.linalg.solve(A, b)
    xt_ref = np.linalg.solve(A.T, b)
    x, xt = s_col.solve(b), s_col.solve(b, transpose=True)
    assert rel(x, x_ref) < 1e-12 and rel(xt, xt_ref) < 1e-12
    assert rel(x, s_row.solve(b)) < 1e-12
    assert rel(xt, s_row.solve(b, transpose=True)) < 1e-12
    assert "column border" in s_col.explain()


def test_colborder_partial_columns_and_identity_without_dense():
    """Only some of the r pinned columns dense; and a matrix without any dense column gives the
    row-border result to rounding (the column border then only holds in-band pinned entries)."""
    n, r = 120, 4
    for dense in (2, 0):
        A = bordered_with_dense_columns(n, 3, 5, r, float, dense_cols=dense)
        s = BorderedBanded(sp.csr_matrix(A), list(range(r)), col_border=True)
        b = rng.standard_normal(n)
        assert rel(s.solve(b), np.linalg.solve(A, b)) < 1e-12
        assert rel(s.solve(b, transpose=True), np.linalg.solve(A.T, b)) < 1e-12


@pytest.mark.parametrize("dtype", [float, complex])
def test_colborder_batch(dtype):
    n, r, batch = 90, 5, 6
    mats = [bordered_with_dense_columns(n, 4, 7, r, dtype) for _ in range(batch)]
    bb = BorderedBandedBatch([sp.csr_matrix(A) for A in mats], list(range(r)), col_border=True)
    assert bb.col_border and bb.kl <= 4 and bb.ku <= 7
    b = rng.standard_normal((batch, n)).astype(dtype)
    x, xt = bb.solve(b), bb.solve(b, transpose=True)
    for k, A in enumerate(mats):
        assert rel(x[k], np.linalg.solve(A, b[k])) < 1e-12
        assert rel(xt[k], np.linalg.solve(A.T, b[k])) < 1e-12
    assert "column border" in bb.explain()


@pytest.mark.parametrize("dtype", [float, complex])
def test_colborder_heterogeneous_batch(dtype):
    """Modes of one batch with different column borders (none, one dense column, all) and
    different bands: per-mode offsets of X and the common band (B2 review)."""
    n, r = 70, 4
    mats = [bordered_with_dense_columns(n, 2, 3, r, dtype, dense_cols=0),
            bordered_with_dense_columns(n, 6, 2, r, dtype, dense_cols=1),
            bordered_with_dense_columns(n, 3, 8, r, dtype)]
    bb = BorderedBandedBatch([sp.csr_matrix(A) for A in mats], list(range(r)), col_border=True)
    assert bb.kl <= 6 and bb.ku <= 8
    b = rng.standard_normal((3, n)) + (1j * rng.standard_normal((3, n)) if dtype is complex else 0)
    x, xt = bb.solve(b), bb.solve(b, transpose=True)
    for k, A in enumerate(mats):
        assert rel(x[k], np.linalg.solve(A, b[k])) < 1e-12
        assert rel(xt[k], np.linalg.solve(A.T, b[k])) < 1e-12
