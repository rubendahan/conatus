"""Column border on the GPU: batched bordered banded solves whose first r columns
are dense (core/include/conatus/banded.hpp, "Column border"). The device result equals the host
solver and the dense solve to rounding, for the band kernel and the level-scheduled kernel
(merged or not), forward, transposed and with fused gather/scatter maps."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.runtime import BorderedBandedBatch
from conatus.runtime import cuda

rng = np.random.default_rng(23)


def rel(a, b):
    return float(np.max(abs(a - b)) / max(np.max(abs(b)), 1e-300))


def colborder_matrix(n, kl, ku, r, dtype):
    """Sparse band (diagonally dominant core), r dense border rows, dense first r columns."""
    cplx = dtype is complex
    rnd = (lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)) if cplx else \
        (lambda *s: rng.standard_normal(s))
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            if j == i or rng.random() < 0.4:
                A[i, j] = rnd(1)[0]
    A[np.arange(r, n), np.arange(r, n)] += 2.0 * (kl + ku + 2)
    A[:r] = rnd(r, n)
    for k in range(r):
        rows = r + np.nonzero(rng.random(n - r) < 0.5)[0]
        A[rows, k] = rnd(rows.size)
    return A


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("r", [3, 9, 40])
@pytest.mark.parametrize("level,merge", [(False, False), (True, False), (True, True)])
def test_gpu_colborder_equals_host_and_dense(dtype, r, level, merge):
    n, batch, kl, ku = 260, 6, 9, 14
    dense = [colborder_matrix(n, kl, ku, r, dtype) for _ in range(batch)]
    mats = [sp.csr_matrix(A) for A in dense]
    border = list(range(r))
    gpu = cuda.BandedBatch(mats, border, level_schedule=level, level_merge=merge, col_border=True)
    cpu = BorderedBandedBatch(mats, border, col_border=True)
    assert gpu.col_border and gpu.kl <= kl and gpu.ku <= ku
    assert gpu.level_scheduled == level, gpu.level_note
    b = rng.standard_normal((batch, n)) + (1j * rng.standard_normal((batch, n)) if dtype is complex else 0)
    d, dt = cuda.to_device(b), cuda.to_device(b)
    gpu.solve(d)
    gpu.solve_transpose(dt)
    x, xt = d.to_host(), dt.to_host()
    xc, xtc = cpu.solve(b), cpu.solve(b, transpose=True)
    for k, A in enumerate(dense):
        assert rel(x[k], np.linalg.solve(A, b[k])) < 1e-12
        assert rel(xt[k], np.linalg.solve(A.T, b[k])) < 1e-12
    assert rel(x, xc) < 1e-13 and rel(xt, xtc) < 1e-13
    assert "column-border" in gpu.explain()


def test_gpu_colborder_mapped():
    import cupy as cp
    n, batch, r = 150, 4, 5
    mats = [sp.csr_matrix(colborder_matrix(n, 4, 6, r, complex)) for _ in range(batch)]
    cpu = BorderedBandedBatch(mats, list(range(r)), col_border=True)
    for level in (True, False):
        bb = cuda.BandedBatch(mats, list(range(r)), level_schedule=level, col_border=True)
        N = batch * n + 11
        rhs = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        rows = rng.permutation(N)[: batch * n].reshape(batch, n)
        cols = rng.permutation(N)[: batch * n].reshape(batch, n)
        x = cp.zeros(N, dtype=np.complex128)
        bb.solve_mapped(cp.asarray(rhs), cp.asarray(rows), x, cp.asarray(cols))
        xh = cp.asnumpy(x)
        ref = cpu.solve(rhs[rows])
        assert rel(xh[cols], ref) < 1e-13
