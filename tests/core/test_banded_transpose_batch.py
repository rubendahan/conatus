"""Batched transposed bordered-banded solves (discrete adjoint path, D8/): CPU batch vs the
single-system transpose and vs dense, and the GPU batch vs the CPU batch. Plain transpose, no
conjugation, same factors as the direct solve."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.runtime import BorderedBanded, BorderedBandedBatch


def _bordered(n, kl, ku, r, rng, dtype):
    """Random banded matrix with r dense boundary rows on top (as the tau systems)."""
    A = sp.diags([rng.standard_normal(n - abs(k)) for k in range(-kl, ku + 1)], list(range(-kl, ku + 1)),
                 shape=(n, n)).toarray()
    A += 4 * np.eye(n)
    A[:r] = rng.standard_normal((r, n))
    if np.dtype(dtype).kind == "c":
        A = A + 1j * sp.diags([rng.standard_normal(n - abs(k)) for k in range(-kl, ku + 1)],
                              list(range(-kl, ku + 1)), shape=(n, n)).toarray()
        A[:r] += 1j * rng.standard_normal((r, n))
    return A.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("r", [0, 2, 4])
def test_batch_transpose_matches_single_and_dense(dtype, r):
    rng = np.random.default_rng(7)
    n, B = 40, 5
    mats = [_bordered(n, 3, 5, r, rng, dtype) for _ in range(B)]
    rhs = rng.standard_normal((B, n)).astype(dtype)
    batch = BorderedBandedBatch([sp.csr_matrix(A) for A in mats], list(range(r)), dtype=dtype)
    xb = batch.solve(rhs, transpose=True)
    for k, A in enumerate(mats):
        single = BorderedBanded(sp.csr_matrix(A), list(range(r)), dtype=dtype).solve(rhs[k], transpose=True)
        dense = np.linalg.solve(A.T, rhs[k])
        assert np.abs(xb[k] - single).max() <= 1e-13 * np.abs(single).max()
        assert np.abs(xb[k] - dense).max() <= 1e-11 * np.abs(dense).max()
    # the direct batch solve is unchanged
    xd = batch.solve(rhs)
    assert np.abs(xd[0] - np.linalg.solve(mats[0], rhs[0])).max() <= 1e-11 * np.abs(xd[0]).max()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("r", [0, 3])
def test_gpu_batch_transpose_matches_cpu(dtype, r):
    from conatus.runtime import cuda
    if not cuda.available():
        pytest.skip("no CUDA device on this node")
    rng = np.random.default_rng(11)
    n, B = 64, 7
    mats = [_bordered(n, 4, 6, r, rng, dtype) for _ in range(B)]
    rhs = rng.standard_normal((B, n)).astype(dtype)
    ref = BorderedBandedBatch([sp.csr_matrix(A) for A in mats], list(range(r)), dtype=dtype).solve(
        rhs, transpose=True)
    gb = cuda.BandedBatch([sp.csr_matrix(A) for A in mats], list(range(r)), dtype=dtype)
    d = cuda.to_device(rhs)
    gb.solve_transpose(d)
    x = d.to_host()
    assert np.abs(x - ref).max() <= 1e-12 * np.abs(ref).max()
