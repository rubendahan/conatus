"""Level-scheduled GPU solves: same LU factors and same per-row arithmetic as the band
kernel, factors streamed without their structural zeros. Checked against the band kernel (same
factors: agreement at rounding), the host solver and the residual; fused gather/scatter maps.
Merged levels (several levels per warp step, rows expanded on the host 512): equal to the
unmerged solve to rounding, fewer steps, bitwise reproducible."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.runtime import BorderedBanded
from conatus.runtime import cuda

rng = np.random.default_rng(11)


def rel(a, b):
    return np.max(abs(a - b)) / max(np.max(abs(b)), 1e-300)


def sparse_banded(n, kl, ku, r, dtype, density=0.4):
    """Almost-banded matrix with a sparse band (structural zeros inside it, like the spectral
    operators) and r dense border rows; diagonal kept non-zero, pivoting still exercised."""
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            if j == i or rng.random() < density:
                A[i, j] = rng.standard_normal() + (1j * rng.standard_normal() if dtype == complex else 0)
    for i in range(r):
        A[i] = rng.standard_normal(n) + (1j * rng.standard_normal(n) if dtype == complex else 0)
    return A


def conditioned(n, kl, ku, r, shift=8.0, dtype=float):
    """sparse_banded with its banded rows made diagonally dominant: condition number ~1e3-1e4.
    The plain random matrices reach cond 1e11-1e17 at n = 900 (host LAPACK solve residuals up
    to 0.7, measured): a residual bound on them tests the matrix, not the solver ( 512)."""
    A = sparse_banded(n, kl, ku, r, dtype)
    A[np.arange(r, n), np.arange(r, n)] += shift
    return sp.csr_matrix(A)


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("r", [0, 2, 6])
@pytest.mark.parametrize("n", [40, 300])
def test_level_matches_band_kernel_and_host(dtype, r, n, merge):
    batch = 7
    kl, ku = 9, 14
    mats = [sp.csr_matrix(sparse_banded(n, kl, ku, r, dtype)) for _ in range(batch)]
    border = list(range(r))
    lvl = cuda.BandedBatch(mats, border, level_schedule=True, level_merge=merge)
    band = cuda.BandedBatch(mats, border, level_schedule=False)
    assert lvl.level_scheduled, lvl.level_note
    assert not band.level_scheduled
    assert lvl.level_merged == merge
    if not merge:
        assert lvl.level_steps == lvl.level_depth
    else:                                         # merging never adds steps, and here removes some
        assert all(s <= d for s, d in zip(lvl.level_steps, lvl.level_depth))
        assert sum(lvl.level_steps) < sum(lvl.level_depth)
    b = rng.standard_normal((batch, n)) + (1j * rng.standard_normal((batch, n)) if dtype == complex else 0)
    d1, d2 = cuda.to_device(b), cuda.to_device(b)
    lvl.solve(d1)
    band.solve(d2)
    x1, x2 = d1.to_host(), d2.to_host()
    if r == 0 and not merge:
        assert rel(x1, x2) < 1e-13                # same factors, same arithmetic per row
    if merge:                                     # replays are bitwise identical (fixed order)
        d3 = cuda.to_device(b)
        lvl.solve(d3)
        assert np.array_equal(d3.to_host(), x1)
    for k, A in enumerate(mats):
        x_cpu = BorderedBanded(A, border).solve(b[k])
        res_cpu = np.max(abs(A @ x_cpu - b[k]))
        tol = 10 * res_cpu + 1e-13 * n * np.max(abs(b[k]))
        assert np.max(abs(A @ x1[k] - b[k])) < tol       # level-scheduled kernel
        assert np.max(abs(A @ x2[k] - b[k])) < tol       # band kernel
    assert "level-scheduled" in lvl.explain()
    assert ("merged into" in lvl.explain()) == merge


@pytest.mark.parametrize("dtype", [float, complex])
def test_merged_levels_equal_unmerged_to_rounding(dtype):
    """Merged and unmerged level solves of the same factors: rounding-level agreement, measured
    against the band kernel's own distance to the host solve (both well-conditioned here)."""
    n, batch, kl, ku, r = 400, 5, 6, 10, 3
    mats = []
    for _ in range(batch):
        A = sparse_banded(n, kl, ku, r, dtype, density=0.3)
        A[np.arange(r, n), np.arange(r, n)] += 4.0        # well conditioned: rounding is visible
        mats.append(sp.csr_matrix(A))
    border = list(range(r))
    b = rng.standard_normal((batch, n)) + (1j * rng.standard_normal((batch, n)) if dtype == complex else 0)
    xs = {}
    for merge in (False, True):
        bb = cuda.BandedBatch(mats, border, level_merge=merge)
        d = cuda.to_device(b)
        bb.solve(d)
        xs[merge] = d.to_host()
    assert rel(xs[True], xs[False]) < 1e-13


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_solve_mapped_gather_scatter(dtype):
    import cupy as cp
    n, batch, r = 120, 5, 2
    cplx = dtype == np.complex128
    # well-conditioned systems: the absolute residual bound below is about the maps, not about
    # the conditioning of a random matrix (a plain sparse_banded draw failed it at 1.6e-10)
    mats = [conditioned(n, 4, 6, r, dtype=complex if cplx else float) for _ in range(batch)]
    for level in (True, False):
        bb = cuda.BandedBatch(mats, list(range(r)), level_schedule=level)
        N = batch * n + 17
        rhs = rng.standard_normal(N) + (1j * rng.standard_normal(N) if cplx else 0)
        rows = rng.permutation(N)[: batch * n].reshape(batch, n)
        cols = rng.permutation(N)[: batch * n].reshape(batch, n)
        x = cp.full(N, 7.0, dtype=dtype)
        bb.solve_mapped(cp.asarray(rhs.astype(dtype)), cp.asarray(rows), x, cp.asarray(cols))
        xh = cp.asnumpy(x)
        for k, A in enumerate(mats):
            xk = xh[cols[k]]
            assert np.max(abs(A @ xk - rhs[rows[k]])) < 1e-10
        untouched = np.setdiff1d(np.arange(N), cols.ravel())
        assert np.all(xh[untouched] == 7.0)


@pytest.mark.parametrize("complexify", [False, True])
def test_rb2d_system_level_vs_band(complexify):
    import sys
    from pathlib import Path
    import cupy as cp
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ivp"))
    from ivp_cases import rayleigh_benard_2d
    from conatus.timestep.device import DeviceSystem
    from conatus.timestep.system import GlobalSystem
    plan, _ = rayleigh_benard_2d(32, 24, Ra=1e6, Lx=4.0)
    host = GlobalSystem(plan, complexify=complexify)
    a, b = 1.0, 0.37e-3
    rhs = rng.standard_normal(host.n)
    xs = []
    for level, merge in ((True, True), (False, True), (True, False)):
        dev = DeviceSystem(host, level_schedule=level, level_merge=merge)
        x = cp.asnumpy(dev.solve(a, b, cp.asarray(rhs)))
        for bb in dev.factor(a, b):
            assert bb.level_scheduled == level, bb.level_note
            if level and merge:                   # spectral operators: several levels per step
                assert sum(bb.level_steps) < sum(bb.level_depth)
        xs.append(x)
    x_host = host.solve(a, b, rhs)
    A = (a * host.M + b * host.L).tocsr()
    assert rel(xs[0], xs[1]) < 1e-12
    assert rel(xs[2], xs[1]) < 1e-12
    assert rel(xs[0], x_host) < 1e-11
    res_host = np.linalg.norm(A @ x_host - rhs)
    for x in xs:                                  # level and band kernels: host residual level
        assert np.linalg.norm(A @ x - rhs) <= 10 * res_host + 1e-14 * np.linalg.norm(rhs)


def _solve_rel_residual(bb, mats, b):
    d = cuda.to_device(b)
    bb.solve(d)
    x = d.to_host()
    return max(np.linalg.norm(A @ x[k] - b[k]) / np.linalg.norm(b[k]) for k, A in enumerate(mats))




def test_level_two_sizes_in_one_process():
    """Regression: the dynamic shared-memory limit of the level kernel is a
    per-kernel attribute; a second BandedBatch with a smaller n used to lower it below the first
    one's need, whose next solve then failed to launch. Well-conditioned matrices (see
    conditioned): the first version used plain random ones and failed on their conditioning."""
    big = [conditioned(900, 6, 8, 2) for _ in range(3)]
    small = [conditioned(24, 3, 4, 2) for _ in range(3)]
    A = cuda.BandedBatch(big, [0, 1])
    B = cuda.BandedBatch(small, [0, 1])
    assert A.level_scheduled and B.level_scheduled, (A.level_note, B.level_note)
    b_big = rng.standard_normal((3, 900))
    b_small = rng.standard_normal((3, 24))
    assert _solve_rel_residual(A, big, b_big) < 1e-12         # launch must not fail
    assert _solve_rel_residual(B, small, b_small) < 1e-12
    band = cuda.BandedBatch(big, [0, 1], level_schedule=False)    # same factors, other kernel
    d1, d2 = cuda.to_device(b_big), cuda.to_device(b_big)
    A.solve(d1)
    band.solve(d2)
    assert rel(d1.to_host(), d2.to_host()) < 1e-13


def test_level_falls_back_when_static_shared_memory_is_counted():
    """Regression: n just under the old check (dynamic + MAXR words only) but over
    the real limit (the kernel's static shared arrays count too) must fall back to the band
    kernel with a reason, not raise from the constructor."""
    import cupy as cp
    optin = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
    ring = 256 * 4 * 8                                    # LVL_RING words of 8 bytes
    n = (optin - ring - 64 * 8) // 8 - 8                  # passed the old check by 64 bytes
    A = sp.diags([np.full(n - 1, 0.3), np.full(n, 4.0), np.full(n - 1, -0.2)], [-1, 0, 1]).tocsr()
    bb = cuda.BandedBatch([A], [], level_schedule=True)
    assert not bb.level_scheduled
    assert "static" in bb.level_note and "band kernel" in bb.explain()
    b = rng.standard_normal((1, n))
    assert _solve_rel_residual(bb, [A], b) < 1e-12


def test_solve_mapped_refuses_maps_it_would_misread():
    """The kernel reads int64 maps through raw pointers: int32 maps, a wrong shape or a strided
    right-hand side used to be read as other memory without an error."""
    import cupy as cp
    n, batch = 30, 3
    mats = [sp.csr_matrix(sparse_banded(n, 3, 4, 1, float)) for _ in range(batch)]
    bb = cuda.BandedBatch(mats, [0])
    rhs, x = cp.zeros(batch * n), cp.zeros(batch * n)
    ok = cp.arange(batch * n, dtype=cp.int64).reshape(batch, n)
    with pytest.raises(ValueError, match="int64"):
        bb.solve_mapped(rhs, ok.astype(cp.int32), x, ok)
    with pytest.raises(ValueError, match="int64"):
        bb.solve_mapped(rhs, ok, x, ok.reshape(n, batch))
    with pytest.raises(ValueError, match="contiguous"):
        bb.solve_mapped(cp.zeros(2 * batch * n)[::2], ok, x, ok)
    bb.solve_mapped(rhs, ok, x, None)                 # the valid forms still run
    bb.solve_mapped(rhs, None, x, ok)


# --- edge structures of the merged levels ( 512 review) --------------------------------------
# Own generator: the tests above keep their draws (a shared generator moved their matrices).
_rng_edge = np.random.default_rng(5)


def _rnd(shape, dtype):
    r = _rng_edge.standard_normal(shape)
    return r + 1j * _rng_edge.standard_normal(shape) if dtype == complex else r


def _band(n, kl, ku, r, dtype, density=1.0, shift=0.0):
    """Banded rows (dense or sparse band) with a diagonal shift, r dense border rows."""
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            if j == i or _rng_edge.random() < density:
                A[i, j] = _rnd((), dtype)
    A[np.arange(n), np.arange(n)] += shift
    for i in range(r):
        A[i] = _rnd(n, dtype)
    return A


def _three_kernels_agree(mats, r, dtype, tol_rel=1e-12):
    """Band kernel, one level per step and merged levels against the host solve; merged replays
    bitwise identical (a missing read-before-write barrier would show up as varying results)."""
    border, batch, n = list(range(r)), len(mats), mats[0].shape[0]
    b = _rnd((batch, n), dtype)
    out, merged = {}, None
    for name, kw in (("band", dict(level_schedule=False)), ("level", dict(level_merge=False)),
                     ("merged", dict(level_merge=True))):
        bb = cuda.BandedBatch([sp.csr_matrix(A) for A in mats], border, dtype=dtype, **kw)
        assert name == "band" or bb.level_scheduled, bb.level_note
        d = cuda.to_device(b)
        bb.solve(d)
        out[name] = d.to_host()
        if name == "merged":
            merged = bb
            for _ in range(10):
                d = cuda.to_device(b)
                bb.solve(d)
                assert np.array_equal(d.to_host(), out[name])
    for k, A in enumerate(mats):
        x = BorderedBanded(sp.csr_matrix(A), border).solve(b[k])
        for name, xs in out.items():
            assert rel(xs[k], x) < tol_rel, (name, k, rel(xs[k], x))
    return merged


@pytest.mark.parametrize("dtype", [float, complex])
def test_merged_heterogeneous_batch(dtype):
    """One batch, four very different level structures: no L and one wide U level (diagonal
    band), chains of one row per level (tridiagonal), sparse and dense bands."""
    n, r = 200, 3
    mats = [_band(n, 0, 0, r, dtype, shift=3.0), _band(n, 1, 1, r, dtype, shift=4.0),
            _band(n, 6, 9, r, dtype, density=0.3, shift=5.0), _band(n, 6, 9, r, dtype, shift=8.0)]
    bb = _three_kernels_agree(mats, r, dtype)
    assert sum(bb.level_steps) < sum(bb.level_depth)
    # the diagonal mode's backward level has 200 rows: a step of its own, declared as such
    assert "a wider level stays a step of its own" in bb.explain()


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("n, r", [(1, 0), (2, 0), (2, 1), (3, 1), (5, 0), (5, 1)])
def test_merged_tiny_systems(n, r, dtype):
    _three_kernels_agree([_band(n, 1, 1, r, dtype, shift=3.0) for _ in range(3)], r, dtype)


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("r", [15, 16, 17, 24])
def test_level_border_rank_around_shared_capacitance(r, dtype):
    """r <= 16: capacitance LU in shared memory, solved one lane per entry, W loaded r entries at
    a time; r > 16: serial path from global memory. Both sides of the switch."""
    mats = [_band(160, 4, 6, r, dtype, density=0.5, shift=6.0) for _ in range(3)]
    _three_kernels_agree(mats, r, dtype, tol_rel=1e-10)


@pytest.mark.parametrize("dtype", [float, complex])
def test_merged_wide_level_then_chain(dtype):
    """100 independent rows (a level of > 32 rows, never merged), then chains that merge."""
    n, mats = 180, []
    for _ in range(3):
        A = np.diag(3.0 + np.abs(_rnd(n, float))).astype(dtype)
        for i in range(100, n):
            A[i, i - 1] = _rnd((), dtype)
            A[i, i - 50] = 0.5 * _rnd((), dtype)
            if i + 2 < n:
                A[i, i + 2] = 0.5 * _rnd((), dtype)
        mats.append(A)
    _three_kernels_agree(mats, 0, dtype)


@pytest.mark.parametrize("dtype", [float, complex])
def test_merged_wide_band_near_ring_capacity(dtype):
    """Dense band kl = 30, ku = 40: merged rows grow until a step would not fit the ring chunk."""
    _three_kernels_agree([_band(300, 30, 40, 2, dtype, shift=20.0) for _ in range(2)], 2, dtype,
                         tol_rel=1e-10)
