"""Review of the v05b CUDA fixes: attempts to break the device IMEX history
check, the stream-ordered work vector of the band kernel (two streams, CUDA graph capture, leaks)
and the mapped-solve index checks. Each test states its oracle: a fresh integrator, a serial
out-of-place solve, or the device memory before the loop. Skipped without a CUDA device."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.runtime import cuda

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not cuda.available(), reason="no CUDA device on this node")]


def _heat_like(N, kappa, dtype=float):
    """(M, L, border) of u_t - u_xx + kappa u + taus = F on Chebyshev, u(+-1) = 0 (tau form);
    the system of tests/core/test_cuda_risks_gpu.py, optionally complex (kappa complex)."""
    from conatus.discretize.ultraspherical import conversion_chain, differentiation
    S = conversion_chain(N, 0, 2)
    D2 = differentiation(N, 0, 2)
    n = N + 2
    M = sp.lil_matrix((n, n), dtype=dtype)
    L = sp.lil_matrix((n, n), dtype=dtype)
    M[:N, :N] = S
    L[:N, :N] = -D2 + kappa * S
    L[N - 1, N] = 1.0
    L[N - 2, N + 1] = 1.0
    L[N, :N] = (-1.0) ** np.arange(N)
    L[N + 1, :N] = 1.0
    return M.tocsr(), L.tocsr(), [N, N + 1]


def _imex(scheme, systems, dt, g=None, h=None):
    return cuda.Imex([s[0] for s in systems], [s[1] for s in systems], systems[0][2], scheme, dt, g=g, h=h)


def _state(rng, batch, N, complex_=False):
    x = np.zeros((batch, N + 2), dtype=complex if complex_ else float)
    x[:, :N] = rng.standard_normal((batch, N)) * np.exp(-0.3 * np.arange(N))
    if complex_:
        x[:, :N] += 1j * rng.standard_normal((batch, N)) * np.exp(-0.3 * np.arange(N))
    return x


@pytest.mark.parametrize("scheme", ["SBDF2", "SBDF3", "SBDF4"])
def test_device_imex_continuation_through_a_copy_and_refusals(scheme, rng):
    """The continuation check compares bits, not buffers: the returned state copied to another
    device array continues the history (oracle: one run of the summed steps). The right state at
    a wrong time, and a perturbed state at the right time, are refused, and a refused call leaves
    the history usable."""
    N, batch, dt = 24, 3, 2e-3
    systems = [_heat_like(N, k) for k in (0.0, 1.0, 2.0)]
    x0 = _state(rng, batch, N)
    one = cuda.to_device(x0)
    _imex(scheme, systems, dt).run(one, 0.0, 30)
    im = _imex(scheme, systems, dt)
    a = cuda.to_device(x0)
    t = im.run(a, 0.0, 11)
    xa = a.to_host()
    with pytest.raises(ValueError, match="reset"):
        im.run(cuda.to_device(xa), t + dt, 1)                    # right state, wrong time
    bad = xa.copy()
    bad[1, 3] = np.nextafter(bad[1, 3], np.inf)                  # one ulp in one entry
    with pytest.raises(ValueError, match="reset"):
        im.run(cuda.to_device(bad), t, 1)
    b = cuda.to_device(xa)                                       # a copy of the returned state
    im.run(b, t, 19)
    assert np.array_equal(one.to_host(), b.to_host())


def test_device_imex_reset_equals_a_fresh_integrator_complex(rng):
    """Complex systems (ImexZ): after reset() a new trajectory equals a fresh integrator bit for
    bit, and without reset() it is refused."""
    N, batch, dt = 20, 2, 1e-3
    systems = [_heat_like(N, k, complex) for k in (0.5 + 1.0j, 2.0 - 0.5j)]
    xa, xb = _state(rng, batch, N, True), _state(rng, batch, N, True)
    im = _imex("SBDF3", systems, dt)
    im.run(cuda.to_device(xa), 0.0, 7)
    db = cuda.to_device(xb)
    with pytest.raises(ValueError, match="reset"):
        im.run(db, 0.0, 7)
    im.reset()
    assert im.history_length == 0
    im.run(db, 0.0, 7)
    fresh = cuda.to_device(xb)
    _imex("SBDF3", systems, dt).run(fresh, 0.0, 7)
    assert np.array_equal(db.to_host(), fresh.to_host())


def test_device_imex_rk_has_no_history_to_refuse(rng):
    """Runge-Kutta schemes keep no history: two unrelated trajectories on one integrator are both
    accepted and equal fresh integrators."""
    N, batch, dt = 24, 2, 2e-3
    systems = [_heat_like(N, k) for k in (0.0, 1.0)]
    im = _imex("RK443", systems, dt)
    for _ in range(2):
        x = _state(rng, batch, N)
        d = cuda.to_device(x)
        im.run(d, 0.0, 5)
        f = cuda.to_device(x)
        _imex("RK443", systems, dt).run(f, 0.0, 5)
        assert np.array_equal(d.to_host(), f.to_host())


def _batch(rng, n, batch, kl=3, ku=3, r=2):
    mats = []
    for _ in range(batch):
        A = sp.lil_matrix((n, n))
        for i in range(n):
            lo, hi = max(0, i - kl), min(n, i + ku + 1)
            A[i, lo:hi] = rng.standard_normal(hi - lo)
            A[i, i] += 4 * (kl + ku + 1)
        A[:r] = rng.standard_normal((r, n))
        for k in range(r):
            A[k, k] += 4 * n
        mats.append(A.tocsr())
    return mats


def test_solve_mapped_refuses_or_survives_an_aliased_rhs(rng):
    """rhs and x the same buffer with different maps: every system reads its right-hand side
    while other systems write their solutions into the same array. With many more systems than
    resident blocks, a late block reads entries an early block already overwrote (wrong values on
    an L40S before run). Oracle: the same maps out of place.
    Either the call is refused or it equals the oracle."""
    import cupy as cp
    n, batch = 40, 8192
    mats = _batch(rng, n, 4)
    bb = cuda.BandedBatch([mats[i % 4] for i in range(batch)], [0, 1])
    b = cp.asarray(rng.standard_normal(batch * n))
    ident = cp.arange(batch * n, dtype=np.int64).reshape(batch, n)
    rows = cp.roll(ident, 1, axis=0).copy()                    # system m reads system m-1's slot
    ref = cp.zeros_like(b)
    bb.solve_mapped(b, rows, ref, ident)                        # out of place: the oracle
    x = b.copy()
    try:
        bb.solve_mapped(x, rows, x, ident)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        cp.cuda.Device().synchronize()
        assert cp.array_equal(x, ref)
    # in place with one map for both, and a view of half of x as rhs: the safe case passes and
    # equals the out-of-place solve; the overlapping view is refused
    y = b.copy()
    bb.solve_mapped(y, ident, y, ident)
    ref2 = cp.zeros_like(b)
    bb.solve_mapped(b, ident, ref2, ident)
    cp.cuda.Device().synchronize()
    assert cp.array_equal(y, ref2)
    big = cp.zeros(2 * batch * n)
    with pytest.raises(ValueError, match="overlap"):
        bb.solve_mapped(big[batch * n // 2: batch * n // 2 + batch * n], None, big[:batch * n], None)


@pytest.mark.parametrize("level_schedule", [False, True])
def test_band_solve_captured_in_a_cuda_graph(level_schedule, rng):
    """The band kernel's work vector is allocated on the solve's stream (cudaMallocAsync) for
    n * 8 > 40 KiB. Captured in a CUDA graph it becomes an allocation node: the replayed graph must
    give the eager solve's bits for new right-hand sides, also while an eager solve of the same
    batch runs on another stream."""
    import cupy as cp
    n, batch = 6000, 16
    bb = cuda.BandedBatch(_batch(rng, n, batch, kl=4, ku=4), [0, 1], level_schedule=level_schedule)
    b = cp.zeros(batch * n)
    x = cp.zeros(batch * n)
    s = cp.cuda.Stream(non_blocking=True)
    with s:
        bb.solve_mapped(b, None, x, None)                      # warm-up
        s.synchronize()
        s.begin_capture()
        bb.solve_mapped(b, None, x, None)
        graph = s.end_capture()
    other = cp.cuda.Stream(non_blocking=True)
    for _ in range(6):
        rhs = cp.asarray(rng.standard_normal(batch * n))
        rhs2 = cp.asarray(rng.standard_normal(batch * n))
        ref, ref2 = cp.zeros_like(rhs), cp.zeros_like(rhs)
        bb.solve_mapped(rhs, None, ref, None)
        bb.solve_mapped(rhs2, None, ref2, None)
        cp.cuda.Device().synchronize()
        b[...] = rhs
        cp.cuda.Device().synchronize()
        out2 = cp.zeros_like(rhs2)
        graph.launch(s)
        with other:
            bb.solve_mapped(rhs2, None, out2, None)
        cp.cuda.Device().synchronize()
        assert cp.array_equal(x, ref)
        assert cp.array_equal(out2, ref2)


def test_band_solves_do_not_leak_device_memory(rng):
    """Many eager band-kernel solves with the per-call work vector: the free device memory after
    the loop equals the free memory after the first solve (the driver pool returns the work
    vectors; a leak of batch * n * 8 bytes per call would show after 200 calls)."""
    import cupy as cp
    n, batch = 6000, 32
    bb = cuda.BandedBatch(_batch(rng, n, batch, kl=4, ku=4), [0, 1], level_schedule=False)
    rhs = cp.asarray(rng.standard_normal(batch * n))
    x = cp.zeros_like(rhs)
    bb.solve_mapped(rhs, None, x, None)
    cp.cuda.Device().synchronize()
    free0, _ = cp.cuda.runtime.memGetInfo()
    for _ in range(200):
        bb.solve_mapped(rhs, None, x, None)
    cp.cuda.Device().synchronize()
    free1, _ = cp.cuda.runtime.memGetInfo()
    per_call = batch * n * 8
    assert free0 - free1 < 4 * per_call, (free0, free1, per_call)
