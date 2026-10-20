"""GPU (CUDA backend) vs CPU (NumPy reference transforms, C++ host solves): transforms, batched bordered banded solves,
batched IMEX, determinism (agreement within a stated error model, not bitwise)."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.ir import CartesianCoordinates, Chebyshev, ComplexFourier, RealFourier
# CPU side: the NumPy reference transforms (pocketfft); the C++ host FFTW transforms were removed
from conatus.reference import transforms as CPU
from conatus.runtime import BorderedBanded, ImexIntegrator
from conatus.runtime import cuda

rng = np.random.default_rng(7)
_c = CartesianCoordinates("x")


def rel(a, b):
    return np.max(abs(a - b)) / max(np.max(abs(b)), 1e-300)


# ---------------------------------------------------------------------------------- transforms
@pytest.mark.parametrize("dealias", [1, 1.5])
@pytest.mark.parametrize("kind", ["cheb_real", "cheb_complex", "realfourier", "complexfourier"])
def test_transforms_match_cpu(kind, dealias):
    N, batch = 64, 37
    if kind.startswith("cheb"):
        b = Chebyshev(_c["x"], N, (-1, 1), dealias=dealias)
    elif kind == "realfourier":
        b = RealFourier(_c["x"], N, (0, 2 * np.pi), dealias=dealias)
    else:
        b = ComplexFourier(_c["x"], N, (0, 2 * np.pi), dealias=dealias)
    M = CPU.grid_size(b, True)
    cplx = kind in ("cheb_complex", "complexfourier")
    f = rng.standard_normal((batch, M)) + (1j * rng.standard_normal((batch, M)) if cplx else 0)
    c_cpu = CPU.forward(f, b, axis=1)
    c_gpu = cuda.forward(cuda.to_device(f), b).to_host()
    assert rel(c_gpu, c_cpu) < 1e-13
    g_cpu = CPU.backward(c_cpu, b, axis=1)
    g_gpu = cuda.backward(cuda.to_device(c_cpu), b).to_host()
    assert rel(g_gpu, g_cpu) < 1e-13


def test_raw_ffts_match_numpy():
    x = rng.standard_normal((9, 48)) + 1j * rng.standard_normal((9, 48))
    cu = cuda._need()
    assert rel(cu.fft_c2c(cuda.to_device(x), True).to_host(), np.fft.fft(x, axis=1)) < 1e-13
    assert rel(cu.fft_c2c(cuda.to_device(x), False).to_host(), np.fft.ifft(x, axis=1)) < 1e-13
    xr = x.real.copy()
    X = cu.rfft(cuda.to_device(xr)).to_host()
    assert rel(X, np.fft.rfft(xr, axis=1)) < 1e-13
    assert rel(cu.irfft(cuda.to_device(X), 48).to_host(), xr) < 1e-13


def test_transform_determinism_and_pool_reuse():
    b = Chebyshev(_c["x"], 128, (-1, 1), dealias=1.5)
    f = rng.standard_normal((64, CPU.grid_size(b, True)))
    d = cuda.to_device(f)
    a1 = cuda.forward(d, b).to_host()
    a2 = cuda.forward(d, b).to_host()
    assert np.array_equal(a1, a2)                      # bitwise, same plan / device
    assert cuda.pool_stats()["n_reuse"] > 0            # scratch buffers come from the pool


# ------------------------------------------------------------------------------ banded solves
def almost_banded(n, kl, ku, r, dtype, dominant=False):
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            A[i, j] = rng.standard_normal() + (1j * rng.standard_normal() if dtype == complex else 0)
        if dominant:
            A[i, i] += 4 * (kl + ku + 1)
    for i in range(r):
        A[i] = rng.standard_normal(n) + (1j * rng.standard_normal(n) if dtype == complex else 0)
    return A


@pytest.mark.parametrize("dominant", [True, False])
@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("r", [1, 2, 4])
def test_batched_bordered_solve_matches_cpu(dtype, r, dominant):
    """Same algorithm as the host solver (banded LU + Woodbury). Well-conditioned bands: residual at
    rounding. Non-dominant random bands: the bordering itself loses accuracy with cond(B) (measured on
    the host too), so the GPU must match the host residual level, not a fixed tolerance."""
    n, batch = 96, 11
    border = list(range(r))
    mats = [almost_banded(n, int(rng.integers(1, 5)), int(rng.integers(1, 5)), r, dtype, dominant)
            for _ in range(batch)]
    gpu = cuda.BandedBatch([sp.csr_matrix(A) for A in mats], border)
    assert gpu.rank == r and gpu.batch == batch
    b = rng.standard_normal((batch, n)) + (1j * rng.standard_normal((batch, n)) if dtype == complex else 0)
    d = cuda.to_device(b)
    gpu.solve(d)
    x = d.to_host()
    for k, A in enumerate(mats):
        res_gpu = np.max(abs(A @ x[k] - b[k]))
        x_cpu = BorderedBanded(sp.csr_matrix(A), border).solve(b[k])
        res_cpu = np.max(abs(A @ x_cpu - b[k]))
        if dominant:
            assert res_gpu < 1e-12 * np.max(abs(b[k])) * n
            assert rel(x[k], x_cpu) < 1e-12
        else:
            assert res_gpu < 10 * res_cpu + 1e-13 * n * np.max(abs(b[k]))
    assert "one warp per mode" in gpu.explain()


def test_no_pivot_on_declared_dominant_band_and_singular_refused():
    n, batch = 64, 5
    mats = [almost_banded(n, 3, 2, 2, complex, dominant=True) for _ in range(batch)]
    gpu = cuda.BandedBatch([sp.csr_matrix(A) for A in mats], [0, 1], pivot=False)
    b = rng.standard_normal((batch, n)) + 0j
    d = cuda.to_device(b)
    gpu.solve(d)
    x = d.to_host()
    for k, A in enumerate(mats):
        assert np.max(abs(A @ x[k] - b[k])) < 1e-11 * n
    S = np.eye(n)
    S[5, 5] = 0.0                                         # singular banded part (row 5 empty)
    with pytest.raises(RuntimeError, match="singular"):
        cuda.BandedBatch([sp.csr_matrix(S)], [])


def test_solve_determinism():
    n, batch = 80, 9
    mats = [sp.csr_matrix(almost_banded(n, 2, 3, 2, complex)) for _ in range(batch)]
    gpu = cuda.BandedBatch(mats, [0, 1])
    b = rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))
    d1, d2 = cuda.to_device(b), cuda.to_device(b)
    gpu.solve(d1)
    gpu.solve(d2)
    assert np.array_equal(d1.to_host(), d2.to_host())


# -------------------------------------------------------------------------------------- IMEX
def heat_like(N, kappa):
    """(M, L, border) for u_t - u_xx + kappa u + taus = F on Chebyshev, u(+-1) = 0 (tau form)."""
    from conatus.discretize.ultraspherical import conversion_chain, differentiation
    S = conversion_chain(N, 0, 2)
    D2 = differentiation(N, 0, 2)
    n = N + 2
    M = sp.lil_matrix((n, n))
    L = sp.lil_matrix((n, n))
    M[:N, :N] = S
    L[:N, :N] = -D2 + kappa * S
    L[N - 1, N] = 1.0
    L[N - 2, N + 1] = 1.0
    L[N, :N] = (-1.0) ** np.arange(N)
    L[N + 1, :N] = 1.0
    return M.tocsr(), L.tocsr(), [N, N + 1]


@pytest.mark.parametrize("scheme", ["SBDF1", "SBDF2", "SBDF3", "SBDF4", "RK111", "RK222", "RK443"])
def test_batched_imex_matches_cpu(scheme):
    N, batch, dt, steps = 32, 6, 2e-3, 100
    kappas = np.linspace(0.0, 5.0, batch)
    systems = [heat_like(N, k) for k in kappas]
    n = N + 2
    g = 0.3 * rng.standard_normal((batch, n))
    h = 0.1 * rng.standard_normal((batch, n))
    g[:, N:] = 0.0                                        # no explicit forcing on the tau rows
    h[:, N:] = 0.0
    x0 = np.zeros((batch, n))
    x0[:, :N] = rng.standard_normal((batch, N)) * np.exp(-0.3 * np.arange(N))
    gpu = cuda.Imex([s[0] for s in systems], [s[1] for s in systems], systems[0][2], scheme, dt, g=g, h=h)
    d = cuda.to_device(x0)
    t = gpu.run(d, 0.0, steps)
    x_gpu = d.to_host()
    assert abs(t - steps * dt) < 1e-12
    for k, (M, L, border) in enumerate(systems):
        gk, hk = g[k].copy(), h[k].copy()
        cpu = ImexIntegrator(M, L, border, scheme, dt, F=lambda x, t, gk=gk, hk=hk: gk * x + hk)
        x_cpu, _ = cpu.run(x0[k], 0.0, steps)
        # error model: roundoff of 100 steps with O(1)-conditioned solves -> far below 1e-10
        assert rel(x_gpu[k], x_cpu) < 1e-10, (scheme, k, rel(x_gpu[k], x_cpu))
    if scheme == "RK443":
        assert gpu.n_factorizations == 1                  # trap P1 on the device too
    d2 = cuda.to_device(x0)
    cuda.Imex([s[0] for s in systems], [s[1] for s in systems], systems[0][2], scheme, dt,
              g=g, h=h).run(d2, 0.0, steps)
    assert np.array_equal(d2.to_host(), x_gpu)           # GPU run-to-run determinism
