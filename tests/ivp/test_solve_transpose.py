"""GlobalSystem.solve_transpose: (a M + b L)^T x = r with the same factors as solve, checked by
the bilinear identity <G^{-1} u, r> = <u, G^{-T} r> (full space, subspace, complexified)."""
import numpy as np
import pytest

from ivp_cases import rayleigh_benard_2d

from conatus.timestep import IVPSolver


@pytest.mark.parametrize("sym,cx", [(False, False), (True, False), (False, True)])
def test_solve_transpose_bilinear_identity(sym, cx):
    plan, _ = rayleigh_benard_2d(16, 8, Ra=1e5, Lx=2.0, symmetric=sym)
    s = IVPSolver(plan, "RK222", 1e-3, complexify=cx).system
    rng = np.random.default_rng(5)
    u, r = rng.standard_normal(s.n), rng.standard_normal(s.n)
    a, b = 1.0, 3e-3
    lhs = np.dot(s.solve(a, b, u), r)
    rhs = np.dot(u, s.solve_transpose(a, b, r))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0)


@pytest.mark.gpu
@pytest.mark.parametrize("sym,cx", [(False, False), (True, False), (False, True)])
def test_device_solve_transpose_matches_cpu(sym, cx):
    from conatus.runtime import cuda
    if not cuda.available():
        pytest.skip("no CUDA device on this node")
    import cupy as cp
    plan, _ = rayleigh_benard_2d(16, 8, Ra=1e5, Lx=2.0, symmetric=sym)
    h = IVPSolver(plan, "RK222", 1e-3, complexify=cx).system
    g = IVPSolver(plan, "RK222", 1e-3, complexify=cx, backend="cuda").system
    r = np.random.default_rng(6).standard_normal(h.n)
    x = h.solve_transpose(1.0, 3e-3, r)
    y = cp.asnumpy(g.solve_transpose(1.0, 3e-3, cp.asarray(r)))
    assert np.abs(x - y).max() <= 1e-11 * np.abs(x).max()


def test_solve_refuses_a_lossy_right_hand_side():
    """A complex right-hand side for real factors had its imaginary part dropped
    (CPU: only a ComplexWarning; GPU: the kernels read the complex memory as real numbers). An
    exact widening (float32 -> float64) is done, a narrowing refused, at every solve entry."""
    from conatus.errors import DtypeError
    plan, _ = rayleigh_benard_2d(16, 8, Ra=1e5, Lx=2.0)
    s = IVPSolver(plan, "RK222", 1e-3).system
    r = np.random.default_rng(6).standard_normal(s.n)
    for solve in (s.solve, s.solve_transpose):
        with pytest.raises(DtypeError, match="complex128 for float64 factors"):
            solve(1.0, 3e-3, r + 1j * r)
        r32 = r.astype(np.float32)
        assert np.array_equal(solve(1.0, 3e-3, r32), solve(1.0, 3e-3, r32.astype(np.float64)))


@pytest.mark.gpu
def test_device_solve_refuses_a_lossy_right_hand_side():
    """The same check on the GPU, where the raw-pointer kernels gave no error at all."""
    from conatus.errors import DtypeError
    from conatus.runtime import cuda
    if not cuda.available():
        pytest.skip("no CUDA device on this node")
    import cupy as cp
    plan, _ = rayleigh_benard_2d(16, 8, Ra=1e5, Lx=2.0)
    s = IVPSolver(plan, "RK222", 1e-3, backend="cuda").system
    r = cp.asarray(np.random.default_rng(7).standard_normal(s.n))
    for solve in (s.solve, s.solve_transpose):
        with pytest.raises(DtypeError, match="complex128 for float64 factors"):
            solve(1.0, 3e-3, r + 1j * r)
        assert cp.array_equal(solve(1.0, 3e-3, r.astype(cp.float32)),
                              solve(1.0, 3e-3, r.astype(cp.float32).astype(cp.float64)))
