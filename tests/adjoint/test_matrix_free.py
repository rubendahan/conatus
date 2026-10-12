"""Matrix-free transpose of the linearized RHS and of the dealiased transforms ( phase 2)."""
import numpy as np
import pytest
from rounding_bounds import adjoint_dot_check

from conatus.adjoint import FlowMap
from conatus.adjoint.rhs_transpose import LinearizedRHS, coeffs_T, grid_T
from conatus.ir import CartesianCoordinates
from conatus.ir.bases import Chebyshev, ComplexFourier, RealFourier
from conatus.reference import transforms as T
from conatus.timestep import IVPSolver
from ivp_cases import burgers_1d, ks_1d, rayleigh_benard_2d

rng = np.random.default_rng(3)


def _vec(n, cplx, shape=()):
    v = rng.standard_normal((n,) + shape)
    return v + 1j * rng.standard_normal((n,) + shape) if cplx else v


@pytest.mark.parametrize("kind", ["cheb", "real", "complex"])
@pytest.mark.parametrize("dealias", [1.0, 1.5, 2.0])
def test_transform_transposes_are_exact(kind, dealias):
    """⟨backward c, w⟩ = ⟨c, gridᵀ w⟩ and ⟨forward f, c⟩ = ⟨f, coeffsᵀ c⟩ along a middle axis."""
    x = CartesianCoordinates("x")["x"]
    B = {"cheb": Chebyshev(x, 12, (-1, 2), dealias=dealias),
         "real": RealFourier(x, 12, (0, 3), dealias=dealias),
         "complex": ComplexFourier(x, 12, (0, 3), dealias=dealias)}[kind]
    N, M, cp = B.size, B.grid_size, kind == "complex"
    c, w = np.moveaxis(_vec(N, cp, (2, 3)), 0, 1), np.moveaxis(_vec(M, cp, (2, 3)), 0, 1)
    lhs, rhs = np.sum(T.backward(c, B, 1) * w), np.sum(c * grid_T(w, B, 1))
    assert abs(lhs - rhs) < 1e-13 * abs(lhs)
    f, ct = np.moveaxis(_vec(M, cp, (2, 3)), 0, 1), np.moveaxis(_vec(N, cp, (2, 3)), 0, 1)
    lhs, rhs = np.sum(T.forward(f, B, 1) * ct), np.sum(f * coeffs_T(ct, B, 1))
    assert abs(lhs - rhs) < 1e-13 * abs(lhs)


CASES = {
    "burgers": lambda: (burgers_1d(N=16)[0], False),
    "burgers_complex": lambda: (burgers_1d(N=16, basis="complex")[0], True),
    "ks": lambda: (ks_1d(N=32)[0], False),
    "rb2d": lambda: (rayleigh_benard_2d(8, 8, 1e4)[0], False),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_linearized_rhs_matches_ir_and_transposes(case):
    plan, cp = CASES[case]()
    s = IVPSolver(plan, "RK222", 1e-3)
    n = s.system.n
    X, V, W = (0.1 * _vec(n, cp)).astype(s.system.dtype), _vec(n, cp), _vec(n, cp)
    lr = LinearizedRHS(s)
    a = lr.apply(X, V)
    ir = FlowMap(s, 1, f_transpose="dense")._JF(X, V)       # IR linearization (phase-1 oracle)
    assert np.linalg.norm(a - ir) < 1e-13 * np.linalg.norm(ir)
    lhs, rhs = np.sum(a * W), np.sum(V * lr.transpose(X, W))
    assert abs(lhs - rhs) < 1e-13 * abs(lhs)
    assert "matrix-free" in lr.explain()


def test_polynomial_vjp_equals_dense_oracle_and_superlu():
    plan, _ = CASES["ks"]()
    s = IVPSolver(plan, "SBDF3", 1e-2, startup="rk443")
    s.set_state({s.system.variables[0]: lambda x: np.cos(x / 16) * (1 + np.sin(x / 16))})
    X0, W = s.X.copy(), _vec(s.system.n, False)
    new = FlowMap(s, 6).vjp(X0, W)
    old = FlowMap(s, 6, f_transpose="dense", solves="superlu").vjp(X0, W)
    assert np.linalg.norm(new - old) < 1e-11 * np.linalg.norm(old)


@pytest.mark.parametrize("scheme,startup", [("SBDF2", "lower_order"), ("SBDF3", "rk443"),
                                            ("SBDF4", "rk443")])
def test_revolve_multistep_equals_full_storage(scheme, startup):
    plan, _ = CASES["burgers"]()
    s = IVPSolver(plan, scheme, 1e-2, startup=startup)
    s.set_state({s.system.variables[0]: lambda x: np.sin(x) + 0.3 * np.cos(2 * x)})
    X0, W = s.X.copy(), _vec(s.system.n, False)
    full = FlowMap(s, 11).vjp(X0, W)
    rev = FlowMap(s, 11, checkpoints=2)
    assert np.array_equal(rev.vjp(X0, W), full)          # same arithmetic, bit-for-bit
    assert rev.n_recompute > 0 and "augmented state" in rev.explain()


def test_rb2d_dot_and_taylor_polynomial_banded():
    plan, _ = CASES["rb2d"]()
    s = IVPSolver(plan, "RK443", 1e-3)
    n = s.system.n
    # own generator: the module generator made the vectors depend on which tests ran
    # before in the same xdist worker, and the test failed in one order (CI)
    r = np.random.default_rng(29)
    X0 = 0.05 * r.standard_normal(n)
    fm = FlowMap(s, 3)
    V, W = r.standard_normal(n), r.standard_normal(n)
    JV = fm.jvp(X0, V)
    lhs, rhs = np.sum(JV * W), np.sum(V * fm.vjp(X0, W))
    # dot-product test scaled by ||JV|| ||W||, not by |<JV, W>| (which cancellation makes small)
    assert abs(lhs - rhs) < 1e-12 * np.linalg.norm(JV) * np.linalg.norm(W)
    P0 = fm(X0)
    errs = [np.linalg.norm(fm(X0 + e * V) - P0 - e * JV) for e in (1e-3, 5e-4, 2.5e-4)]
    assert min(np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])) > 1.9


def test_complexified_groups_dot_test():
    """complexify=True: the solver's solve_transpose handles the complex half-size groups."""
    plan, _ = CASES["rb2d"]()
    s = IVPSolver(plan, "RK222", 1e-3, complexify=True)
    n = s.system.n
    X0, V, W = 0.05 * _vec(n, False), _vec(n, False), _vec(n, False)
    fm = FlowMap(s, 2)
    # rounding-derived bound (tests/rounding_bounds.py rule; the former fixed 1e-11 |lhs| was met
    # at 2.3e-9 / 7.9e-9 on genoa, margin 3.4, run). cap 1e-8: a
    # rounding-sized linearity defect of these small problems (cond < 1e7), far below the 1e-6
    # adjoint error the check must catch
    adjoint_dot_check(lambda v: fm.jvp(X0, v), lambda w: fm.vjp(X0, w), V, W, _vec(n, False),
                      _vec(n, False), cap=1e-8, dims=(n, n))
    ref = FlowMap(IVPSolver(plan, "RK222", 1e-3), 2).vjp(X0, W)
    assert np.linalg.norm(fm.vjp(X0, W) - ref) < 1e-10 * np.linalg.norm(ref)


@pytest.mark.parametrize("scheme", ["RK443", "SBDF2"])
def test_linearization_sees_in_place_updates_of_the_state(scheme):
    """ ( in the Cartesian adjoint): ``LinearizedRHS`` kept factor grids
    across calls in an LRU keyed by the array's id and pointer; ``fm.vjp(X0, W)``, then
    ``X0[:] = X1``, then ``fm.vjp(X0, W)`` returned the adjoint about the old X0 (``apply``: 4x the
    norm of the answer off). No state-dependent data is kept across calls any more."""
    plan, _ = CASES["burgers"]()
    s = IVPSolver(plan, scheme, 1e-3)
    n = s.system.n
    X0, W, X1 = 0.1 * _vec(n, False), _vec(n, False), 0.1 * _vec(n, False)
    fm = FlowMap(s, 2)
    fm.vjp(X0, W)
    X0[:] = X1
    assert np.array_equal(fm.vjp(X0, W), FlowMap(s, 2).vjp(X1.copy(), W))
    fm.jvp(X0, W)
    X0[:] = 2 * X1
    assert np.array_equal(fm.jvp(X0, W), FlowMap(s, 2).jvp(2 * X1, W))
    lr = LinearizedRHS(s)
    Y = 0.1 * _vec(n, False)
    lr.apply(Y, W)
    lr.transpose(Y, W)
    Y[:] = X1
    assert np.array_equal(lr.apply(Y, W), LinearizedRHS(s).apply(X1, W))
    assert np.array_equal(lr.transpose(Y, W), LinearizedRHS(s).transpose(X1, W))
