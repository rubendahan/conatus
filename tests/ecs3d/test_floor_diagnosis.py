"""A tolerance below the attainable residual floor is diagnosed, not hidden (the full-space
3D Couette continuation and a 48x33x48 one in the symmetric subspace crawled at small steps
because the corrector stalled at 2-4e-10 against tol 1e-10). Numerics unchanged: the Newton
failure says why and what to declare instead, and the continuation's explain() counts them."""
import numpy as np
import pytest

from conatus.continuation.krylov import KrylovArclength, KrylovContinuationSettings
from conatus.ecs import NewtonKrylovHookstep, NewtonNotConverged

FLOOR = 3e-9          # deterministic 'rounding' added to the residual


def test_newton_failure_near_tol_says_the_tolerance_is_below_the_floor():
    nk = NewtonKrylovHookstep(tol=1e-9, max_iterations=6, gmres_rtol=1e-12, gmres_max_iterations=5,
                              delta0=1.0, delta_min=1e-12, delta_max=10.0)
    # the third residual component is a constant FLOOR no step can remove (A singular there)
    A = np.diag([1.0, 2.0, 0.0])
    b = np.array([1.0, 1.0, 0.0])
    with pytest.raises(NewtonNotConverged) as ei:
        nk.solve(lambda z: A @ z - b + FLOOR * np.eye(3)[2], lambda z, v: A @ v, np.zeros(3),
                 scale=1.0)
    e = ei.value
    assert e.stagnated_near_tol and e.relative_residual < 10 * e.tol
    assert "attainable residual floor" in str(e) and "tol=" in str(e)


def test_continuation_explain_counts_floor_failures():
    nk = NewtonKrylovHookstep(tol=1e-9, max_iterations=5, gmres_rtol=1e-12, gmres_max_iterations=5,
                              delta0=1.0, delta_min=1e-12, delta_max=10.0)
    cfg = KrylovContinuationSettings(ds=0.05, ds_min=1e-3, ds_max=0.1, adapt=1.3, fast_iterations=4,
                                     max_points=6, p_weight=1.0, fold_refinements=1,
                                     fold_refine_factor=4.0)
    # second component: a constant FLOOR (x[1] does not enter G): no corrector can go below it
    def G(x, p):
        return np.array([x[0] ** 2 + p - 1, FLOOR])

    def Jv(x, v, p):
        return np.array([2 * x[0] * v[0], 0.0])
    ca = KrylovArclength(G, Jv, lambda x, p: np.array([1.0, 0.0]), cfg, nk,
                         report=lambda x, p: float(np.linalg.norm(G(x, p))))
    ca.run(np.array([0.5, 0.0]), 0.75, np.array([0.6, 0.0]), 0.64, p_bounds=(0.0, 2.0))
    assert ca.floor_failures and "below the attainable residual floor" in ca.explain()


def test_a_run_still_converging_at_its_budget_is_not_called_a_floor():
    """Linear convergence (inexact Jacobian) that runs out of iterations near tol: no floor label."""
    nk = NewtonKrylovHookstep(tol=1e-9, max_iterations=4, gmres_rtol=1e-12, gmres_max_iterations=5,
                              delta0=10.0, delta_min=1e-12, delta_max=10.0)
    A = np.diag([1.0, 2.0, 3.0])
    b = np.ones(3)
    z_star = np.linalg.solve(A, b)
    # Jacobian off by 1 %: each step divides the error by ~100, from 1e-1 to ~1e-9 in 4 steps
    with pytest.raises(NewtonNotConverged) as ei:
        nk.solve(lambda z: A @ z - b, lambda z, v: 1.01 * (A @ v), z_star + 0.1, scale=1.0)
    assert not ei.value.stagnated_near_tol and "floor" not in str(ei.value)
