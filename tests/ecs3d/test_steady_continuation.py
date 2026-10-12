"""Matrix-free steady states and continuation: affine L(Re) = A + B/Re, laminar Couette
is an exact solution, the Stokes-preconditioned Jacobian equals the directional derivative of the
residual, the Krylov arclength corrector locates a fold of a model problem. The Nagata /
Waleffe equilibria themselves are GPU runs (tests/ecs3d/run_couette_ecs.py)."""
import numpy as np
import pytest

from couette_cases import BOX_GHC, couette_pt, shift_reflect_group
from conatus.continuation.krylov import KrylovArclength, KrylovContinuationSettings
from conatus.ecs import NewtonKrylovHookstep
from conatus.ecs.steady import AffineSteadyProblem
from conatus.timestep import IVPSolver
from conatus.timestep.system import GlobalSystem

NEWTON = NewtonKrylovHookstep(tol=1e-11, max_iterations=20, gmres_rtol=1e-12, gmres_max_iterations=40,
                              delta0=1.0, delta_min=1e-8, delta_max=10.0)


@pytest.mark.slow                   # 3.2 s in the fast suite (qual_durations0)
def test_affine_steady_problem_on_couette():
    N = (6, 13, 6)
    plan, f = couette_pt(*N, 300.0, BOX_GHC)
    # coupled cos/sin blocks (decouple=False): the Oseen preconditioner couples them, and the
    # state/row layouts of all the systems must coincide
    s = IVPSolver(plan, "RK443", 0.02, f_backend="fused")
    sys2 = GlobalSystem(couette_pt(*N, 200.0, BOX_GHC)[0])
    sys3 = GlobalSystem(couette_pt(*N, 450.0, BOX_GHC)[0])
    G = shift_reflect_group(s)
    # Oseen preconditioner: the same problem linearized about U = y (same state layout)
    sP = IVPSolver(couette_pt(*N, 300.0, BOX_GHC, linear_about="couette")[0], "RK443", 0.02)
    pb = AffineSteadyProblem(s, 300.0, sys2, 200.0, sys3, 450.0, group=G, affine_tol=1e-12,
                             preconditioner=sP, preconditioner_name="Oseen (about U = y)")
    assert pb.affine_error < 1e-13
    s.set_state({f["U"]: lambda X, Y, Z: Y + 0 * X})
    x = pb.active(s.X)
    assert pb.full_residual(x, 377.0) < 1e-14          # laminar Couette: exact for every Re
    rng = np.random.default_rng(2)
    x1 = x + 1e-2 * rng.standard_normal(x.size) * np.exp(-np.arange(x.size) / x.size)
    v = rng.standard_normal(x.size)
    Jv = pb.jvp(x1, v, 377.0)
    h = 1e-3
    fd = (pb.residual(x1 + h * v, 377.0) - pb.residual(x1 - h * v, 377.0)) / (2 * h)
    assert np.linalg.norm(Jv - fd) < 1e-9 * np.linalg.norm(fd)      # quadratic: exact up to rounding
    dp = 1e-3
    fdp = (pb.residual(x1, 377.0 + dp) - pb.residual(x1, 377.0 - dp)) / (2 * dp)
    assert np.linalg.norm(pb.dRdp(x1, 377.0) - fdp) < 1e-6 * np.linalg.norm(fdp)
    # Newton from a smooth perturbed laminar state returns to it (the only solution nearby)
    s.set_state({f["U"]: lambda X, Y, Z: Y + 0.05 * (1 - Y**2) * Y + 0 * X,
                 f["phi"]: lambda X, Y, Z: 0.01 * (1 - Y**2) ** 2 * np.cos(2.28 * X) * np.cos(5 * Z),
                 f["psi"]: lambda X, Y, Z: 0.01 * (1 - Y**2) * np.sin(2.28 * X) * np.sin(5 * Z)})
    sol = pb.solve(s.X, 377.0, NEWTON)
    assert sol.residual < 1e-12 and sol.newton.iterations <= 6


def test_krylov_arclength_locates_a_fold():
    """G(x, p) = (x0^2 + p - 1, x1 - x0^3, x2 + x0): fold at p = 1 (x0 = 0)."""
    def G(x, p):
        return np.array([x[0] ** 2 + p - 1, x[1] - x[0] ** 3, x[2] + x[0]])

    def Jv(x, v, p):
        return np.array([2 * x[0] * v[0], v[1] - 3 * x[0] ** 2 * v[0], v[2] + v[0]])

    def Gp(x, p):
        return np.array([1.0, 0.0, 0.0])

    def pt(x0):
        return np.array([x0, x0 ** 3, -x0]), 1 - x0 ** 2
    cfg = KrylovContinuationSettings(ds=0.1, ds_min=1e-6, ds_max=0.2, adapt=1.5, fast_iterations=3,
                                     max_points=40, p_weight=1.0, fold_refinements=3,
                                     fold_refine_factor=4.0)
    ka = KrylovArclength(G, Jv, Gp, cfg, NEWTON, report=lambda x, p: np.linalg.norm(G(x, p)),
                         observe=lambda x, p: {"x1": float(x[1])})
    (x0, p0), (x1, p1) = pt(-0.8), pt(-0.75)
    ka.run(x0, p0, x1, p1, p_bounds=(-5.0, 1.5))
    assert len(ka.folds) == 1
    fo = ka.folds[0]
    assert abs(fo["p"] - 1.0) < 1e-6 and abs(fo["x1"]) < 1e-3
    assert max(q.residual for q in ka.points) < 1e-10
