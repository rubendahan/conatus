"""Phase-space geometry: dense-output event location, parametrization of unstable
manifolds (closed forms, resonance and degree refusals), connections (Lorenz homoclinic orbit as
a connection), KS U+ manifolds, subspace restriction and the SH23 periodic fold / Maxwell point.
Notes:"""
import numpy as np
import pytest

from conatus.bifurcation import DynamicalSystem, ir_system, newton_equilibrium, ode_problem
from conatus.ecs import FourierOps, NewtonKrylovHookstep, find_equilibrium, ks_solver
from conatus.errors import OptionError, UnsupportedError
from conatus.manifolds import (ODEIntegrator, PoincareSection, SolverIntegrator, ecs_system,
                               equilibrium_connection, even_subspace, find_connection,
                               first_return_map, first_return_stretch, orbit_manifold_seeds,
                               poincare_crossings, restrict_system, unstable_left_basis,
                               unstable_manifold)


def callables(f, n, jac=None):
    return DynamicalSystem.from_callables(f, n, {}, jac=jac, fd_x=1e-4, fd_p=1e-6, name="test")


# ------------------------------------------------------------------ dense output and sections
def test_crossings_of_a_rotation_are_exact():
    """ẋ = −y, ẏ = x from (1, 0): y = 0 crossed downwards at t = π, 3π, ... (DOP853 dense output)."""
    ds = callables(lambda x, p: np.array([-x[1], x[0]]), 2, jac=lambda x, p: np.array([[0, -1.0], [1, 0]]))
    tr = ODEIntegrator(ds, method="DOP853", rtol=1e-12, atol=1e-14).integrate([1.0, 0.0], 20.0)
    sec = PoincareSection.hyperplane([0, 1.0], [0, 0], direction=-1, basis=[[1.0, 0]])
    cr = poincare_crossings(tr, sec, xtol=1e-14)
    assert np.allclose(cr.t, [np.pi, 3 * np.pi, 5 * np.pi], atol=1e-10)
    pairs, taus = first_return_map(cr, sec)
    assert np.allclose(taus, 2 * np.pi, atol=1e-10) and np.allclose(pairs, -1.0, atol=1e-10)
    assert "crossings" in cr.explain()


def test_hermite_dense_output_and_exact_restep_on_ks():
    """KS discrete map: a crossing located on the cubic Hermite dense output is within
    O(dt⁴) of the exact partial step; ``refine`` puts it on the discrete map (|g| ~ rounding)."""
    s = ks_solver(L=22.0, N=32, dealias=1.5, scheme="RK443", timestep=0.05)
    integ = SolverIntegrator(s, dt=0.05)
    rng = np.random.default_rng(3)
    c = np.zeros(32)
    c[2:12] = rng.normal(size=10) * 0.5
    X0 = s.system.gather_state([c])
    tr = integ.integrate(X0, 40.0)
    with pytest.raises(OptionError, match="multiple"):    # used to round t_end
        integ.integrate(X0, 40.03)
    with pytest.raises(OptionError, match="every"):
        integ.integrate(X0, 40.0, every=0)
    # review of find_connection integrates a fixed-step map up to the end-sphere
    # crossing, a time between two steps; it now runs to the first step at or after it
    from conatus.manifolds.connections import _covering_time
    t = _covering_time(integ, 1.2345)
    assert abs(t - 1.25) < 1e-15 and integ.integrate(X0, t).t[-1] >= 1.2345
    assert _covering_time(integ, 2.0) == 40 * 0.05 and _covering_time(object(), 1.2345) == 1.2345
    p = np.zeros(32)
    p[2] = 1.0
    P = s.system.gather_state([p])
    sec = PoincareSection(lambda X: float(X @ P), direction=0, name="cos1 = 0")
    cr = poincare_crossings(tr, sec, xtol=1e-13)
    assert len(cr.t) >= 1 and np.nanmax(cr.interp_defect) < 1e-5
    ref = poincare_crossings(tr, sec, xtol=1e-13, refine=True)
    assert ref.g_residual.max() < 1e-12 * np.abs(tr.X).max()


# ------------------------------------------------------------------ parametrization method
def test_parametrization_reproduces_closed_form_manifolds():
    """ẋ = λx, ẏ = μy + x²: W^u(0) = {y = x²/(2λ − μ)} (λ > 0 > μ). Real 2D with a complex pair:
    ẋ = ax − by, ẏ = bx + ay, ẇ = −w + x² + y²: W^u(0) = {w = (x² + y²)/(2a + 1)}."""
    lam, mu = 0.7, -1.3
    ds = callables(lambda x, p: np.array([lam * x[0], mu * x[1] + x[0] ** 2]), 2,
                   jac=lambda x, p: np.array([[lam, 0], [2 * x[0], mu]]))
    ds._B = lambda x, u, v, par: np.array([0.0, 2 * u[0] * v[0]])        # exact (quadratic field)
    m = unstable_manifold(ds, np.zeros(2), order=6, degree=2, eigvec_scale=1.0, unstable_tol=1e-9)
    for s in (0.1, 0.5, 1.0):
        x = m([s])
        assert abs(x[1] - x[0] ** 2 / (2 * lam - mu)) < 1e-14
        assert m.defect([s]) < 1e-13
    a, b = 0.2, 1.1

    def f(x, p):
        return np.array([a * x[0] - b * x[1], b * x[0] + a * x[1], -x[2] + x[0] ** 2 + x[1] ** 2])
    ds2 = callables(f, 3, jac=lambda x, p: np.array([[a, -b, 0], [b, a, 0], [2 * x[0], 2 * x[1], -1]]))
    ds2._B = lambda x, u, v, par: np.array([0.0, 0.0, 2 * (u[0] * v[0] + u[1] * v[1])])
    m2 = unstable_manifold(ds2, np.zeros(3), order=4, degree=2, eigvec_scale=1.0, unstable_tol=1e-9)
    assert m2.dim == 2 and m2.pairs == [(0, 1)]
    for s in m2.circle(0.7, 12):
        x = m2(s)
        assert abs(x[2] - (x[0] ** 2 + x[1] ** 2) / (2 * a + 1)) < 1e-14
        assert m2.imag_part(s) < 1e-14
    assert "parametrization order" in m2.explain()


def test_resonance_and_degree_are_refused():
    lam = 0.5
    ds = callables(lambda x, p: np.array([lam * x[0], 2 * lam * x[1] + x[0] ** 2]), 2,
                   jac=lambda x, p: np.array([[lam, 0], [2 * x[0], 2 * lam]]))
    with pytest.raises(UnsupportedError, match="resonance"):         # 2λ = eigenvalue: y ~ x² ln x
        unstable_manifold(ds, np.zeros(2), order=3, degree=2, eigvec_scale=1.0, unstable_tol=1e-9)
    # the strong unstable direction alone (dims = 1 keeps the most unstable, 2λ) has no resonance
    m = unstable_manifold(ds, np.zeros(2), order=3, degree=2, eigvec_scale=1.0, unstable_tol=1e-9,
                          dims=1)
    assert m.eigenvalues[0].real == pytest.approx(2 * lam) and any("sub-manifold" in n for n in m.notes)
    ds4 = callables(lambda x, p: np.array([x[0] + x[0] ** 4, -x[1]]), 2)
    with pytest.raises(UnsupportedError, match="polynomial"):
        unstable_manifold(ds4, np.zeros(2), order=3, degree=3, eigvec_scale=1.0, unstable_tol=1e-9)


def lorenz_ds():
    pb = ode_problem(["x", "y", "z"], {"sigma": 10.0, "r": 13.9, "b": 8 / 3},
                     lambda X, P: [P["sigma"] * (X["y"] - X["x"]), P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                   X["x"] * X["y"] - P["b"] * X["z"]], "Lorenz")
    return ir_system(pb, ["r"])


def test_lorenz_origin_manifold_converges_with_order():
    ds = lorenz_ds()
    d = [unstable_manifold(ds, np.zeros(3), order=K, degree=2, eigvec_scale=1.0, unstable_tol=1e-9,
                           par={"r": 28.0}).defect([0.5]) for K in (1, 3, 6, 10)]
    assert all(d[i + 1] < 1e-2 * d[i] for i in range(2)) and d[-1] < 1e-13


@pytest.mark.slow                   # 6.2 s in the fast suite (qual_durations0)
def test_lorenz_homoclinic_orbit_as_a_connection():
    """Origin → origin at r_hom: projection condition on the sphere |x| = 0.1 (declared), r the
    unknown. Converged value 13.9265574075 (job man_lorenz, stable to 1e-10 in δ, K, ρ, rtol);
    published 13.926 (Kaplan & Yorke 1979; Sparrow 1982)."""
    ds = lorenz_ds()

    def setup(q):
        par = {"r": float(q[0])}
        man = unstable_manifold(ds, np.zeros(3), order=10, degree=2, eigvec_scale=1.0,
                                unstable_tol=1e-9, par=par)
        return dict(x0=man([0.5]), integrator=ODEIntegrator(ds, method="DOP853", rtol=1e-11,
                                                            atol=1e-13, par=par),
                    x_plus=np.zeros(3), M=ds.M, start_defect=man.defect([0.5]), end_radius=0.1,
                    Psi=unstable_left_basis(ds, np.zeros(3), unstable_tol=1e-9, par=par))
    con = find_connection(setup, 8.0, bracket=(13.92, 13.93), xtol=1e-11)
    assert abs(con.unknowns[0] - 13.9265574075) < 1e-8
    assert abs(con.end_gap - 0.1) < 1e-10 and np.abs(con.projection).max() < 1e-7
    assert round(con.unknowns[0], 3) == 13.927 and abs(con.unknowns[0] - 13.926) < 1e-3
    assert "projection" in con.explain()


# ------------------------------------------------------------------ KS in U+
@pytest.fixture(scope="module")
def ks_up():
    s = ks_solver(L=22.0, N=32, dealias=1.5, scheme="RK443", timestep=0.02)
    nk = NewtonKrylovHookstep(tol=1e-13, max_iterations=40, gmres_rtol=1e-6, gmres_max_iterations=60,
                              delta0=1.0, delta_min=1e-10, delta_max=10.0)
    E = {}
    for name, (m, a) in {"E1": (1, 1.0), "E2": (2, 1.0), "E3": (3, 2.0)}.items():
        c = np.zeros(32)
        c[2 * m + 1] = -a
        E[name] = find_equilibrium(s, s.system.gather_state([c]), nk).X
    E["tE2"] = FourierOps(s).translate(E["E2"], 22.0 / 4)
    ds = ecs_system(s, odd=True, degree=2, dense_max=64)
    return s, ds, {k: ds.space.active(v) for k, v in E.items()}


@pytest.mark.slow                   # 2.1 s in the fast suite (qual_durations0)
def test_ks_unstable_dimensions_in_U_plus(ks_up):
    """In U+: E1 and E2 2D (pairs 0.0824 ± 0.3402i, 0.1390 ± 0.2384i: Cvitanović et al. 2010,
    Table 5.1), E3 1D (0.0933), τ_{L/4}E2 stable. Parametrization defect at ρ = 0.05, order 8."""
    s, ds, x = ks_up
    dims = {}
    for k in ("E1", "E2", "E3"):
        m = unstable_manifold(ds, x[k], order=8, degree=2, eigvec_scale=1.0, unstable_tol=1e-6)
        dims[k] = m.dim
        assert m.max_defect(0.05, 12) < 1e-7          # measured: E3 3.9e-8, E1, E2 below
        if k == "E2":
            assert abs(m.eigenvalues[0] - (0.1390 + 0.2384j)) < 1e-4
    assert dims == {"E1": 2, "E2": 2, "E3": 1}
    assert unstable_left_basis(ds, x["tE2"], unstable_tol=1e-6).shape[1] == 0
    assert unstable_left_basis(ds, x["E3"], unstable_tol=1e-6).shape[1] == 1


def test_ks_connection_setup_refuses_unreached_sphere(ks_up):
    """An orbit that never enters the end sphere is not a connection (explicit refusal)."""
    s, ds, x = ks_up
    integ = SolverIntegrator(s, dt=0.02, embed=ds.space.full, restrict=ds.space.active)
    setup = equilibrium_connection(ds, integ, x["E2"], x["E3"], order=4, degree=2, rho=0.05,
                                   eigvec_scale=1.0, unstable_tol=1e-6, end_radius=1e-3)
    from conatus.errors import ConvergenceError
    with pytest.raises(ConvergenceError):
        find_connection(setup, 2.0, q0=[0.3], fd_step=1e-6, xtol=1e-10, max_iterations=1)


# ------------------------------------------------------------------ periodic orbits (seeding)
def test_orbit_seeds_and_linear_stretch():
    """Linear seeding on a planar map with a known multiplier: the stretch → |Λ|."""
    Lam = -1.7
    seeds, notes = orbit_manifold_seeds(np.zeros(2), np.array([1.0, 0.0]), Lam, np.array([0.0, 1.0]),
                                        eps=1e-3, n_delta=4)
    assert seeds.shape == (8, 2) and any("flips" in n for n in notes)
    st = first_return_stretch(lambda x: np.array([Lam * x[0], 0.3 * x[1]]), np.zeros(2),
                              np.array([1.0, 0.0]), 1e-6)
    assert abs(st - abs(Lam)) < 1e-12
    with pytest.raises(OptionError):
        orbit_manifold_seeds(np.zeros(2), np.array([1.0, 1j]), 1.2 + 0.5j, np.array([0, 1.0]),
                             eps=1e-3, n_delta=2)
    # oblique section: the seeds are projected along the velocity onto the section (n · seed = 0)
    n = np.array([0.3, 1.0])
    seeds, _ = orbit_manifold_seeds(np.zeros(2), np.array([1.0, 0.2]), 2.0, np.array([0.1, 1.0]),
                                    eps=1e-3, n_delta=3, normal=n)
    assert np.abs(seeds @ n).max() < 1e-15


# ------------------------------------------------------------------ snaking helpers
def _sh23(N, gamma):
    from conatus.discretize import discretize
    from conatus.ir import IVP, CartesianCoordinates, Domain, Field, Parameter, RealFourier, dt, lap
    from conatus.timestep import IVPSolver
    cs = CartesianCoordinates("x")
    D = Domain(cs, [RealFourier(cs["x"], N, (-gamma / 2, gamma / 2), dealias=2)])
    u = Field("u", D)
    r = Parameter("r", -0.2)
    pb = IVP([u], name="SH23")
    pb.add_equation((dt(u) - r * u + u + 2 * lap(u) + lap(lap(u)), 1.8 * u * u - u * u * u))
    return u, ir_system(pb, ["r"]), IVPSolver(discretize(pb), "RK443", 0.1)


def test_restriction_to_even_states_and_periodic_fold():
    """SH23 (b2 = 1.8) on Γ = 2π: the even subspace is invariant (odd one is not: u² is even);
    the periodic branch folds at r_sn = −0.37437 (secondary source: −0.3744, arXiv:1411.1513 p. 4)."""
    from conatus.bifurcation import continue_equilibria
    from conatus.continuation import ContinuationSettings
    u, ds, sol = _sh23(16, 2 * np.pi)
    idx = even_subspace(sol)
    with pytest.raises(OptionError, match="not invariant"):
        restrict_system(ds, np.setdiff1d(np.arange(ds.n), idx), name="odd states")
    dr = restrict_system(ds, idx, name="even states")
    sol.set_state({u: lambda x: np.cos(x)})
    y, res, _ = newton_equilibrium(dr, sol.X[idx], tol=1e-12, max_iter=40, par={"r": -0.2})
    st = ContinuationSettings(ds=0.02, ds_min=1e-6, ds_max=0.05, adapt=1.3, fast_iterations=3,
                              tolerance=1e-10, max_newton=10, max_points=40, dp_fd=None)
    cur = continue_equilibria(dr, y, "r", st, direction=-1.0, marginal_tol=1e-9, tests=("fold",),
                              par={"r": -0.2}, bounds=(-0.6, 0.05))
    folds = [e.point.z[-1] for e in cur.events if e.kind == "fold" and e.converged]
    assert abs(folds[0] - (-0.3743701525)) < 1e-8 and round(folds[0], 4) == -0.3744
