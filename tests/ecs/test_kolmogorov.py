"""Kolmogorov 2D (Chandler & Kerswell 2013, J. Fluid Mech. 722, 554–595): laminar threshold."""
import numpy as np
import pytest

from conatus.ecs.fourier_ops import AxisOps
from conatus.ecs.kolmogorov import kolmogorov_solver, laminar_state, linear_stability_dense


def lead(Re, N):
    s = kolmogorov_solver(Re=Re, n=4, N=N, dealias=1.5, scheme="RK443", timestep=0.01)
    X = laminar_state(s, Re=Re, n=4)
    r = s.F(X, 0.0) - s.system.L @ X
    assert np.abs(r).max() < 1e-12                          # laminar state is steady
    ev = linear_stability_dense(s, X, dense_max=2000, finite_max=1e8)
    # k = 0 group: two algebraic rows, the gauge ∫ψ = 0 and the vorticity equation (lap ψ_0 = 0, so
    # it reads τ = <RHS>). Was 1 before the "bare unknown in M" fix (gallery):
    # the bare τ was put in M as well (dt τ + τ = <RHS>).
    assert ev.n_infinite == 2
    return ev[0]


@pytest.mark.slow                   # 4.4 s in the fast suite (qual_durations0)
def test_re_lin_matches_published_value():
    """Re_lin = 9.9669 (CK13, Fig. 3 caption, p. 567): the leading eigenvalue changes sign inside
    the rounding interval [9.96685, 9.96695]; it is real (steady bifurcation) at N = 24."""
    lo, hi = lead(9.96685, 24), lead(9.96695, 24)
    assert lo.real < 0 < hi.real
    assert abs(lo.imag) < 1e-10 and abs(hi.imag) < 1e-10


def test_axis_ops_generator_and_period():
    s = kolmogorov_solver(Re=40.0, n=4, N=16, dealias=1.5, scheme="RK443", timestep=0.01)
    rng = np.random.default_rng(0)
    X = rng.normal(size=s.system.n)
    for coord in ("x", "y"):
        ax = AxisOps(s, coord)
        h = 1e-6
        fd = (ax.translate(X, h) - ax.translate(X, -h)) / (2 * h)
        assert np.abs(fd - ax.ddx(X)).max() < 1e-8
        assert np.abs(ax.translate(X, 2 * np.pi) - X).max() < 1e-12


@pytest.mark.slow                   # 11.6 s in the fast suite (qual_durations0)
def test_steady_state_and_stability_dae():
    """E1 branch just above Re_lin (N = 16): Newton from the time-integrated state, then the
    generalized stability problem returns the marginal translation mode and a finite spectrum."""
    from conatus.ecs import NewtonKrylovHookstep
    from conatus.ecs.dae import find_relative_equilibrium_dae, stability_dae
    Re = 10.3
    s = kolmogorov_solver(Re=Re, n=4, N=16, dealias=1.5, scheme="RK443", timestep=0.05)
    lam = laminar_state(s, Re=Re, n=4)
    c = np.zeros((16, 16))
    c[2, 0] = 0.8                                     # large-scale kx = 1 mode (critical direction)
    s.X = lam + s.system.gather_state([c, np.zeros((1, 1))])
    s.run(4000)                                       # E1 is stable just above Re_lin
    nk = NewtonKrylovHookstep(tol=1e-12, max_iterations=30, gmres_rtol=1e-8, gmres_max_iterations=200,
                              delta0=1.0, delta_min=1e-12, delta_max=100.0)
    sol = find_relative_equilibrium_dae(s, s.X, 0.0, nk, coord="x")
    assert sol.residual < 1e-11 and abs(sol.params["c"]) < 1e-12
    assert np.linalg.norm(sol.X - lam) > 0.1 * np.linalg.norm(lam)      # not the laminar state
    st = stability_dae(s, sol, dense_max=2000, finite_max=1e8)
    assert abs(st["marginal"]) < 1e-8 and st["marginal_overlap"] > 0.99
    assert st["n_unstable"] == 0 and st["n_infinite"] >= 1


def test_dissipation_ratio_laminar_is_one():
    from conatus.ecs.kolmogorov import dissipation_ratio
    s = kolmogorov_solver(Re=40.0, n=4, N=16, dealias=1.5, scheme="RK443", timestep=0.01)
    assert abs(dissipation_ratio(s, laminar_state(s, Re=40.0, n=4), Re=40.0, n=4) - 1.0) < 1e-13
