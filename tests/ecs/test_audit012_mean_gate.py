"""Audit of 0.1.2, F01 and F02 (orbit side): invariant-solution searches keep the seed's mean and
refuse a result whose full equations are not satisfied; the orbit map of a period does not change
when another period was evaluated in between.

Oracles: the exact travelling wave 1 + cos x of u_t + u_x = 0 (speed 1, mean 1); u_t = 1 has no
equilibrium (its mean row is 1 everywhere); the implicit-Euler map of u_t = -u,
Phi_T u = (1 + T/n)^-n u (closed form).
"""
import numpy as np
import pytest

import conatus as cn
from conatus.ecs import NewtonKrylovHookstep, find_travelling_wave
from conatus.ecs.fourier_ops import FourierOps
from conatus.ecs.solutions import _Orbit
from conatus.errors import NewtonNotConverged


def _sim(eq, u0, dt=.01, scheme="RK443", n=16):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations(eq)
    s = m.simulate(N=n, scheme=scheme, dt=dt)
    s.initial(u=u0)
    return s


def test_travelling_wave_keeps_the_seed_mean():
    """1 + cos x moves at speed 1 unchanged: the returned wave is the seed itself (mean 1),
    not cos x (the mean was set to 0)."""
    s = _sim("dt(u)+dx(u)=0", lambda x: 1 + np.cos(x))
    x = s.solver.X.copy()
    sol = find_travelling_wave(s, x, 1.0, NewtonKrylovHookstep(tol=1e-10))
    k = FourierOps(s.solver).mean_index
    assert abs(x[k]) > 0.5                      # the seed has a mean
    assert np.allclose(sol.X, x, atol=1e-12)    # the exact wave is the seed itself
    assert abs(sol.params["c"] - 1.0) < 1e-10
    assert np.mean(s.grid("u")) == pytest.approx(1.0, abs=1e-12)   # seed grid mean, analytic 1
    assert any("full residual" in n for n in sol.notes)


def test_nonconserved_mean_is_refused():
    """u_t = 1: no equilibrium exists; the full residual (mean row) is 1, the search refuses."""
    s = _sim("dt(u)=1", lambda x: np.cos(x))
    with pytest.raises(NewtonNotConverged, match="full equations are not satisfied"):
        find_travelling_wave(s, s.solver.X.copy(), 0.0, NewtonKrylovHookstep(tol=1e-10))


def test_orbit_map_of_a_period_is_the_same_after_another_period():
    """G(X, 1), G(X, 2), G(X, 1): the two G(X, 1) agree, and equal the closed form
    ((1 + 1/10)^-10 - 1) X of implicit Euler on u_t = -u."""
    s = _sim("dt(u)+u=0", lambda x: np.cos(x), dt=.1, scheme="RK111", n=8)
    x = s.solver.X.copy()
    orb = _Orbit(s.solver, 10, False)
    a = orb.G(x, 1.0, 0.0)
    orb.G(x, 2.0, 0.0)
    b = orb.G(x, 1.0, 0.0)
    exact = ((1 + 0.1) ** -10 - 1) * x
    assert np.allclose(a, exact, rtol=1e-12, atol=1e-15)
    assert np.allclose(b, exact, rtol=1e-12, atol=1e-15)
    assert s.solver.dt == 0.1          # the solver's own step is put back


def test_orbit_period_column_is_the_derivative_of_the_discrete_map():
    """F10: Phi_T u = (1 + T/n)^-n u for implicit Euler on u_t = -u, so
    dPhi_T/dT = -(1 + T/n)^-(n+1) u (closed form). The endpoint velocity -Phi_T u used before
    was off by the factor 1 + T/n (relative 0.1 at n = 10, T = 1)."""
    s = _sim("dt(u)+u=0", lambda x: np.cos(x), dt=.1, scheme="RK111", n=8)
    x = s.solver.X.copy()
    orb = _Orbit(s.solver, 10, False)
    col = orb.dG(x, 1.0, 0.0, np.zeros_like(x), 1.0, 0.0)
    exact = -(1 + 0.1) ** -11 * x
    assert np.linalg.norm(col - exact) <= 1e-8 * np.linalg.norm(exact)
