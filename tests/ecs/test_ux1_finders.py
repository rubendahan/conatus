"""The invariant-solution finders accept the native objects and refuse,
with their scope and the alternatives, problems outside one periodic real-Fourier direction."""
import numpy as np
import pytest

import conatus as cn
from conatus.ecs import NewtonKrylovHookstep, find_equilibrium, find_relative_periodic_orbit
from conatus.ecs.solutions import SCOPE
from conatus.errors import InputTypeError, UnsupportedError

N = 32


def _ks_model():
    m = cn.Model(cn.Box(x=cn.periodic(22.0)), name="KS")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    return m


def _seed(solver, m=1, a=1.0):
    c = np.zeros(N)
    c[2 * m + 1] = -a                       # a sin(k_m x) in the (cos, -sin) convention
    return solver.system.gather_state([c])


def test_native_simulation_and_model_are_accepted():
    nk = NewtonKrylovHookstep(tol=1e-12)                  # algorithm controls: declared defaults
    sim = _ks_model().simulate(N=N, scheme="RK443", dt=0.01)
    e1 = find_equilibrium(sim, _seed(sim.solver), nk)
    assert e1.residual < 1e-12 and abs(e1.energy - 0.2609) < 6e-5     # KS E1, L = 22
    e1m = find_equilibrium(_ks_model(), _seed(sim.solver), nk, N=N)
    assert np.allclose(e1m.X, e1.X, atol=1e-11)
    assert any("placeholder" in n for n in e1m.notes)


def test_outside_the_scope_is_refused_with_the_alternatives():
    nk = NewtonKrylovHookstep(tol=1e-12)
    rb = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2000, Pr=1)
    sim = rb.simulate(N=(8, 8), scheme="RK443", dt=0.01)
    with pytest.raises(UnsupportedError, match=r"(?s)RealFourier.*Chebyshev.*AffineSteadyProblem"
                                               r".*PipeTravellingWave"):
        find_equilibrium(sim, np.zeros(sim.solver.system.n), nk)
    assert "one periodic direction" in SCOPE


def test_orbit_finder_needs_a_scheme():
    nk = NewtonKrylovHookstep(tol=1e-12)
    with pytest.raises(InputTypeError, match="no time scheme"):
        find_relative_periodic_orbit(_ks_model(), np.zeros(N), 10.0, 0.0, nk, steps=10)
    with pytest.raises(InputTypeError, match="needs its resolution"):
        find_equilibrium(_ks_model(), np.zeros(N), nk)
