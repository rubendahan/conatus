"""The steady-state guard of the orbit finders is relative to the orbit's amplitude: it can no longer retract a genuine small orbit near a Hopf point.

u = M + ε (cos x + cos 2x) under u_t + u_xxx = 0 on [0, 2π) is a periodic orbit of period 2π (the
modes turn at frequencies 1 and 8), not a travelling wave (two phase speeds), whatever ε: the shape
of a small oscillation about the steady state M. Its comoving velocity relative to ‖u‖ is ~ 8 ε / M:
the former test (velocity below 1e-2 ‖u‖) retracted it as "steady" for ε / M < 1e-3. The new test
measures the shape change over the period, ~ ε / M, against STEADY_TOL x the accuracy of the solve."""
import numpy as np
import pytest

import conatus as cn
from conatus.ecs.solutions import STEADY_TOL, ECSSolution, _guard_steady, _Orbit

L = 2 * np.pi


def _small_orbit(eps, M=1.0, nsteps=400):
    m = cn.Model(cn.Box(x=cn.periodic(L)), name="linear dispersion")
    m.scalar("u")
    m.equations("dt(u) + dx(dx(dx(u))) = 0")
    sim = m.simulate(N=16, scheme="RK443", dt=L / nsteps)
    sim.initial(u=lambda x: M + eps * (np.cos(x) + np.cos(2 * x)))
    return sim.solver, sim.solver.X.copy(), nsteps


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_a_small_orbit_near_a_steady_state_is_kept(eps):
    s, X, n = _small_orbit(eps)
    orb = _Orbit(s, n, False)
    v, dx = orb.v(X), orb.ops.ddx(X)
    c = -float(v @ dx) / float(dx @ dx)
    velocity = float(np.linalg.norm(v + c * dx) / np.linalg.norm(X))
    assert velocity < 1e-2                    # the former guard (1e-2 on this number) retracted it
    sol = ECSSolution("relative_periodic_orbit", X, {"T": L, "ell": 0.0}, 1e-12, None, 0.0)
    _guard_steady(sol, orb)
    assert sol.status == "provisional", sol.notes
    note = sol.notes[-1]
    assert "shape change over the period" in note and f"{STEADY_TOL:g} x" in note
    amp = float(note.split("= ")[1].split(" ")[0])
    assert 0.3 * eps < amp < 10 * eps         # the amplitude of the orbit, relative to ‖u‖ ~ M


def test_the_same_state_without_oscillation_is_retracted():
    s, X, n = _small_orbit(0.0)
    sol = ECSSolution("relative_periodic_orbit", X, {"T": L, "ell": 0.0}, 1e-12, None, 0.0)
    _guard_steady(sol, _Orbit(s, n, False))
    assert sol.status == "retracted"
