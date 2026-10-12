"""An orbit search that lands on a steady state (equilibrium or travelling wave) is retracted.

Found by the first Studio Newton seed (run): a pre-periodic-orbit search
converged (residual 7e-12) onto the equilibrium E2 (‖v(u)‖/‖u‖ = 1.3e-11, energy 0.43816), since a
steady state solves S τ_ℓ Φ_T(u) = u for a continuum of (T, ℓ); it was returned as "provisional".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conatus.ecs import (NewtonKrylovHookstep, find_equilibrium, find_relative_periodic_orbit,
                         find_travelling_wave, ks_solver)
from conatus.ecs.solutions import STEADY_TOL, ECSSolution, _guard_steady, _Orbit

L22, N = 22.0, 32


def newton(it: int) -> NewtonKrylovHookstep:
    return NewtonKrylovHookstep(tol=1e-11, max_iterations=it, gmres_rtol=1e-6, gmres_max_iterations=80,
                                delta0=0.5, delta_min=1e-10, delta_max=5.0)


def test_equilibrium_found_by_an_orbit_search_is_retracted():
    s = ks_solver(L=L22, N=N, dealias=1.5, scheme="RK443", timestep=0.05)
    c = np.zeros(N)
    c[2 * 2 + 1] = -1.0                                  # E2 seed (as the)
    e2 = find_equilibrium(s, s.system.gather_state([c]), newton(40))
    assert abs(e2.energy - 0.4382) < 1e-4
    sol = find_relative_periodic_orbit(s, e2.X, 3.0, 0.0, newton(10), steps=60, reflect=True)
    assert sol.residual < 1e-10                           # it does solve the orbit equations…
    assert sol.status == "retracted"                      # …but it is not an orbit
    assert any("steady state" in n for n in sol.notes) and any(f"{STEADY_TOL:g}" in n for n in sol.notes)


@pytest.mark.slow                                          # ~15 s (TW Newton then RPO Newton)
def test_travelling_wave_found_by_an_orbit_search_is_retracted():
    s = ks_solver(L=L22, N=N, dealias=1.5, scheme="RK443", timestep=0.05)
    rng = np.random.default_rng(0)
    for _ in range(8):                                    # TW1 seed of tests/ecs/test_ecs_ks.py
        c = np.zeros(N)
        c[2:10] = rng.normal(size=8) * 0.8
    tw = find_travelling_wave(s, s.system.gather_state([c]), 0.737, newton(60))
    assert abs(tw.params["c"] - 0.73699) < 1e-4
    T = 2.0
    sol = find_relative_periodic_orbit(s, tw.X, T, tw.params["c"] * T, newton(10), steps=40)
    assert sol.residual < 1e-10 and sol.status == "retracted"
    assert any("steady state" in n for n in sol.notes)


def test_a_true_orbit_is_not_retracted():
    """The PPO 10.25 fixture (Conatus orbit) is far from steady: the guard leaves it alone."""
    d = json.loads((Path(__file__).parent / "data" / "ks_l22_ppo10.json").read_text())
    s = ks_solver(L=L22, N=N, dealias=1.5, scheme="RK443", timestep=d["T"] / d["nsteps"])
    X = s.system.gather_state([np.array(d["coeffs"])])
    sol = ECSSolution("preperiodic_orbit", X, {"T": d["T"]}, 0.0, None, 0.0)
    _guard_steady(sol, _Orbit(s, d["nsteps"], True))
    # shape change 6.1e-01 over the period (the decision); comoving velocity 2.8e-01 (reported)
    amp = float(sol.notes[-1].split("‖u_0‖ = ")[1].split(" ")[0])
    assert sol.status == "provisional" and abs(amp - 0.613) < 2e-3
    assert "2.821e-01" in sol.notes[-1]
