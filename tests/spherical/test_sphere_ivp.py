"""Spherical IVPs: exact decay rates (the scale of M, ERREURS_COURANTES) and temporal orders."""
import numpy as np
import pytest
from scipy.special import spherical_jn

import conatus.expert as C
from conatus.discretize import discretize


def _s2_heat(scheme, dt, T=0.2, R=1.3):
    s2 = C.S2Coordinates("phi", "theta")
    D = C.Domain(s2, [C.RealFourier(s2["phi"], 8, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(s2["theta"], 6, radius=R, dealias=3 / 2)])
    u = C.Field("u", D)
    pb = C.IVP([u])
    pb.add_equation((C.dt(u) - C.lap(u), 0))
    plan = discretize(pb)
    s = C.IVPSolver(plan, scheme, dt)
    # l = 2 (x z) and l = 1 (z): decay e^{-l(l+1) t / R^2}
    s.set_state({u: lambda P, Th: np.sin(Th) * np.cos(Th) * np.cos(P) + 0.5 * np.cos(Th)})
    s.run(int(round(T / dt)))
    P, Th = np.meshgrid(*plan.space.grids(False), indexing="ij")
    exact = (np.exp(-6 * T / R ** 2) * np.sin(Th) * np.cos(Th) * np.cos(P)
             + 0.5 * np.exp(-2 * T / R ** 2) * np.cos(Th))
    return np.abs(s.grid("u") - exact).max()


def test_sphere_heat_decay_and_orders():
    assert _s2_heat("RK443", 1e-3) < 1e-9                     # O(dt^3) with lambda = 3.6
    e1, e2 = _s2_heat("SBDF2", 0.02), _s2_heat("SBDF2", 0.01)
    assert 1.8 < np.log2(e1 / e2) < 2.3
    e1, e2 = _s2_heat("RK222", 0.02), _s2_heat("RK222", 0.01)
    assert 1.8 < np.log2(e1 / e2) < 2.3


@pytest.mark.parametrize("kind", ["ball", "shell"])
def test_3d_heat_decay_dirichlet(kind):
    """u = j_0(k r) (ball, k = pi) or the first radial Dirichlet mode of the shell, times l = 0:
    decays at exactly k^2 (the M scale and the tau rows together)."""
    sc = C.SphericalCoordinates("phi", "theta", "r")
    r = sc["r"]
    br = (C.BallBasis(r, 20, 1.0, dealias=3 / 2) if kind == "ball"
          else C.ShellBasis(r, 20, (0.5, 1.5), dealias=3 / 2))
    D = C.Domain(sc, [C.RealFourier(sc["phi"], 4, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(sc["theta"], 3, dealias=3 / 2), br])
    u = C.Field("u", D)
    taus = [C.Field(f"t{i}", D.without(r)) for i in range(1 if kind == "ball" else 2)]
    pb = C.IVP([u] + taus)
    pb.add_equation((C.dt(u) - C.lap(u) + sum(C.lift(t, br, -1 - i, lam=2) for i, t in enumerate(taus)), 0))
    pb.add_equation((C.interp(u, r, "right"), 0))
    if kind == "shell":
        pb.add_equation((C.interp(u, r, "left"), 0))
    plan = discretize(pb)
    k = np.pi                                    # ball: j_0(pi) = 0; shell: sin(pi (r - 1/2)) / r
    f = ((lambda P, Th, R: spherical_jn(0, k * R)) if kind == "ball"
         else (lambda P, Th, R: np.sin(k * (R - 0.5)) / R))
    s = C.IVPSolver(plan, "RK443", 1e-3)
    s.set_state({u: f})
    T = 0.05
    s.run(int(round(T / 1e-3)))
    P, Th, R = np.meshgrid(*plan.space.grids(False), indexing="ij")
    # RK443 error O((k^2 dt)^3) ~ 1e-8; a wrong M scale would give O(1)
    assert np.abs(s.grid("u") - np.exp(-k * k * T) * f(P, Th, R)).max() < 1e-7
