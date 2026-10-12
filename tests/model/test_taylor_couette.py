"""cn.TaylorCouette: annular cylinder ri <= r <= ro, z periodic (native layer).

1. Onset of Taylor vortices (outer cylinder at rest, radius ratio eta = 0.5): at Re = 68.1863,
   k = 3.1625 the least stable axisymmetric mode is neutral; the native model, written as on paper,
   gives the eigenvalue of the independent r^2-multiplied primitive-variable code
   (conatus.discretize.polar.taylor_couette_eigs) to round-off, as the expert IR does in
   tests/polar/test_polar_cylinder.py (same critical point as there).
2. Time stepping in the m = 0, k_z = 0 group (singular before the divergence tau of the annulus
   recipe): an azimuthal Bessel mode decays at its exact rate.
"""
import warnings

import numpy as np
import pytest

import conatus as cn
from conatus.errors import CoordinateError

ETA, RE_C, K_C = 0.5, 68.1863, 3.1625


def _tc_model(Re):
    ri, ro = ETA / (1 - ETA), 1 / (1 - ETA)
    geo = cn.TaylorCouette(ri, ro, length=2 * np.pi / K_C)
    m = cn.Model(geo, name="Taylor-Couette", Re=Re)
    u, p = m.velocity("u"), m.pressure("p")
    m.equations("dt(u) + grad(p) - lap(u)/Re = -u@grad(u)", "div(u) = 0")
    B = 1.0 / (1.0 / ri - ri / ro ** 2)
    A = -B / ro ** 2
    m.at("inner", u=0)
    m.at("outer", u=0)
    return m, u, p, (A, B), geo


def test_geometry_declaration():
    geo = cn.TaylorCouette(1.0, 2.0, length=3.0)
    assert geo.eta == 0.5 and "TaylorCouette(inner=1, outer=2, length=3)" == repr(geo)
    with pytest.warns(DeprecationWarning, match="read .eta"):
        assert geo.radius_ratio == 0.5
    assert set(geo.boundaries()) >= {"inner", "outer", "walls"}
    with pytest.raises(CoordinateError, match="0 < inner < outer"):
        cn.TaylorCouette(0.0, 1.0)


def test_taylor_vortex_onset_matches_independent_code():
    from conatus.discretize.polar import taylor_couette_eigs
    N = 24
    m, u, p, (A, B), _ = _tc_model(RE_C)
    base = lambda r: np.stack([A * r + B / r, 0 * r, 0 * r])      # noqa: E731  (phi, r, z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.linearize(about={u: base, p: 0}).eigenvalues((4, N, 4), m=0, kz=K_C)
    top = res.values[np.argmax(res.values.real)]
    ref = taylor_couette_eigs(N, ETA, RE_C, K_C)
    ref = ref[np.argmax(ref.real)]
    print(f"native top eigenvalue {top}, independent code {ref}")
    assert abs(top - ref) < 1e-11 and abs(top) < 1e-5
    assert res.residuals[np.argmax(res.values.real)] < 1e-12


def test_spin_down_exact_bessel_mode():
    """Time stepping in the group m = 0, k_z = 0 (formerly singular; regular since the
    annulus recipe carries a divergence tau): the azimuthal Bessel mode
    u_phi = J1(k r) Y1(k ri) - J1(k ri) Y1(k r) with u_phi(ro) = 0 decays as exp(-k^2 t / Re)
    exactly (its nonlinear term is radial, balanced by the pressure)."""
    from scipy.optimize import brentq
    from scipy.special import j1, y1
    ri, ro, Re = 1.0, 2.0, 20.0
    k = brentq(lambda q: j1(q * ri) * y1(q * ro) - j1(q * ro) * y1(q * ri), 2.0, 4.0)
    mode = lambda r: j1(k * r) * y1(k * ri) - j1(k * ri) * y1(k * r)       # noqa: E731
    m = cn.Model(cn.TaylorCouette(ri, ro, 2 * np.pi), name="spin-down", Re=Re)
    m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) + grad(p) - lap(u)/Re = -u@grad(u)", "div(u) = 0")
    m.at("walls", u=0)
    sim = m.simulate(N=(4, 24, 4), scheme="RK443", dt=1e-3)
    sim.initial(u=lambda phi, r, z: np.stack([mode(r) + 0 * phi + 0 * z, 0 * r + 0 * phi + 0 * z,
                                              0 * r + 0 * phi + 0 * z]))
    sim.run(steps=200)
    r = sim.coordinate("r")[None, :, None]
    g = sim.grid("u")
    exact = np.exp(-k ** 2 * sim.t / Re) * mode(r)
    assert np.abs(g[0] - exact).max() < 1e-9 * np.abs(mode(r)).max()
    assert np.abs(g[1:]).max() < 1e-12
