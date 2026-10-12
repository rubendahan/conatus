"""cn.navier_slip(l): Navier partial slip u·n = 0, u_t + l dn(u_t) = 0 (native layer).

Exact check: Stokes shear modes u = u_x(z) e_x between walls z = 0 and 1 (kx = 0) decay at
sigma = -nu k^2 with u_x = l k cos(kz) + sin(kz), where k solves 2 l k cos k + (1 - l^2 k^2) sin k
= 0 (bottom: u_x = l u_x', top: u_x = -l u_x'). l = 0 gives the no-slip k = n pi. On a curved
wall the shear carries the curvature term (disk: u_phi = J1(k r) modes, below). Refusals:
sine/cosine axes, a negative or infinite slip length.
"""
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

import conatus as cn
from conatus.errors import BoundaryConditionError, InputTypeError

NU = 0.7


def _roots(l, n):
    f = lambda k: 2 * l * k * np.cos(k) + (1 - l * l * k * k) * np.sin(k)     # noqa: E731
    ks, grid = [], np.linspace(1e-6, 40, 40001)
    v = f(grid)
    for a, b, fa, fb in zip(grid[:-1], grid[1:], v[:-1], v[1:], strict=True):
        if fa * fb < 0:
            ks.append(brentq(f, a, b, xtol=1e-15))
    return np.array(ks[:n])


def _model(wall):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="stokes slip", nu=NU)
    u = m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = 0", "div(u) = 0")
    m.at("walls", u=wall)
    return m, u


@pytest.mark.parametrize("l", [0.0, 0.1, 0.5])
def test_shear_mode_decay_rates(l):
    m, u = _model(cn.navier_slip(l))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.linearize(about={u: 0}).eigenvalues(48, k=0.0)
    ev = np.asarray(res.values)
    for k in _roots(l, 4):
        s = -NU * k * k
        i = int(np.argmin(abs(ev - s)))
        assert abs(ev[i] - s) < 1e-9 * abs(s), (l, k, ev[i], s)
        assert res.residuals[i] < 1e-10
    if l == 0.0:
        assert np.allclose(_roots(0.0, 3), np.pi * np.arange(1, 4))


def test_explain_and_refusals():
    m, _ = _model(cn.navier_slip(0.25))
    txt = m.explain()
    assert "Navier slip" in txt and "u_t - 0.25 dz(u_t)" in txt and "u_t + 0.25 dz(u_t)" in txt
    assert repr(cn.navier_slip(0.25)) == "cn.navier_slip(0.25)"
    for bad in (-1.0, float("inf"), True):
        with pytest.raises(InputTypeError, match="slip length"):
            cn.navier_slip(bad)
    s = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1, basis="sincos")), name="sc", nu=NU)
    s.velocity("u")
    s.pressure("p")
    s.equations("dt(u) + grad(p) - nu*lap(u) = 0", "div(u) = 0")
    s.at("walls", u=cn.navier_slip(0.1))
    with pytest.raises(BoundaryConditionError, match="sine/cosine"):
        s.compile((8, 8))


@pytest.mark.parametrize("l", [0.0, 0.2, 1.0])
def test_disk_navier_slip_azimuthal_modes(l):
    """Disk of radius 1, m = 0 azimuthal modes u_phi = J1(k r): Navier slip at r = 1,
    u_phi + l r d/dr(u_phi / r) = 0, i.e. J1(k) + l (k J0(k) - 2 J1(k)) = 0 (l = 0: = 0,
    l -> infinity: = 0, the free-slip roots); decay -nu k^2."""
    from scipy.special import j0, j1
    m = cn.Model(cn.Disk(1.0), name="disk slip", nu=NU)
    u = m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = 0", "div(u) = 0")
    m.at("wall", u=cn.navier_slip(l))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.linearize(about={u: 0}).eigenvalues((8, 32), m=0)
    ev = np.asarray(res.values)
    f = lambda k: j1(k) + l * (k * j0(k) - 2 * j1(k))                    # noqa: E731
    grid = np.linspace(0.5, 20, 20000)
    v = f(grid)
    ks = [brentq(f, a, b) for a, b, fa, fb in zip(grid[:-1], grid[1:], v[:-1], v[1:], strict=True)
          if fa * fb < 0][:3]
    for k in ks:
        s = -NU * k * k
        i = int(np.argmin(abs(ev - s)))
        assert abs(ev[i] - s) < 1e-9 * abs(s), (l, k, ev[i], s)
    if l > 0:
        assert "Navier slip on the curved wall" in m.compile((8, 32)).explain()
