"""Symmetries of the equations act on the data: the solution of the transformed data is the
transformed solution.

  KS / Burgers (periodic x)  translation by a whole number of grid cells: np.roll of the grid
                             values; reflection u(x) -> -u(-x): the reflected grid values;
  Rayleigh-Benard (box)      x -> -x with u_x -> -u_x;
  disk                       rotation by one grid angle: np.roll along phi.
Translations of a whole number of cells are exact in the Fourier basis, so the relations hold to
rounding (INVARIANCE_ROUNDING), whatever the scheme.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import conatus as cn
from conatus._tolerances_internal import INTERNAL

RTOL = INTERNAL["INVARIANCE_ROUNDING"].value
L = 22.0
N = 32


def _rel(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(b)))


def _ks(scheme):
    m = cn.Model(cn.Box(x=cn.periodic(L)), name="KS")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    return m.simulate(N=N, scheme=scheme, dt=0.02)


def _f(x):
    return 0.6 * np.cos(2 * np.pi * x / L) + 0.4 * np.sin(4 * np.pi * x / L + 0.3) \
        + 0.2 * np.cos(6 * np.pi * x / L - 1.0)


@pytest.mark.parametrize("scheme", ["RK443", "SBDF3", "CNAB2"])
def test_ks_translation_and_reflection(scheme):
    h = L / N
    s = 5
    a = _ks(scheme).initial(u=_f).run(steps=50)
    b = _ks(scheme).initial(u=lambda x: _f(x - s * h)).run(steps=50)
    assert _rel(b.grid("u"), np.roll(a.grid("u"), s)) < RTOL
    c = _ks(scheme).initial(u=lambda x: -_f(-x)).run(steps=50)
    ua = a.grid("u")
    reflected = -np.roll(ua[::-1], 1)                 # x_j -> -x_j = x_{N-j}
    assert _rel(c.grid("u"), reflected) < RTOL


def _rb(sign):
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="RB", Ra=3000.0, Pr=1.0)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        div(u) = 0
        dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=0.5)
    m.at("top", u=0, T=-0.5)
    sim = m.simulate(N=(16, 12), scheme="RK222", dt=1e-3)
    sim.initial(T=lambda x, z: -z + 1e-2 * np.cos(np.pi * (sign * x) + 0.4) * np.cos(np.pi * z)
                + 5e-3 * np.sin(2 * np.pi * sign * x) * np.cos(np.pi * z))
    return sim.run(steps=60)


def test_rb_reflection_x():
    a, b = _rb(1.0), _rb(-1.0)
    Ta, Tb = a.grid("T"), b.grid("T")
    refl = lambda g: np.roll(g[::-1], 1, axis=0)       # noqa: E731  x -> -x on the periodic grid
    assert _rel(Tb, refl(Ta)) < RTOL
    ua, ub = a.grid("u"), b.grid("u")
    assert _rel(ub[0], -refl(ua[0])) < RTOL and _rel(ub[1], refl(ua[1])) < RTOL


def test_disk_rotation():
    def run(shift):
        m = cn.Model(cn.Disk(1.0), name="disk", kappa=0.1)
        m.scalar("T")
        m.equations("dt(T) - kappa*lap(T) = -T*T")
        m.at("wall", T=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = m.simulate(N=(16, 12), scheme="SBDF2", dt=1e-2)
            sim.initial(T=lambda phi, r: (1 - r**2) * (1 + 0.3 * r * np.cos(phi - shift)
                                                       + 0.2 * r**2 * np.sin(2 * (phi - shift))))
        return sim.run(steps=20)
    dphi = 2 * np.pi / 16
    a, b = run(0.0), run(3 * dphi)
    assert _rel(b.grid("T"), np.roll(a.grid("T"), 3, axis=0)) < RTOL
