"""Pressure of a normal-mode model in the native layer: the onset of
Rayleigh-Benard convection written in primitive variables u, w, p, T for modes exp(i k x + s t)
on one wall axis is compiled without user taus (the native layer places them) and has the
spectrum of the w-theta form; the refusals that used to loop into tau placement name
m.pressure instead."""
import numpy as np
import pytest

import conatus as cn
import conatus.expert as X
from conatus.errors import BoundaryConditionError


def _primitive(declare="pressure", p_condition=False):
    m = cn.Model(cn.Box(z=cn.walls(-0.5, 0.5)), name="RB onset", Ra=1708.0, Pr=1.0, k=3.1)
    m.scalar("u")
    m.scalar("w")
    getattr(m, declare)("p")
    m.scalar("T")
    m.eigenvalue("s", growth="real")
    m.define("L(f) = dz(dz(f)) - k**2*f")
    m.equations("s*u + 1j*k*p - Pr*L(u) = 0",
                "s*w + dz(p) - Pr*L(w) - Ra*Pr*T = 0",
                "1j*k*u + dz(w) = 0",
                "s*T - w - L(T) = 0")
    m.at("walls", u=0, w=0, T=0)
    if p_condition:
        m.at("bottom", p=0)
    return m


def _w_theta():
    m = cn.Model(cn.Box(z=cn.walls(-0.5, 0.5)), name="RB onset", Ra=1708.0, Pr=1.0, k=3.1)
    m.scalar("w")
    m.scalar("T")
    m.eigenvalue("s", growth="real")
    m.define("L(f) = dz(dz(f)) - k**2*f")
    m.equations("s*L(w) - Pr*L(L(w)) + Pr*Ra*k**2*T = 0", "s*T - w - L(T) = 0")
    m.at("walls", "w = 0", "dz(w) = 0", "T = 0")
    return m


def test_primitive_form_has_the_spectrum_of_the_w_theta_form():
    m = _primitive()
    a = m.eigenvalues(24, drift=32, drift_tol=1e-6, Ra=2500.0, k=np.pi)
    b = _w_theta().eigenvalues(24, drift=32, drift_tol=1e-6, Ra=2500.0, k=np.pi)
    sa, sb = a.values[a.order()[:3]], b.values[b.order()[:3]]
    assert np.allclose(sa, sb, rtol=1e-9, atol=1e-9)
    assert abs(sa[0].real - 5.5425059) < 1e-6                 # Reid & Harris units
    ex = m.explain(24, full=True)
    assert "Normal-mode pressure recognized: p" in ex and "divergence tau" in ex


def test_critical_point_of_the_primitive_form():
    P = _primitive().parametric(24, over=("Ra", "k"), drift=32, drift_tol=1e-6)
    crit = X.critical(P, "Ra", bracket=(1500.0, 2000.0), over={"k": (2.5, 3.8)}, xtol=1e-9)
    assert abs(crit.value - 1707.7618) < 1e-3 and abs(crit.over["k"] - 3.11632) < 1e-4


def test_refusals_name_the_pressure_declaration():
    with pytest.raises(BoundaryConditionError, match=r"declare it with m.pressure\('p'\)"):
        _primitive("scalar").eigenvalues(16)
    with pytest.raises(BoundaryConditionError, match=r"looks like the pressure of a normal-mode"):
        _primitive("scalar", p_condition=True).eigenvalues(16)
    with pytest.raises(BoundaryConditionError, match="a pressure takes no condition"):
        _primitive("pressure", p_condition=True).eigenvalues(16)
