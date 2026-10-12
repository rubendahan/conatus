"""Native default dealiasing on Hermite / Laguerre axes.

No factor is alias-free on a whole-line (Hermite) or half-line (Laguerre) axis: the functions carry
a Gaussian / exponential envelope, so a product is not a finite series in the same functions. The
factor only reduces aliasing; for products of the unknowns (degree p >= 2) the native default there
is max(2, (p+1)/2) instead of max(3/2, (p+1)/2), printed with its reason. Linear models keep 3/2,
and a value given with ``dealias=`` always wins.
"""
import numpy as np
import pytest

import conatus as cn
from conatus.model.compiler import dealias_factor
from conatus.model.run import _native_sentence
from tests.transforms.test_dealias_exactness import _product_error


def test_rule_values():
    assert [dealias_factor(p, True) for p in (1, 2, 3, 4)] == [1.5, 2.0, 2.0, 2.5]
    assert [dealias_factor(p) for p in (1, 2, 3, 4)] == [1.5, 1.5, 2.0, 2.5]


def _mixed():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), y=cn.line(scale=1.0)), name="nl")
    m.scalar("u")
    m.equations("dt(u) - lap(u) = -u*u")
    return m


def test_quadratic_model_gets_two_on_the_hermite_axis_only():
    c = _mixed().compile((16, 32))
    assert c.dealias == {"x": 1.5, "y": 2.0}
    assert c.bases["y"].dealias == 2.0 and c.bases["x"].dealias == 1.5
    short, full = c.explain(), c.explain(full=True)
    assert "Dealiasing: x: 3/2, y: 2" in short and "default there max(2, (p+1)/2)" in short
    assert "there the default is 2 = max(2, (p+1)/2), because the factor only reduces" in full
    assert "the default dealiasing factor is 2" in _native_sentence(c)


def test_half_line_and_explicit_value_wins():
    m = cn.Model(cn.Box(z=cn.half_line(scale=0.5)), name="nl")
    m.scalar("u")
    m.equations("dt(u) - lap(u) = -u*u*u")
    m.at("wall", u=0)
    assert m.compile(24).dealias == {"z": 2.0}                       # max(2, (3+1)/2)
    given = _mixed().compile((16, 32), dealias=1.5)
    assert given.dealias == {"x": 1.5, "y": 1.5} and "(given)" in given.explain()
    assert _mixed().compile((16, 32), dealias={"x": 2, "y": 3}).dealias == {"x": 2, "y": 3}


def test_linear_models_keep_three_halves():
    """No product of unknowns: the published linear unbounded-axis results are unchanged."""
    m = cn.Model(cn.Box(x=cn.line(scale=1.0)), name="QHO")
    m.scalar("u")
    m.eigenvalue("E", growth="real")
    m.given(V=lambda x: x**2)
    m.equations("E*u + lap(u) - V*u = 0")
    c = m.compile(32)
    assert c.dealias == {"x": 1.5}
    assert "there the default is" not in c.explain(full=True)


@pytest.mark.parametrize("name", ["hermite", "laguerre"])
def test_two_lowers_the_quadratic_aliasing(name):
    """The reason printed by explain(): 2 against 3/2 lowers the aliasing error of a quadratic
    product by one decade (unit spectra) to two (spectra decaying like e^{-0.3 n}), N = 16."""
    unit = [_product_error(name, d) for d in (1.5, 2.0)]
    dec = [_product_error(name, d, decay=0.3) for d in (1.5, 2.0)]
    assert unit[0] / unit[1] > 3, unit
    assert dec[0] / dec[1] > 8, dec
