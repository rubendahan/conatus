"""The helpers shared by the polar and spherical discretizations (external review of 0.1.1, F31).

The two geometries carried identical copies of the spin-basis dot product and of the constant
evaluator. They now import one implementation; these tests check it against independent oracles
(``numpy.einsum`` with the metric, direct arithmetic) and that neither geometry keeps a copy.
"""
from __future__ import annotations

import numpy as np
import pytest

from conatus.discretize import curvilinear_common as cc
from conatus.discretize import polar_eval, polar_plan, sphere_compiler, sphere_eval
from conatus.discretize.compiler import DiscretizationError
from conatus.ir.expr_base import Constant, Parameter


@pytest.mark.parametrize("ra, rb", [(1, 1), (2, 1), (1, 2)])
def test_spin_dot_is_the_metric_contraction(ra, rb):
    rng = np.random.default_rng(3)
    nc, grid = 3, (5, 4)
    G = np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 1.0]])         # polar spin metric
    a = rng.standard_normal((nc,) * ra + grid)
    b = rng.standard_normal((nc,) * rb + grid)
    # oracle: sum_{s,t} G[s,t] a[..., s, grid] b[t, ..., grid]
    la = "ijk"[:ra - 1]
    lb = "lmn"[:rb - 1]
    want = np.einsum(f"st,{la}sxy,t{lb}xy->{la}{lb}xy", G, a, b)
    np.testing.assert_allclose(cc.spin_dot(a, ra, b, rb, G), want, rtol=1e-14, atol=1e-14)


def test_spin_dot_refuses_a_scalar():
    with pytest.raises(DiscretizationError):
        cc.spin_dot(np.ones((4,)), 0, np.ones((3, 4)), 1, np.eye(3))


def test_const_value_evaluates_constants_parameters_powers_products():
    Re = Parameter("Re", 400.0)
    param = {"Re": 400.0}
    f = Constant(2.0) * Re ** 2
    assert cc.const_value(f, lambda p: param[p.name]) == pytest.approx(2.0 * 400.0 ** 2, rel=1e-15)
    assert cc.const_value(Constant(3.5), lambda p: None) == 3.5


def test_both_geometries_use_the_one_implementation():
    assert polar_eval._dot is cc.spin_dot and sphere_eval._dot is cc.spin_dot
    for mod in (polar_plan, sphere_compiler):
        src = open(mod.__file__).read()
        assert "isinstance(f, E.Power) and not f.free_fields()" not in src, mod.__name__
