"""Hopf and pitchfork coefficients of non-canonical planar systems against textbook closed forms
(numerical audit).

Hopf: for x' = mu x - w y + f(x, y), y' = w x + mu y + g(x, y) with f, g of degree 2 and 3,
Guckenheimer & Holmes (1983) eq. (3.4.11) give the real part of the cubic normal-form coefficient

    a = (f_xxx + f_xyy + g_xxy + g_yyy) / 16
        + (f_xy (f_xx + f_yy) - g_xy (g_xx + g_yy) - f_xx g_xx + f_yy g_yy) / (16 w),

the quadratic terms entering through the centre-manifold (second-order) correction. In the
code's normalisation (||q|| = 1, p^H q = 1) the canonical system z' = (mu + i w) z + (a + i b) z|z|^2
has c1 = 2 (a + i b) (tests/bifurcation/test_bifurcation_core.py), and x + i y is that z, so
Re c1 = 2 a and l1 = Re c1 / w = 2 a / w for every f, g.

Pitchfork with a slaved mode: x' = mu x - x y, y' = -y + c x^2 has the centre manifold
y = c x^2 + O(x^4), hence x' = mu x - c x^3: a3 / (q_x^3 p_x) = -c, branch x^2 = mu / c.
"""
from __future__ import annotations

import numpy as np
import pytest

from conatus.bifurcation import ir_system, normal_form, ode_problem


def _gh_a(F, G, w):
    """GH (3.4.11) from the coefficient dicts {(i, j): c} of f = sum c x^i y^j and g."""
    def d(P, i, j):                               # d^{i+j} P / dx^i dy^j at 0
        return P.get((i, j), 0.0) * float(np.prod(range(1, i + 1)) * np.prod(range(1, j + 1)))
    f = lambda i, j: d(F, i, j)                   # noqa: E731
    g = lambda i, j: d(G, i, j)                   # noqa: E731
    return ((f(3, 0) + f(1, 2) + g(2, 1) + g(0, 3)) / 16
            + (f(1, 1) * (f(2, 0) + f(0, 2)) - g(1, 1) * (g(2, 0) + g(0, 2))
               - f(2, 0) * g(2, 0) + f(0, 2) * g(0, 2)) / (16 * w))


MONO = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]


@pytest.mark.parametrize("seed, w", [(1, 1.0), (2, 2.3), (3, 0.6)])
def test_planar_hopf_matches_guckenheimer_holmes(seed, w):
    rng = np.random.default_rng(seed)
    F = {m: float(rng.uniform(-1, 1)) for m in MONO}
    G = {m: float(rng.uniform(-1, 1)) for m in MONO}

    def poly(C, X):
        out = 0
        for (i, j), c in C.items():
            t = c
            for _ in range(i):
                t = t * X["x"]
            for _ in range(j):
                t = t * X["y"]
            out = out + t
        return out

    pb = ode_problem(["x", "y"], {"mu": 0.0},
                     lambda X, P: [P["mu"] * X["x"] - w * X["y"] + poly(F, X),
                                   w * X["x"] + P["mu"] * X["y"] + poly(G, X)], "planar Hopf")
    ds = ir_system(pb, ["mu"])
    nf = normal_form(ds, np.zeros(2), kind="hopf", parameter="mu")
    a = _gh_a(F, G, w)
    assert abs(nf.omega - w) < 1e-12
    assert abs(nf.c1.real - 2 * a) < 1e-12 * max(1.0, abs(a)), (nf.c1, a)
    assert abs(nf.l1 - 2 * a / w) < 1e-12 * max(1.0, abs(a / w))
    assert abs(nf.dlambda - 1) < 1e-12


@pytest.mark.parametrize("c", [0.5, 2.0])
def test_slaved_pitchfork_cubic_coefficient(c):
    pb = ode_problem(["x", "y"], {"mu": 0.0},
                     lambda X, P: [P["mu"] * X["x"] - X["x"] * X["y"],
                                   -1.0 * X["y"] + c * X["x"] * X["x"]], "slaved pitchfork")
    ds = ir_system(pb, ["mu"])
    S = np.diag([-1.0, 1.0])
    nf = normal_form(ds, np.zeros(2), kind="zero", parameter="mu", symmetry=S)
    assert nf.kind == "pitchfork"
    # branch x^2 = -mu/a3 * q_x^2 (the convention of test_lorenz_pitchfork_symmetry_and_hopf)
    assert abs(-nf.mu / nf.a3 * nf.q[0] ** 2 - 1 / c) < 1e-12
