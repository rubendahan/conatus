"""Floquet winding with a double exponent near the circle (hunt 5D).

Oracle: the decoupled Hill system M = I, L0 = diag(-a_k), no coupling, has
det M(mu) = prod_n prod_k (mu + i n - a_k); for a radius 0.3 < 1/2 around c with |Im c| < 0.2 only
n = 0 matters and the winding is #{k : |a_k - c| < r}. The walk shared with contour.count aliased
the 2 pi turn of a double exponent 1e-3 from the circle (82 of 180 positions wrong); a refusal
(ConvergenceError) is allowed, a wrong integer is not.
"""
import numpy as np
import pytest

from conatus.errors import ConvergenceError
from conatus.floquet.hill_general import hill_from_blocks

C, R = 0.05 + 0.02j, 0.3


@pytest.mark.parametrize("d", [1e-3, -1e-3])
@pytest.mark.parametrize("n_points", [8, 32])
def test_double_exponent_near_the_circle(d, n_points):
    wrong = []
    for th in np.linspace(0, 2 * np.pi, 91)[:-1]:
        base = C + R * (1 - d) * np.exp(1j * th)
        a = [base, base + 1e-7, C + 0.05]
        truth = sum(abs(x - C) < R for x in a)
        hb = hill_from_blocks(np.eye(3), np.diag(-np.asarray(a, complex)), {}, Nh=2, omega=1.0,
                              label="t", note="t")
        try:
            w = hb.winding(C, R, n_points)
        except ConvergenceError:
            continue
        if w != truth:
            wrong.append((round(float(th), 3), w, truth))
    assert not wrong, wrong[:5]
