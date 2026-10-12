"""An LBVP with a complex LHS and real data returns the complex solution (ERREURS_COURANTES).

Before the fix, Plan.solve() allocated real output arrays whenever the layout and the data were
real, and the complex solution of i u - u'' = 1 lost its imaginary part (numpy ComplexWarning only).
"""
import warnings

import numpy as np

from conatus.discretize import discretize
from conatus.ir import LBVP, CartesianCoordinates, Chebyshev, Domain, Field, diff, interp
from conatus.reference import transforms as T


def test_complex_lhs_real_data_keeps_imaginary_part():
    cs = CartesianCoordinates("y")
    y = cs["y"]
    b = Chebyshev(y, 24, (-1, 1), dealias=1)
    u = Field("u", Domain(cs, [b]), dtype=complex)
    pb = LBVP([u])
    pb.add_equation((1j * u - diff(diff(u, y), y), 1))           # real data, complex operator
    pb.add_bc(interp(u, y, -1), interp(u, y, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        sol = discretize(pb).solve()
    Y = T.grid(b, dealias=False)
    k = np.sqrt(1j)
    exact = -1j * (1 - np.cosh(k * Y) / np.cosh(k))
    assert np.abs(sol.grid("u") - exact).max() < 1e-13
