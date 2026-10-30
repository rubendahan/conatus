"""F742 (audit of 0.2.1, N24): an antilinear right-hand side has no complex matrix.

Oracle: u_t = conj(u) on a complex state: F(i v) = -i F(v). The control u_t = (1+2j) u is complex-linear and the dense result is its multiplier.
"""
from _acceptance import check, passed, start

start(742, "dense FlowMap refuses an antilinear RHS")
import numpy as np  # noqa: E402


from conatus.adjoint import FlowMap  # noqa: E402
from conatus.errors import UnsupportedError  # noqa: E402
from conatus.expert import (CartesianCoordinates, ComplexFourier, Domain, Field, IVP, IVPSolver,  # noqa: E402
                            conj, discretize, dt)

cs = CartesianCoordinates("x")
D = Domain(cs, [ComplexFourier(cs["x"], 4, (0, 2 * np.pi), dealias=1)])


def solver(rhs):
    u = Field("u", D, dtype=complex)
    pb = IVP([u])
    pb.add_equation((dt(u), rhs(u)))
    return IVPSolver(discretize(pb), "RK222", .1)


s = solver(conj)
s.X[:] = 1 + 1j
try:
    FlowMap(s, 1, f_transpose="dense").vjp(s.X.copy(), np.ones_like(s.X))
    check(False, "conj(u) accepted")
except UnsupportedError:
    pass
s = solver(lambda u: (1 + 2j) * u)
s.X[:] = 1 + 1j
w = np.ones_like(s.X)
got = FlowMap(s, 1, f_transpose="dense").vjp(s.X.copy(), w)
h = .1 * (1 + 2j)
check(np.allclose(got, (1 + h + h * h / 2) * w, rtol=1e-11, atol=0), f"{got[0]}")
passed("refused, linear control exact")
