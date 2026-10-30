"""F750 (audit of 0.2.1, N41): Newton cannot converge on data that overflowed.

Oracle: u^2 = 1 at u = 1e154 has residual about 1e308 by hand.
"""
from _acceptance import check, passed, start

start(750, "Newton after arithmetic overflow")
import warnings  # noqa: E402


from conatus.errors import ConatusError  # noqa: E402
from conatus.expert import CartesianCoordinates, Domain, Field, NLBVP, NewtonSolver  # noqa: E402


def case():
    cs = CartesianCoordinates("x")
    D = Domain(cs, [])
    u = Field("u", D, dtype=float)
    pb = NLBVP([u], name="root")
    pb.add_equation((u ** 2, 1.))
    return u, NewtonSolver(pb, tol=1e-12, max_iterations=30)


for x0 in (1e154, 1e155, 1e160):
    for required in (False, True):
        u, s = case()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sol = s.solve({u: lambda x, x0=x0: x0 + 0 * x}, require_convergence=required)
            except (ConatusError, ArithmeticError, ValueError):
                continue
        check(not sol.converged, f"{x0}: reported converged")
        check(not sol.residual == 0., f"{x0}: residual zero")
u, s = case()
sol = s.solve({u: lambda x: 2.0 + 0 * x})
check(sol.converged and abs(abs(sol.coeffs["u"].ravel()[0]) - 1) < 1e-10, "control root")
passed("never converged")
