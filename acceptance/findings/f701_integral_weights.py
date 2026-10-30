"""F701 (audit of 0.2.1, N15): nonperiodic integrals in the batched RHS and its tangent.

Oracle: Gauss-Legendre quadrature of u^2 for a Jacobi expansion, and the exact 2 integ(u^2) along V = u."""
from _acceptance import check, passed, start

start(701, "nonperiodic integrals in the batched RHS and its tangent")
import numpy as np  # noqa: E402
from scipy.special import eval_jacobi  # noqa: E402

from conatus.ensembles import Ensemble  # noqa: E402
from conatus.expert import IVPSolver, discretize  # noqa: E402
from conatus.ir import CartesianCoordinates, Domain, Field, IVP, diff, dt, integ  # noqa: E402
from conatus.ir.bases import Jacobi  # noqa: E402

cs = CartesianCoordinates("x")
x = cs["x"]
dom = Domain(cs, [Jacobi(x, 8, (-1, 1), a=.5, b=-.25, dealias=2)])
u, v = Field("u", dom), Field("v", dom)
pb = IVP([u, v])
pb.add_equation((dt(u), diff(u * u, x)))
pb.add_equation((dt(v), integ(u * u)))
c = np.zeros(8)
c[0], c[1] = 1, .2
s = IVPSolver(discretize(pb), "RK443", .001)
s.X = s.system.gather_state([c, np.zeros_like(c)])
nodes, weights = np.polynomial.legendre.leggauss(16)
exact = float(weights @ (1 + .2 * eval_jacobi(1, .5, -.25, nodes)) ** 2)
rows = s.system.rows_obj == 1
en = Ensemble(s, 1, rhs="batched")
got = en.F(s.X[None])[0][rows][0]
check(abs(got - exact) < 1e-12, f"batched integral {got} vs {exact}")
jv = en.rhs.jvp(s.X[None], s.X[None, None])[0, 0][rows][0]
check(abs(jv - 2 * exact) < 1e-12, f"batched tangent {jv} vs {2 * exact}")
passed(f"integral {got:.12g}")
