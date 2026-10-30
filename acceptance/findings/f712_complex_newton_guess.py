"""F712 (audit of 0.2.1, N39): a complex Newton guess on a constant domain.

Oracle: i is a root of u^2 = -1."""
from _acceptance import check, passed, start

start(712, "a complex Newton guess on a constant domain")
import numpy as np  # noqa: E402

from conatus import expert as X  # noqa: E402

cs = X.CartesianCoordinates("x")
u = X.Field("u", X.Domain(cs, []), dtype=complex)
pb = X.NLBVP([u], name="f712")
pb.add_equation((u**2, -1.))
sol = X.NewtonSolver(pb, tol=1e-12, max_iterations=30).solve({u: lambda *a: 1 + 1j})
check(sol.converged, "no convergence")
err = np.max(np.abs(sol.grid("u") - 1j))
check(err < 1e-10, f"root off by {err:.2e}")
passed(f"error {err:.1e}")
