"""F748 (audit of 0.2.1, N32): the defect is not monotone in the radius.

Oracle: x_t = x + x^2 has P(s) = s + s^2 exactly at order 2, relative defect |2 s^3 + s^4| / |s + 2 s^2| from the invariance equation, with brentq for the last crossing of tol.
"""
from _acceptance import check, passed, start

start(748, "manifold validity radius")
import numpy as np  # noqa: E402


from scipy.optimize import brentq  # noqa: E402

from conatus.bifurcation import ir_system, ode_problem  # noqa: E402
from conatus.manifolds import unstable_manifold  # noqa: E402

pb = ode_problem(["x"], {}, lambda x, p: [x["x"] + x["x"] ** 2])
ds = ir_system(pb, [])
um = unstable_manifold(ds, np.zeros(ds.n), order=2, degree=2, eigvec_scale=1.0, unstable_tol=1e-8)


def exact(r):
    return max(abs(2 * s ** 3 + s ** 4) / abs(s + 2 * s ** 2) for s in (r, -r))


tol, cap = 1.1, 1.1
got = um.validity_radius(tol, m=2, rho_max=cap)
grid = np.linspace(.5, cap, 2001)
ok = grid[[exact(r) <= tol for r in grid]]
last = ok.max()
root = brentq(lambda r: exact(r) - tol, last, last + (grid[1] - grid[0]))
check(abs(got - root) < 1e-6, f"{got} against {root}")
check(um.max_defect(got, 2) <= tol, "returned radius is not admissible")
small = um.validity_radius(1e-3, m=2, rho_max=.2)
check(exact(small) <= 1.01e-3 and small > 0, f"tight tolerance {small}")
passed(f"radius {got:.6f}")
