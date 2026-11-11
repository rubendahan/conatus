"""F923 (C9): the collocation orbit tools state what |mu_triv - 1| measures and store a check on
all multipliers.

Oracle: the Hopf normal form r' = r (p - r^2), theta' = 1 plus a decoupled z' = -c z, written in
(x, y, z): the orbit r = sqrt(p) has multipliers 1, exp(-4 pi p) and exp(-2 pi c), product
exp(T * mean trace). The Liouville defect is small on a resolved mesh, and much larger on a mesh
too coarse for the fast transverse rate c (whose trivial multiplier still reads accurate).
"""
import numpy as np
from _acceptance import check, passed, start

start(923, "orbit multipliers: trivial error described as such; Liouville defect stored")
import conatus.bifurcation.orbits as O  # noqa: E402

doc = O.__doc__
check("accuracy of the multipliers" not in doc, "module docstring still calls |mu_triv - 1| the accuracy of all multipliers")
check("Liouville" in doc, "Liouville defect not documented")


def run(ntst, c, p=0.25):
    from conatus.bifurcation import DynamicalSystem

    def G(u, par):
        x, y, z = u
        r2 = x * x + y * y
        return np.array([x * (par["p"] - r2) - y, y * (par["p"] - r2) + x, -c * z])

    def J(u, par):
        x, y, z = u
        r2 = x * x + y * y
        return np.array([[par["p"] - r2 - 2 * x * x, -2 * x * y - 1, 0.0],
                         [-2 * x * y + 1, par["p"] - r2 - 2 * y * y, 0.0], [0.0, 0.0, -c]])

    def dfdp(u, par, name):
        return np.array([u[0], u[1], 0.0])
    ds = DynamicalSystem.from_callables(G, 3, {"p": p}, jac=J, dfdp=dfdp, fd_x=1e-4,
                                        name="hopf normal form + z")
    mesh = O.CollocationMesh.uniform(ntst, 4)
    pb = O.PeriodicOrbitProblem(ds, "p", mesh)
    t = mesh.times()
    U = np.stack([np.sqrt(p) * np.cos(2 * np.pi * t), np.sqrt(p) * np.sin(2 * np.pi * t), 0 * t], 1)
    pb.set_reference(U)
    z = np.concatenate([U.ravel(), [2 * np.pi, p]])
    return O.floquet_from_collocation(pb, z), p


out, p = run(20, 1.0)
mu = np.sort(np.abs(out["multipliers"]))[::-1]
want = np.sort([1.0, np.exp(-4 * np.pi * p), np.exp(-2 * np.pi)])[::-1]
check(np.allclose(mu, want, rtol=1e-6, atol=0), f"multipliers {mu}, oracle {want}")
check(out["liouville_defect"] < 1e-8, f"resolved mesh: Liouville defect {out['liouville_defect']:.1e}")
coarse, _ = run(4, 60.0)
check(coarse["liouville_defect"] > 100 * out["liouville_defect"],
      f"coarse mesh, fast transverse rate: defect {coarse['liouville_defect']:.1e} (resolved {out['liouville_defect']:.1e})")
passed(f"defect {out['liouville_defect']:.1e} resolved, {coarse['liouville_defect']:.1e} under-resolved "
       f"(trivial error {coarse['trivial_error']:.1e})")
