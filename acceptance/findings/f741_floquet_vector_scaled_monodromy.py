"""F741 (audit of 0.2.1, N22): the target multiplier is compared with multipliers of the true map.

Oracle: u, v Stuart-Landau with limit cycle (cos t, sin t), mu = 1, T = 2 pi, plus w_t = 50 w: multipliers 1 (along the orbit), exp(-4 pi), exp(100 pi) (beyond the scaling threshold of the monodromy).
"""
from _acceptance import check, passed, start

start(741, "Floquet vector of a scaled monodromy")
import numpy as np  # noqa: E402


from conatus.bifurcation import ir_system, ode_problem  # noqa: E402
from conatus.bifurcation.orbits import CollocationMesh, PeriodicOrbitProblem  # noqa: E402

pb = ode_problem(["u", "v", "w"], {"mu": 1.}, lambda X, P: [
    P["mu"] * X["u"] - X["v"] - (X["u"] ** 2 + X["v"] ** 2) * X["u"],
    X["u"] + P["mu"] * X["v"] - (X["u"] ** 2 + X["v"] ** 2) * X["v"],
    50 * X["w"]])
s = ir_system(pb, ["mu"])
mesh = CollocationMesh.uniform(120, 4)
po = PeriodicOrbitProblem(s, "mu", mesh)
nt = mesh.npts
tau = np.empty(nt)
for j in range(mesh.ntst):
    for k in range(mesh.ncol):
        tau[j * mesh.ncol + k] = mesh.tmesh[j] + (mesh.tmesh[j + 1] - mesh.tmesh[j]) * k / mesh.ncol
tau[-1] = 1.0
th = 2 * np.pi * tau
U = np.stack([np.cos(th), np.sin(th), 0 * th], axis=1)
z = np.concatenate([U.ravel(), [2 * np.pi, 1.0]])
P, ls = po.monodromy(z)
check(ls > 50, f"the monodromy is not scaled here (log scale {ls})")
F = po.floquet_vector(z, 1.0)
v0 = F[0]
check(abs(v0[2]) < 1e-6 * np.linalg.norm(v0), f"vector for multiplier 1 is {v0}")
check(abs(v0[0]) < 1e-3 * np.linalg.norm(v0), f"not the tangent (0, 1, 0) at t = 0: {v0}")
F = po.floquet_vector(z, np.exp(-4 * np.pi))         # radial multiplier: vector (1, 0, 0) at t = 0
v0 = F[0]
check(abs(v0[1]) < 1e-3 * np.linalg.norm(v0) and abs(v0[2]) < 1e-6 * np.linalg.norm(v0),
      f"vector for the radial multiplier is {v0}")
passed("the tangent and the radial vector of the orbit")
