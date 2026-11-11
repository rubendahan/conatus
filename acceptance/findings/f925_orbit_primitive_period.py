"""F925 (hunt 5A K4): the orbit finders say when a solution is several laps of a shorter orbit.

Oracle: the PPO 10.25 of KS L = 22 solves R tau_l Phi_T u = u (T = 10.2529). So
R tau_l Phi_{3T} u = u (three laps, reflect=True) and Phi_{2T} u = u (two laps, no reflection) hold
exactly, and the primitive period is T in both cases. Control: the seed at T is not flagged.
Also ShootingRPO (shift only) at 4T: two laps of the 2T orbit. And collocation: a limit cycle
of the Hopf normal form traversed twice.
"""
import json
from pathlib import Path

import numpy as np
from _acceptance import check, passed, skipped, start

start(925, "orbit finders report the primitive period of a repeated orbit")
from conatus.ecs import NewtonKrylovHookstep, find_relative_periodic_orbit, ks_solver  # noqa: E402
from conatus.ecs.shooting import ShootingRPO  # noqa: E402
from conatus.ecs.subspace import DiagonalSymmetry, DiagonalSymmetryGroup  # noqa: E402
import conatus.bifurcation.orbits as O  # noqa: E402
from conatus.bifurcation import DynamicalSystem  # noqa: E402

f = Path(__file__).resolve().parents[3] / "tests/ecs/data/ks_l22_ppo10.json"
if not f.exists():
    skipped("PPO fixture not in this tree")
d = json.loads(f.read_text())
T, ell, n = d["T"], d["ell"], d["nsteps"]


def nw(it=10):
    return NewtonKrylovHookstep(tol=1e-10, max_iterations=it, gmres_rtol=1e-6, gmres_max_iterations=80,
                                delta0=0.5, delta_min=1e-10, delta_max=5.0)


s = ks_solver(L=22.0, N=32, dealias=1.5, scheme="RK443", timestep=T / n)
X = s.system.gather_state([np.array(d["coeffs"])])
base = find_relative_periodic_orbit(s, X, T, ell, nw(), steps=n, reflect=True)
check(base.status != "repeated" and "repetitions" not in base.params, f"control flagged: {base.params}")
for T0, l0, steps, refl, k, r in ((3 * T, ell, 3 * n, True, 3, True), (2 * T, 0.0, 2 * n, False, 2, True)):
    sol = find_relative_periodic_orbit(s, X, T0, l0, nw(), steps=steps, reflect=refl)
    check(sol.status == "repeated" and sol.params.get("repetitions") == k
          and abs(sol.params["primitive_period"] - T) < 1e-8 and sol.params["primitive_reflect"] == r,
          f"{k} laps: status {sol.status}, params {sol.params}")
    check(any("REPEATED ORBIT" in x for x in sol.notes), f"{k} laps: no note")

G = DiagonalSymmetryGroup(s, [DiagonalSymmetry("identity", {}, {"u": 1})])
sh = ShootingRPO(s, G, shifts=["x"], nsteps=4 * n, dT_rel=1e-5, cache_stages=False)
sol = sh.solve(X, 4 * T, {"x": 0.0}, nw())
check(sol.status == "repeated" and sol.repetitions == 2, f"ShootingRPO 4T: {sol.status}, {sol.repetitions}")


def G_(u, par):
    x, y = u
    r2 = x * x + y * y
    return np.array([x * (par["p"] - r2) - y, y * (par["p"] - r2) + x])


def J_(u, par):
    x, y = u
    r2 = x * x + y * y
    return np.array([[par["p"] - r2 - 2 * x * x, -2 * x * y - 1], [-2 * x * y + 1, par["p"] - r2 - 2 * y * y]])


ds = DynamicalSystem.from_callables(G_, 2, {"p": 0.25}, jac=J_, dfdp=lambda u, p, nm: np.array(u),
                                    fd_x=1e-4, name="hopf")
for laps in (1, 2):
    pb = O.PeriodicOrbitProblem(ds, "p", O.CollocationMesh.uniform(40, 4))
    t = pb.mesh.times()
    U = 0.5 * np.stack([np.cos(2 * np.pi * laps * t), np.sin(2 * np.pi * laps * t)], 1)
    pb.set_reference(U)
    out = O.floquet_from_collocation(pb, np.concatenate([U.ravel(), [2 * np.pi * laps, 0.25]]))
    check(out["repetitions"] == laps, f"collocation, {laps} lap(s): repetitions {out['repetitions']}")
passed("3T and 2T seeds report the primitive period T; ShootingRPO at 4T and a doubled limit cycle flagged")
