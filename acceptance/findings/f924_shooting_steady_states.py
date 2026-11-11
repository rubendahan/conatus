"""F924 (hunt 5A K1): ShootingRPO does not call a travelling wave or an equilibrium a relative
periodic orbit.

Oracle: KS L = 22. TW1 (c = 0.737) and E2 are steady in a frame: they solve tau(-a) Phi_T X = X
for every T (a = -c T, or a = 0), so they are not periodic orbits, whatever the residual. The 1-D
finder retracts them. Control: the PPO 10.25 fixture run twice (Phi_{2T} u = u, shift 0) is a
genuine orbit of the shift-only shooting problem and is left converged.
"""
import json
from pathlib import Path

import numpy as np
from _acceptance import check, passed, start

start(924, "ShootingRPO retracts travelling waves and equilibria")
from conatus.ecs import NewtonKrylovHookstep, find_equilibrium, find_travelling_wave, ks_solver  # noqa: E402
from conatus.ecs.shooting import ShootingRPO  # noqa: E402
from conatus.ecs.subspace import DiagonalSymmetry, DiagonalSymmetryGroup  # noqa: E402


def nw(it):
    return NewtonKrylovHookstep(tol=1e-11, max_iterations=it, gmres_rtol=1e-6, gmres_max_iterations=80,
                                delta0=0.5, delta_min=1e-10, delta_max=5.0)


s = ks_solver(L=22.0, N=32, dealias=1.5, scheme="RK443", timestep=0.05)
rng = np.random.default_rng(0)
for _ in range(8):
    c = np.zeros(32)
    c[2:10] = rng.normal(size=8) * 0.8
tw = find_travelling_wave(s, s.system.gather_state([c]), 0.737, nw(60))
check(abs(abs(tw.params["c"]) - 0.737) < 1e-2, f"TW1 not found: c = {tw.params['c']}")
c = np.zeros(32)
c[5] = -1.0
e2 = find_equilibrium(s, s.system.gather_state([c]), nw(40))
G = DiagonalSymmetryGroup(s, [DiagonalSymmetry("identity", {}, {"u": 1})])
T = 2.0
for label, X0, a0, kind in (("TW1", tw.X, -tw.params["c"] * T, "travelling_wave"), ("E2", e2.X, 0.0, "equilibrium")):
    sh = ShootingRPO(s, G, shifts=["x"], nsteps=40, dT_rel=1e-5, cache_stages=False)
    sol = sh.solve(X0, T, {"x": a0}, nw(10))
    check(sol.status == "retracted" and sol.kind == kind,
          f"{label}: kind {sol.kind}, status {sol.status}, residual {sol.residual:.1e}")
    check("Relative periodic orbit" not in sol.explain().splitlines()[0], f"{label}: {sol.explain()[:120]}")

d = json.loads((Path(__file__).resolve().parents[3] / "tests/ecs/data/ks_l22_ppo10.json").read_text()
               if (Path(__file__).resolve().parents[3] / "tests/ecs/data/ks_l22_ppo10.json").exists()
               else "null") or None
if d is not None:
    Tp, n = d["T"], d["nsteps"]
    sp = ks_solver(L=22.0, N=32, dealias=1.5, scheme="RK443", timestep=Tp / n)
    Xp = sp.system.gather_state([np.array(d["coeffs"])])
    Gp = DiagonalSymmetryGroup(sp, [DiagonalSymmetry("identity", {}, {"u": 1})])
    sh = ShootingRPO(sp, Gp, shifts=["x"], nsteps=2 * n, dT_rel=1e-5, cache_stages=False)
    sol = sh.solve(Xp, 2 * Tp, {"x": 0.0}, nw(10))
    check(sol.status == "converged" and sol.kind == "relative_periodic_orbit" and sol.repetitions == 1,
          f"control orbit: kind {sol.kind}, status {sol.status}, repetitions {sol.repetitions}")
passed("TW1 and E2 retracted with their kind; a genuine orbit left converged")
