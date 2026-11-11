"""F911 (hunt 5A, K2 and K5): a Neimark-Sacker point of cycles with two slow decoupled modes is
classified by the factor that crossed; a planar fold of cycles under the default tests does not
crash.

Oracles, closed form. NS: base cycle x' = (1 - r^2) x - y, y' = x + (1 - r^2) y, and (u, v) with
rate p and rotation 0.3: NS at p = 0. Two spectators w_i' = -e w_i (multipliers exp(-2 pi e)),
e = 1e-12 (0.3.0: "neutral saddle cycle", no ns event) and 1e-9 (control). Fold: x' = f x - y,
y' = x + f y, f = p + 2 r^2 - r^4: cycles r^2 = 1 +- sqrt(1 + p), fold at p = -1 (0.3.0:
ValueError with the default tests).
"""
import numpy as np
from _acceptance import check, passed, start

start(911, "cycle events classified by the crossing factor; planar fold of cycles")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import CollocationMesh, PeriodicOrbitProblem, continue_orbits  # noqa: E402

CFG = C.ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.1, tolerance=1e-10, max_points=20)


def circle(pb, rad, n):
    tau = pb.mesh.times()
    U = np.zeros((tau.size, n))
    U[:, 0], U[:, 1] = rad * np.cos(2 * np.pi * tau), rad * np.sin(2 * np.pi * tau)
    return U


for e in (1e-12, 1e-9):
    names = ["x", "y", "u", "v", "w0", "w1"]

    def rhs(X, P, e=e):
        x, y = X["x"], X["y"]
        f = 1 - x * x - y * y
        return [f * x - y, x + f * y, P["p"] * X["u"] - 0.3 * X["v"], 0.3 * X["u"] + P["p"] * X["v"],
                -e * X["w0"], -e * X["w1"]]
    pb = PeriodicOrbitProblem(C.ir_system(C.ode_problem(names, {"p": -0.3}, rhs), ["p"]), "p",
                              CollocationMesh.uniform(24, 4))
    U = circle(pb, 1.0, len(names))
    cv = continue_orbits(pb, np.concatenate([U.ravel(), [2 * np.pi, -0.3]]), CFG, bounds={"p": (-0.3, 0.3)})
    evs = cv.events_of("ns")
    where = f"e = {e:g}: {[(v.kind, float(v.point.z[-1]), v.converged) for v in cv.events]}"
    check(len(evs) == 1 and evs[0].converged and abs(float(evs[0].point.z[-1])) <= 1e-7, where)


def frhs(X, P):
    r2 = X["x"] ** 2 + X["y"] ** 2
    f = P["p"] + 2 * r2 - r2 * r2
    return [f * X["x"] - X["y"], X["x"] + f * X["y"]]


pb = PeriodicOrbitProblem(C.ir_system(C.ode_problem(["x", "y"], {"p": -0.5}, frhs), ["p"]), "p",
                          CollocationMesh.uniform(20, 4))
U = circle(pb, np.sqrt(1 + np.sqrt(0.5)), 2)
z0 = np.concatenate([U.ravel(), [2 * np.pi, -0.5]])
cfg = C.ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.2, tolerance=1e-10, max_points=40)
cv = continue_orbits(pb, z0, cfg, bounds={"p": (-2.0, 0.5)},
                     direction=np.concatenate([np.zeros(U.size), [0.0, -1.0]]))
lpc = [float(v.point.z[-1]) for v in cv.events_of("lpc") if v.converged]
check(any(abs(p + 1) <= 1e-6 for p in lpc), f"fold: {[(v.kind, float(v.point.z[-1])) for v in cv.events]}")
passed("NS at p = 0 with two spectators at 1e-12 and 1e-9; planar fold of cycles at p = -1 with "
       "the default tests")
