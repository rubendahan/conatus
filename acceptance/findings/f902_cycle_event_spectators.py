"""F902 (pass 4 H1, H2): PD, NS, BPC and LPC of cycles are placed at their closed-form parameter
with decoupled stable, slow and exactly neutral spectators, and two crossings in one step are
both reported.

Oracles, closed form. Base cycle x' = (1 - r^2) x - y, y' = x + (1 - r^2) y (unit circle, T = 2 pi).
PD: (u, v) rotating at half speed with rate p - 1 + cos/sin coupling, multiplier -exp(2 pi p),
PD at p = 0. NS: (u, v) with rate p and rotation 0.3, NS at p = 0. BPC: u' = p u, BPC at p = 0.
LPC: x' = f x - y, y' = x + f y with f = p + 2 r^2 - r^4, cycles r^2 = 1 +- sqrt(1 + p), fold of
cycles at p = -1. Spectators w' = c w with c = -1, -1e-12 and 0 (multiplier 1). Two crossings:
copies of the PD, NS or BPC block shifted by d = 0.01.
"""
import numpy as np
from _acceptance import check, passed, start

start(902, "cycle events at their closed-form parameter under spectators; two crossings in one step")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import CollocationMesh, PeriodicOrbitProblem, continue_orbits  # noqa: E402

CFG = C.ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.1, tolerance=1e-10, max_points=20)


def pd(X, P, u, v, d):
    x, y, q = X["x"], X["y"], P["p"] - d - 1
    return [q * X[u] - X[v] / 2 + x * X[u] + y * X[v], q * X[v] + X[u] / 2 + y * X[u] - x * X[v]]


def ns(X, P, u, v, d):
    return [(P["p"] - d) * X[u] - 0.3 * X[v], 0.3 * X[u] + (P["p"] - d) * X[v]]


def bpc(X, P, u, v, d):
    return [(P["p"] - d) * X[u], -X[v]]


def run(block, shifts, c, lpc=False):
    names = ["x", "y"] + [f"{s}{k}" for k in range(len(shifts)) for s in ("u", "v")] + ["w"]

    def rhs(X, P):
        x, y = X["x"], X["y"]
        r2 = x * x + y * y
        f = P["p"] + 2 * r2 - r2 * r2 if lpc else 1 - r2
        out = [f * x - y, x + f * y]
        for k, d in enumerate(shifts):
            out += block(X, P, f"u{k}", f"v{k}", d)
        return out + [c * X["w"]]
    p0 = -0.5 if lpc else -0.3
    pb = PeriodicOrbitProblem(C.ir_system(C.ode_problem(names, {"p": p0}, rhs), ["p"]), "p",
                              CollocationMesh.uniform(24, 4))
    tau = pb.mesh.times()
    rad = np.sqrt(1 + np.sqrt(1 + p0)) if lpc else 1.0
    U = np.zeros((tau.size, len(names)))
    U[:, 0], U[:, 1] = rad * np.cos(2 * np.pi * tau), rad * np.sin(2 * np.pi * tau)
    return continue_orbits(pb, np.concatenate([U.ravel(), [2 * np.pi, p0]]), CFG,
                           direction=None if not lpc else np.concatenate(
                               [np.zeros(U.size), [0.0, -1.0]]),
                           bounds={"p": (-1.3, 0.3) if lpc else (p0, 0.3)})


cases = 0
for kind, block in (("pd", pd), ("ns", ns), ("bpc", bpc)):
    for c in (-1.0, -1e-12, 0.0):
        cv = run(block, [0.0], c)
        evs = cv.events_of(kind)
        where = f"{kind}, spectator {c:g}: {[(e.kind, float(e.point.z[-1]), e.converged) for e in cv.events]}"
        check(len(evs) == 1 and evs[0].converged and abs(float(evs[0].point.z[-1])) <= 1e-7, where)
        check(evs[0].data.get("bracket", 1.0) <= CFG.crossing_tol, f"{where}: {evs[0].data}")
        cases += 1
    cv = run(block, [0.0, 0.01], -1.0)
    got = sorted(float(e.point.z[-1]) for e in cv.events_of(kind))
    where = f"{kind} twice: {[(e.kind, float(e.point.z[-1]), e.converged) for e in cv.events]}"
    check(len(got) == 2 and abs(got[0]) <= 1e-7 and abs(got[1] - 0.01) <= 1e-7, where)
    cases += 1
for c in (-1.0, 0.0):
    cv = run(None, [], c, lpc=True)
    evs = cv.events_of("lpc")
    where = f"lpc, spectator {c:g}: {[(e.kind, float(e.point.z[-1]), e.converged) for e in cv.events]}"
    check(len(evs) == 1 and evs[0].converged and abs(float(evs[0].point.z[-1]) + 1) <= 1e-6, where)
    cases += 1
passed(f"{cases} cycle runs: PD, NS, BPC at p = 0 and LPC at p = -1 with spectators -1, -1e-12, 0; "
       "pairs 0.01 apart both located")
