"""F901 (pass 4 H1): two crossings in one continuation step are both reported, and a double
eigenvalue is reported as an unresolved crossing, never dropped.

Oracles, closed form. x' = p x, y' = (p - d) y: branch points at p = 0 and p = d. Two Hopf pairs
p +- i and (p - d) +- 3 i: Hopf points at p = 0 and p = d. With d = 0.01 and 1e-4 both fall in one
step (ds = 0.05). With d = 0 the crossing is double: one event "unresolved crossing" at p = 0
(converged False). Control: d = 1 (two steps apart).
"""
import numpy as np
from _acceptance import check, passed, start

start(901, "two crossings in one step: both located; a double crossing is reported unresolved")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import continue_equilibria  # noqa: E402

CFG = C.ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.1, tolerance=1e-10, max_points=40)


def real(d):
    return C.ode_problem(["x", "y"], {"p": -0.5},
                         lambda X, P: [P["p"] * X["x"], (P["p"] - d) * X["y"]])


def hopf(d):
    def rhs(X, P):
        p, q = P["p"], P["p"] - d
        return [p * X["a"] - X["b"], X["a"] + p * X["b"], q * X["c"] - 3 * X["e"], 3 * X["c"] + q * X["e"]]
    return C.ode_problem(["a", "b", "c", "e"], {"p": -0.5}, rhs)


for make, kind in ((real, "branch point"), (hopf, "hopf")):
    for d in (1.0, 0.01, 1e-4, 0.0):
        pb = make(d)
        cv = continue_equilibria(C.ir_system(pb, ["p"]), np.zeros(len(pb.variables)), "p", CFG,
                                 marginal_tol=1e-9, bounds=(-0.5, 1.5))
        got = sorted(float(e.point.z[-1]) for e in cv.events_of(kind))
        where = f"{kind}, d = {d:g}: {[(e.kind, float(e.point.z[-1]), e.converged) for e in cv.events]}"
        if d > 0:
            check(len(got) == 2 and abs(got[0]) <= 1e-8 and abs(got[1] - d) <= 1e-8, where)
            check(all(e.converged for e in cv.events_of(kind)), where)
        else:
            un = cv.events_of("unresolved crossing")
            check(not got and len(un) == 1 and not un[0].converged
                  and abs(float(un[0].point.z[-1])) <= 1e-8, where)
            check("crossing count" in un[0].reason, un[0].reason)
passed("branch points and Hopf points 1e-4 apart located in one step; a double crossing reported "
       "unresolved at p = 0")
