"""F908 (pass 4 H2, H1 on codimension-2 curves): Bogdanov-Takens and zero-Hopf points on a fold
curve are placed at their closed-form parameter with decoupled spectators and time units.

Oracle: x' = e y, y' = e (p1 + p2 y + x^2), a' = e ((p2 - d) a - b), b' = e (a + (p2 - d) b),
w' = e c w. The fold curve is x = y = 0, p1 = 0 for every p2; its eigenvalues are 0, e p2,
e (p2 - d +- i) and e c: Bogdanov-Takens at p2 = 0, zero-Hopf at p2 = d. d = 0.2, and d = 0.01
(both in one step). Spectators c = -1, -1e-6, -1e-12; time units e = 1e-6, 1, 1e6.
"""
import numpy as np
from _acceptance import check, passed, start

start(908, "codim-2 events on a fold curve at closed-form parameters under spectators and units")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import continue_codim2  # noqa: E402

cfg = C.ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.1, tolerance=1e-10, max_points=30)
cases = 0
for d in (0.2, 0.01):
    for e in (1e-6, 1.0, 1e6):
        for c in (-1.0, -1e-6, -1e-12):
            def rhs(X, P, e=e, c=c, d=d):
                q = P["p2"] - d
                return [e * X["y"], e * (P["p1"] + P["p2"] * X["y"] + X["x"] ** 2),
                        e * (q * X["a"] - X["b"]), e * (X["a"] + q * X["b"]), e * c * X["w"]]
            ds = C.ir_system(C.ode_problem(["x", "y", "a", "b", "w"], {"p1": 0.0, "p2": -0.3}, rhs),
                             ["p1", "p2"])
            cv = continue_codim2(ds, np.zeros(5), "fold", ("p1", "p2"), cfg,
                                 par={"p1": 0.0, "p2": -0.3}, bounds={"p2": (-0.3, 0.4)})
            got = {k: [float(v.point.z[-1]) for v in cv.events_of(k) if v.converged]
                   for k in ("bt", "zh")}
            where = (f"d {d:g}, e {e:g}, c {c:g}: "
                     f"{[(v.kind, float(v.point.z[-1]), v.converged) for v in cv.events]}")
            check(len(got["bt"]) == 1 and abs(got["bt"][0]) <= 1e-8, where)
            check(len(got["zh"]) == 1 and abs(got["zh"][0] - d) <= 1e-8, where)
            cases += 1
passed(f"{cases} fold curves: Bogdanov-Takens at p2 = 0 and zero-Hopf at p2 = d")
