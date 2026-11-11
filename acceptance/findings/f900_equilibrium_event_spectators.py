"""F900 (pass 4 H2, G14): fold, branch point and Hopf are placed at p = 0 whatever the spectators
and the time unit, and each located event verifies its own critical condition.

Oracles, closed form. Fold: x' = e (p - x^2), branch x^2 = p, fold at p = 0. Branch point:
x' = e (p x - x^2), trivial branch x = 0 crosses x = p at p = 0. Hopf: a' = e (p a - b - a r^2),
b' = e (a + p b - b r^2), eigenvalues e (p +- i) at the origin. A decoupled spectator y' = e c y
(c = -1, -1e-6, -1e-12, -1e-14, +1e-14, 0) never moves the event. Time units e = 1e-12, 1, 1e12.
Control: the same runs with c = -1 (the case that already passed in 0.2.4).
"""
import numpy as np
from _acceptance import check, passed, start

start(900, "equilibrium events at p = 0 under spectators and time units, critical condition verified")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import continue_equilibria  # noqa: E402

CFG = C.ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.1, tolerance=1e-10, max_points=40)


def system(kind, e, c):
    def rhs(X, P):
        p, y = P["p"], X["y"]
        if kind == "fold":
            return [e * (p - X["x"] ** 2), e * c * y]
        if kind == "branch point":
            return [e * (p * X["x"] - X["x"] ** 2), e * c * y]
        a, b = X["x"], X["w"]
        r2 = a * a + b * b
        return [e * (p * a - b - a * r2), e * (a + p * b - b * r2), e * c * y]
    names = ["x", "w", "y"] if kind == "hopf" else ["x", "y"]
    return C.ir_system(C.ode_problem(names, {"p": 0.25 if kind == "fold" else -0.3}, rhs), ["p"])


cases = 0
for kind in ("fold", "branch point", "hopf"):
    for e in (1e-12, 1.0, 1e12):
        for c in (-1.0, -1e-6, -1e-12, -1e-14, 1e-14, 0.0):
            ds = system(kind, e, c)
            n = ds.n
            x0 = np.zeros(n)
            if kind == "fold":
                x0[0] = 0.5                      # x = sqrt(p) at p = 0.25, towards the fold
            cv = continue_equilibria(ds, x0, "p", CFG, marginal_tol=abs(e) * 1e-9,
                                     direction=-1.0 if kind == "fold" else 1.0,
                                     bounds=(-0.3, 0.3))
            evs = cv.events_of(kind)
            where = f"{kind}, e = {e:g}, spectator {c:g}"
            check(len(evs) == 1, f"{where}: events {[(v.kind, v.converged) for v in cv.events]}")
            ev = evs[0]
            p = float(ev.point.z[-1])
            check(ev.converged and abs(p) <= 1e-8, f"{where}: p = {p:.3e}, converged {ev.converged}")
            if kind == "fold":
                check(abs(ev.point.tangent[-1]) <= CFG.fold_tol, f"{where}: |t_p| above fold_tol")
            else:
                # the critical condition, as explain() prints it: a factor changes sign across a
                # bracket narrower than crossing_tol x step
                check(ev.data.get("bracket", 1.0) <= CFG.crossing_tol, f"{where}: {ev.data}")
                # (one end may hold the factor at rounding level: zero for the sign)
                f0, f1 = ev.data.get("crossing_factor", (1.0, 1.0))
                check(f0 * f1 <= 0 or min(abs(f0), abs(f1)) < 1e-3 * max(abs(f0), abs(f1)),
                      f"{where}: crossing factor {f0}, {f1}")
                check("bracket" in ev.explain(), f"{where}: explain() {ev.explain()}")
            cases += 1
passed(f"{cases} runs: fold, branch point, Hopf at p = 0 within 1e-8 for spectators -1 to 0 and "
       "time units 1e-12 to 1e12")
