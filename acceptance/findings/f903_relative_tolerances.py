"""F903 (pass 4 H5, review G6): the corrector tolerances are relative to the size of the
equations, and the Newton-Krylov residual norm does not underflow.

Oracles. x' = e (p - x^2): the branch is x^2 = p for every e > 0, the fold at p = 0. With
tolerance 1e-10 the points satisfy |x^2 - p| <= 1e-8 and the fold lies within 1e-8 of 0 for
e = 1e-10 to 1e6 (0.2.4: error 0.33 at e = 1e-10, fold at -0.125). Newton-Krylov on
R(z) = u (z - 2), u = 1, 1e-200, 1e200: the root is z = 2 (0.2.4 returned the seed z = 1 at
u = 1e-200 with residual 0). Steady state of e (u_xx + 2 u - 2 u^3) = 0: its energy does not
depend on e (0.2.4 returned the seed at e = 1e-10). Control: e = 1, u = 1.
"""
import numpy as np
from _acceptance import check, passed, start

start(903, "relative corrector tolerance; Newton-Krylov norm without underflow")
import conatus as cn  # noqa: E402
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import continue_equilibria  # noqa: E402
from conatus.ecs import NewtonKrylovHookstep, find_equilibrium  # noqa: E402

cfg = C.ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.1, tolerance=1e-10, max_points=40)
for e in (1.0, 1e-4, 1e-8, 1e-10, 1e6):
    ds = C.ir_system(C.ode_problem(["x"], {"p": 1.0}, lambda X, P, e=e: [e * (P["p"] - X["x"] ** 2)]),
                     ["p"])
    cv = continue_equilibria(ds, np.ones(1), "p", cfg, marginal_tol=e * 1e-9, direction=-1.0,
                             bounds=(-1.0, 1.5))
    err = max(abs(pt.z[0] ** 2 - pt.z[1]) for pt in cv.points)
    folds = cv.events_of("fold")
    check(err <= 1e-8, f"e = {e:g}: max|x^2 - p| = {err:.2e}")
    check(len(folds) == 1 and abs(float(folds[0].point.z[-1])) <= 1e-8,
          f"e = {e:g}: folds {[float(f.point.z[-1]) for f in folds]}")
for u in (1.0, 1e-200, 1e200):
    out = NewtonKrylovHookstep(tol=1e-12).solve(lambda z, u=u: u * (z - 2), lambda z, v, u=u: u * v,
                                                np.array([1.0]), scale=u)
    check(abs(out.z[0] - 2) <= 1e-10, f"unit {u:g}: z = {out.z}, reported {out.relative_residual}")
energies = []
for eps in (1.0, 1e-6, 1e-10):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), e=eps)
    m.scalar("u")
    m.equations("dt(u) - e*dx(dx(u)) - e*2*u = -e*2*u*u*u")
    s = m.simulate(32, scheme="RK443", dt=0.01)
    s.initial(u=lambda x: np.sin(x))
    sol = find_equilibrium(s, s, NewtonKrylovHookstep(tol=1e-10))
    check(sol.newton.iterations > 0, f"e = {eps:g}: seed returned after 0 iterations")
    energies.append(sol.energy)
check(max(energies) - min(energies) <= 1e-8 * max(energies), f"energies {energies}")
passed("fold within 1e-8 of 0 for e = 1e-10 to 1e6; Newton-Krylov exact at units 1e-200 to 1e200; "
       "steady energy independent of e")
