"""F922 (pass 4 H4, GPT G3): stability paths keep a non-conserved mean.

Oracle: u_t = a u + u_xx (+ conserved or not terms) on a 2 pi-periodic line, linearized at u = 0:
eigenvalues a - k^2 for every k >= 0 (k = 0 is the mean), multipliers exp(T (a - k^2)).
Paths: equilibrium_eigenvalues, floquet_multipliers, expert.ecs_system, symmetry_breaking_check.
Rate scales a = 0.5 and 1e3. Control: Kuramoto-Sivashinsky-like u_t = -u_xx - u_xxxx - u u_x
conserves the mean, which stays dropped (the former behaviour, recorded in the space note).
"""
import numpy as np
from _acceptance import check, passed, start

start(922, "mean kept when not conserved in four ECS stability paths")
import conatus as cn  # noqa: E402
import conatus.expert as X  # noqa: E402
from conatus.ecs import (equilibrium_eigenvalues, floquet_multipliers,  # noqa: E402
                         symmetry_breaking_check)
from conatus.ecs.solutions import Equilibrium, RelativePeriodicOrbit  # noqa: E402


def sim(eq, a, u0=lambda x: 0 * x):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), a=a)
    m.scalar("u")
    m.equations(eq)
    s = m.simulate(16, scheme="RK443", dt=1e-3 if a > 10 else 1e-2)
    s.initial(u=u0)
    return s


for a in (0.5, 1e3):
    s = sim("dt(u)-a*u-dx(dx(u))=-u*u*u", a)
    sol = Equilibrium(kind="equilibrium", X=np.zeros_like(s.solver.X), params={}, residual=0.0,
                      newton=None, energy=0.0)
    ev = equilibrium_eigenvalues(s, sol, 3, arnoldi_tol=1e-12)
    check(abs(ev[0] - a) <= 1e-9 * max(1, a), f"a = {a}: equilibrium top {ev[0]}, oracle {a}")
    check(abs(ev[1] - (a - 1)) <= 1e-9 * max(1, a), f"a = {a}: second {ev[1]}, oracle {a - 1}")

a, T = 0.5, 1.0
s = sim("dt(u)-a*u-dx(dx(u))=-u*u*u", a)
orb = RelativePeriodicOrbit(kind="relative_periodic_orbit", X=np.zeros_like(s.solver.X),
                            params={"T": T, "ell": 0.0, "nsteps": 100}, residual=0.0, newton=None,
                            energy=0.0)
f = floquet_multipliers(s, orb, 3, arnoldi_tol=1e-12)
mu = np.sort(np.abs(f["multipliers"]))[::-1]
check(abs(mu[0] - np.exp(a * T)) <= 1e-6, f"multiplier top {mu[0]}, oracle {np.exp(a * T)}")
check("kept" in f["space"], f"space note {f['space']}")

s = sim("dt(u)-a*u-dx(dx(u))=-u*dx(u)", 0.5)
ds = X.ecs_system(s.solver, odd=False, degree=2, dense_max=64)
ev = np.sort(np.linalg.eigvals(ds.J(np.zeros(ds.n), {})).real)[::-1]
check(abs(ev[0] - 0.5) <= 1e-9, f"ecs_system top {ev[0]}, oracle 0.5")
check("(conserved)" not in ds.explain(), "ecs_system still asserts conservation")

# G3: forced so that sin x is the equilibrium; the mean (antisymmetric under R) has rate a
for a in (-1.0, 1.0):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), a=a)
    m.scalar("u")
    m.given(f=(lambda c: (lambda x: c * np.sin(x)))(1 - a))
    m.equations("dt(u)-a*u-dx(dx(u))=f")
    s = m.simulate(16, scheme="RK443", dt=0.01)
    s.initial(u=lambda x: np.sin(x))
    sol = Equilibrium(kind="equilibrium", X=np.array(s.solver.X), params={}, residual=0.0,
                      newton=None, energy=0.0)
    out = symmetry_breaking_check(s, sol, unstable_tol=1e-8, coupling_tol=1e-10)
    lead = complex(out["leading"]["antisymmetric"])
    check(abs(lead - a) <= 1e-9, f"a = {a}: antisymmetric leading {lead}, oracle {a}")
    check(bool(out["unstable_antisymmetric"]) == (a > 0) and out["verdict"].startswith("no") == (a < 0),
          f"a = {a}: verdict {out['verdict']}")

# control: conserved mean (KS-like) stays dropped, no extra eigenvalue
s = sim("dt(u)+dx(dx(u))+dx(dx(dx(dx(u))))=-u*dx(u)", 0.0)
sol = Equilibrium(kind="equilibrium", X=np.zeros_like(s.solver.X), params={}, residual=0.0,
                  newton=None, energy=0.0)
ev = equilibrium_eigenvalues(s, sol, 40, arnoldi_tol=1e-12)
want = np.sort(np.concatenate([[-(k**4) + k**2] * 2 for k in range(1, 8)] + [[-(8**4) + 8**2]]))[::-1]
check(np.allclose(np.sort(ev.real)[::-1][:6], want[:6], rtol=0, atol=1e-8), f"KS control {ev.real[:6]}")
check("conserved" in sol.stability.get("equilibrium_eigenvalues: space", ""), str(sol.stability))
passed("mean rate a found by equilibrium_eigenvalues, floquet_multipliers, ecs_system, "
       "symmetry_breaking_check; conserved mean still dropped")
