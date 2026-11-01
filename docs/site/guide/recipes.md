# Recipes

Each recipe is the shortest code for one task, with the name of the value to read. A test runs every
block on this page and checks that value against an independent oracle: a closed form, or a value
printed in a paper ({doc}`index`). To adapt a recipe, change the model and keep the checks that matter:
the residual, and a second resolution.

## Leading eigenvalue of a linear stability problem

Plane Poiseuille flow, the Orr–Sommerfeld equation at $Re = 10^4$, $\alpha = 1$. Read `c`, and check
`residual`.

```python
import conatus as cn

m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
m.scalar("v")
m.given(U=lambda y: 1 - y**2)
m.eigenvalue("c", growth="imag")
m.define("L2(f) = dy(dy(f)) - alpha**2*f")
m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")
res = m.eigenvalues(N=64, drift=96, drift_tol=1e-6)     # modes that move with N are removed
i = int(res.order()[0])                                 # least stable first
c, residual = res.values[i], res.residuals[i]
```

## Eigenmodes and their shape

The Laplacian on $[0, 1]$ with $u = 0$ at the walls. Read `values`, and `shape` on the points `x`.

```python
import conatus as cn

m = cn.Model(cn.Box(x=cn.walls(0, 1)), name="Laplacian")
m.scalar("u")
m.eigenvalue("s", growth="real")
m.equations("dx(dx(u)) - s*u = 0")
m.at("walls", u=0)
res = m.eigenvalues(N=32, drift=48, drift_tol=1e-6)
order = res.order()
values = res.values[order[:3]]                          # least stable first
md = res.mode(int(order[0]))
x, shape = md.coordinate("x"), md["u"]
```

## Critical parameter: the onset of an instability

Rayleigh–Bénard convection between no-slip walls, minimised over the wavenumber. Read `crit.value` and
`crit.over["k"]`.

```python
import conatus as cn
import conatus.expert as X

lin = cn.problems.native.rayleigh_benard_evp(k=3.1, Ra=1708.0, Pr=1.0)
P = lin.parametric(24, over=("Ra", "k"), drift=32, drift_tol=1e-6)
crit = X.critical(P, "Ra", bracket=(1500.0, 2000.0), over={"k": (2.5, 3.8)}, xtol=1e-9)
```

## Linear boundary-value problem

$u'' = -\pi^2 \sin(\pi x)$ with $u = 0$ at the walls. Read `u` on the points `x`, and `sol.max_residual`.

```python
import numpy as np
import conatus as cn

m = cn.Model(cn.Box(x=cn.walls(0, 1)), name="Poisson")
m.scalar("u")
m.given(f=lambda x: -np.pi**2 * np.sin(np.pi * x))
m.equations("dx(dx(u)) = f")
m.at("walls", u=0)
sol = m.solve(32)
x, u = sol.coordinate("x"), sol.grid("u")
```

## Nonlinear steady state and a branch with its fold

The Bratu problem $u'' + \lambda e^u = 0$ on $[-1, 1]$. Read `sol.grid("u")` for the state at
$\lambda = 0.3$, on the Chebyshev points `x`, and `fold.p` for the fold of the branch.

```python
import conatus as cn
from conatus.expert import ContinuationSettings, NewtonSolver, NLBVPContinuation, grid

m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
m.scalar("u")
m.equations("dx(dx(u)) + lam*exp(u) = 0")
m.at("walls", u=0)
compiled = m.compile(48, dealias=1)
problem, field = compiled.problem, compiled.fields["u"]
newton = NewtonSolver(problem, tol=1e-13, damping=1.0, max_iterations=20, params={"lam": 0.3})
sol = newton.solve({field: lambda x: 0 * x})
x = grid(compiled.bases["x"], dealias=False)
settings = ContinuationSettings(ds=0.2, ds_min=1e-8, ds_max=0.6, adapt=1.5, fast_iterations=3,
                                tolerance=1e-12, max_newton=15, max_points=12, dp_fd=1e-6)
branch = NLBVPContinuation(newton, "lam", settings).run(sol.coeffs, 0.3, +1, p_bounds=(0.3, 1.0))
(fold,) = branch.folds()
```

## Time integration

The heat equation on a periodic line. Read `sim.grid("u")` on `sim.coordinate("x")` at `sim.t`.

```python
import numpy as np
import conatus as cn

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
sim = m.simulate(N=32, scheme="RK443", dt=1e-2)
sim.initial(u=lambda x: np.sin(x))
sim.run(until=0.5)
```

## Restart from a checkpoint

Write the state, read it into a new simulation, and go on. The restart gives the same bits as the run
that was not interrupted. Read `whole` and `split`.

```python
import numpy as np
import conatus as cn

def heat():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", nu=0.1)
    m.scalar("u")
    m.equations("dt(u) - nu*dx(dx(u)) = 0")
    sim = m.simulate(N=32, scheme="RK443", dt=1e-2)
    return sim

first = heat()
first.initial(u=lambda x: np.sin(x))
first.run(steps=25)
first.save("state.h5")
first.run(steps=25)
whole = first.grid("u")

second = heat()
second.restart("state.h5")
second.run(steps=25)
split = second.grid("u")
```

## Dry run of a script

Check a script before spending compute: it runs up to its first expensive call, and no time step is
taken. Read `report["ok"]`. The check executes the script's own code, so check only what you would run.

```python
from pathlib import Path
from conatus.agent import check

Path("run.py").write_text("""
import numpy as np
import conatus as cn
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
sim = m.simulate(N=32, scheme="RK443", dt=1e-2)
sim.initial(u=lambda x: np.sin(x))
sim.run(until=0.5)
open("finished.txt", "w")
""")
report = check("run.py")
```
