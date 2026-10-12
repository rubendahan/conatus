# Tau terms by hand

In the expert layer you place the boundary terms yourself; a rank check refuses an ill-posed
system before any factorisation.

Conatus imposes boundary conditions with the **generalised tau method**: for each condition, one
extra unknown (a *tau variable*, constant along the bounded coordinate) enters a bulk equation
through a *lift* polynomial, and the system stays square and banded. The native layer writes
these terms for you ({doc}`boundary_conditions`); this page is for problems you write by hand.

## The short way: `add_bc`

```python
import numpy as np
from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter,
                            dt, interp, lap)

cs = CartesianCoordinates("x"); x = cs["x"]
b = Chebyshev(x, 24, (-1, 1), dealias=3/2)
D = Domain(cs, [b])
u = Field("u", D)
a = Parameter("a", 1.0)
pb = IVP([u], name="heat 1D")
pb.add_equation((dt(u) - lap(u), -a * u))
pb.add_bc(interp(u, x, "left"), interp(u, x, "right"))
print(pb.explain())
```

```text
Equations (LHS implicit/linear | RHS explicit):
  [0] -lap(u) + lift(tau_x0_2, x, -2) + lift(tau_x0_1, x, -1) + dt(u) = -a*u
  [1] u(x=-1) = 0
  [2] u(x=1) = 0
```

`add_bc` creates one tau per condition and lifts them into modes $N-1, N-2, \dots$ of the target
equation, in the order of the conditions. The target is the only bulk equation along that
coordinate, or the one named with `into=` when there are several: it is never guessed. With a
symmetry subspace, `add_bc` refuses and asks you to declare the taus and their parities.

## The explicit way

The same problem with the taus written out, which is what `add_bc` produces:

```python
from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, dt,
                            interp, lap, lift)

cs = CartesianCoordinates("x"); x = cs["x"]
b = Chebyshev(x, 24, (-1, 1), dealias=3/2)
D = Domain(cs, [b])
u, a = Field("u", D), Parameter("a", 1.0)
t1, t2 = Field("t1", D.without(x)), Field("t2", D.without(x))   # constant along x
pb = IVP([u, t1, t2], name="heat 1D")
pb.add_equation((dt(u) - lap(u) + lift(t1, b, -1) + lift(t2, b, -2), -a * u))
pb.add_equation((interp(u, x, "left"), 0))
pb.add_equation((interp(u, x, "right"), 0))
```

`lift(tau, basis, n, lam=q)` can also lift into the ultraspherical basis $C^{(q)}$ of an order-$q$
equation, which puts the taus in its last $q$ rows (Dedalus's `derivative_basis(q)`); curvilinear
geometries use this form.

## The rank report refuses ill-posed systems

Two checks run before any factorisation:

1. **On the problem**, `pb.explain()` counts, per bounded direction, the differential order of
   the bulk equations, the lifted taus, the boundary conditions and the tau variables. A mismatch
   is shown as `Status: CHECK` (it can be legitimate: pressure gauges, mixed-order systems).
2. **On the discrete system**, each group's structural rank is computed. A system that is not
   square or not of full rank is refused with a `RankError` before the solver is built, instead
   of a "singular matrix" deep inside a factorisation. Forgetting one boundary condition:

   ```text
   RankError: system refused before factorization:
   Unknown coefficient DOFs         24
   Tau DOFs                          2
   PDE equations                    24
   Boundary equations                1
   Structural rank                  25 / 26
   FAIL
   ```

   `plan.rank_report()` and `plan.check()` give the same analysis on demand; `plan.explain()`
   ends with it.

## Watching the taus during a run

`TauWatch(solver, growth_limit=1e3)` records the largest coefficient of each tau variable and
warns, naming the tau, when one grows by more than the declared factor. It never changes the
solver:

```python
import numpy as np
from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, IVPSolver,
                            TauWatch, discretize, dt, interp, lap)

cs = CartesianCoordinates("x"); x = cs["x"]
u = Field("u", Domain(cs, [Chebyshev(x, 24, (-1, 1), dealias=3/2)]))
pb = IVP([u], name="heat 1D")
pb.add_equation((dt(u) - lap(u), 0))
pb.add_bc(interp(u, x, "left"), interp(u, x, "right"))
solver = IVPSolver(discretize(pb), "RK443", 1e-3)
solver.set_state({u: lambda X: np.cos(np.pi * X / 2)})

tw = TauWatch(solver, growth_limit=1e3)
solver.run(1000, callback=tw, every=10)     # the period is explicit
print(sorted(tw.history))                   # ['tau_x0_1', 'tau_x0_2']: one series per tau
```
