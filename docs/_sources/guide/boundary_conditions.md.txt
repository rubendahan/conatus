# Boundary conditions

Conditions are declared by boundary name, as on paper; Conatus decides how each one enters the
discrete system and prints that decision.

```python
import conatus as cn

m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="convection", Ra=2500, Pr=1)
u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
m.at("bottom", u=0, T=+0.5)             # no slip, hot plate
m.at("top", u=0, T=-0.5)                # no slip, cold plate

os_ = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
v = os_.scalar("v")
os_.at("walls", "v = 0", "dy(v) = 0")   # conditions as text, on both walls
```

Keywords set an unknown to a value (a number, or a callable of the boundary's coordinates). Text
conditions accept any equation that involves one unknown at that boundary: derivatives
(`dz(T) = 0`), Robin conditions (`dz(T) + Bi*T = 0`), values given by a parameter. Boundary names
come from the geometry ({doc}`geometries`): `walls` stands for both ends.

## How many conditions

Each unknown needs as many conditions as its order along the bounded coordinate, times the number
of walls divided by two. A temperature with $\nabla^2 T$ between two plates needs two; the
Orr–Sommerfeld equation, fourth order in $y$, needs four (two per wall); in a disk, a ball or a
pipe the centre is not a boundary, so a second-order unknown needs one condition at `wall`.
A pressure that only appears under `grad(p)` needs none: Conatus recognises the incompressibility
structure (below).

A missing condition is refused before anything is built:

```python
import conatus as cn

m = cn.Model(cn.Box(x=cn.walls(0, 1)), name="heat", kappa=1.0)
T = m.scalar("T")
m.equations("dt(T) - kappa*lap(T) = 0")
m.at("x=0", T=0)
try:
    m.simulate(N=16, scheme="SBDF2", dt=1e-3)
except cn.errors.BoundaryConditionError as err:
    print(err)
```

```text
BoundaryConditionError T: 1 boundary condition(s) for 2 needed (order 2 along x, 2 wall(s))
  where: at(...) calls: at('x=0', T = 0)
  why:   missing conditions leave the discrete system singular
  fix:   declare 2 condition(s) on T (boundaries: 'bottom', 'top', 'x=0', 'x=1', 'walls')
```

## What Conatus does with them

Conatus imposes conditions with the tau method: each condition adds one unknown constant along the
wall, which enters one equation through a fixed polynomial. You do not write these terms. They are
placed by one deterministic rule:

- each condition constrains **one** unknown;
- it goes into the equation where that unknown has its **highest order** along the bounded
  coordinate;
- inside that equation, the conditions fill the highest modes in the order you declared them.

`m.explain()` lists every placement:

```text
Boundary conditions -> tau terms (automatic, deterministic):
  T = 0 at walls (left: x=0): tau_T1 lifted into mode N_x-1 of [0] eq0 -- T has its highest x-order (2) there
  T = 0 at walls (right: x=1): tau_T2 lifted into mode N_x-2 of [0] eq0 -- T has its highest x-order (2) there
```

**Incompressibility** is recognised from the structure of the equations, from a divergence
constraint. The pattern is a scalar that appears only under `grad(·)` and has no condition,
paired with the single equation without a time derivative that contains `div(u)`. Conatus then fixes the mean pressure
(`integ(p) = 0`) and adds the divergence correction of the geometry. `m.pressure("p")` only asks
Conatus to check that this structure is there.

## When the rule cannot decide

When the rule cannot decide, Conatus raises an error. Each message names the condition and what to write:

| Situation | What to write |
|---|---|
| the unknown has its highest order in two equations | `m.at(..., into="momentum")`: name the equation that receives the condition |
| a condition involves two unknowns, or none | split it, or write the problem in the {doc}`expert layer <expert>` |
| too many conditions for an unknown | remove one (the extra one would perturb an equation) |
| a scalar condition on one component of a vector unknown | the {doc}`expert layer <expert>`; a stress-free wall is `u=cn.free_slip` (below) |
| an unknown boundary name | one of the names listed by `geometry.boundaries()` |

Writing `lift(...)` yourself in a native model is refused: the placement is Conatus's job, and the
expert layer is there when you want to do it by hand ({doc}`taus`).

## Spurious modes in eigenvalue problems

Truncation creates a few spurious eigenvalues (very large, sometimes unstable). Conatus does not
filter them with a hidden threshold. `m.eigenvalues(N=64, drift=96, drift_tol=1e-6)` solves the problem again
at 96 modes and keeps only the eigenvalues found at both resolutions within the drift tolerance
(yours, required: it decides which modes are returned); the result says how many were removed. A physical filter is allowed, but it is yours and it is logged:
`res.select(np.flatnonzero(abs(res.values) < 2), why="|c| < 2")`. Worked example: {doc}`../tutorials/01_orr_sommerfeld`.

## Free-slip walls in the native layer

In the native layer (`cn.Model`) conditions are written by boundary name and the taus are
placed for you. A stress-free wall is one declaration on the velocity:

```python
import conatus as cn

box = cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1))
free = cn.Model(box, name="RB stress-free", Ra=700, Pr=1)
u, T = free.velocity("u"), free.scalar("T")
free.at("walls", u=cn.free_slip, T=0)          # u·n = 0 and dz(u_t) = 0 at both walls

rigid = cn.Model(box, name="RB rigid", Ra=2000, Pr=1)
u, T = rigid.velocity("u"), rigid.scalar("T")
rigid.at("walls", u=cn.no_slip, T=0)           # the same as u=0, named for symmetry

slip = cn.Model(box, name="partial slip", nu=1.0)
u = slip.velocity("u")
slip.pressure("p")
slip.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
slip.at("bottom", u=cn.navier_slip(0.1))       # u·n = 0, u_t + 0.1 dn(u_t) = 0 (Navier slip length)
slip.at("top", u=cn.no_slip)
slip.simulate(N=(16, 16), scheme="RK443", dt=1e-3).run(steps=2)
```

For a plane wall of normal $n = e_z$ it is compiled as **one vector condition** on the single
unknown $u$,

$$ B(u) = (I - n n)\cdot\partial_z u + n\,(n\cdot u) = 0, $$

whose normal component is $u_z = 0$ and whose tangential components are $\partial_z u_x = 0$
($\partial_z u_y = 0$ in 3D). It therefore reuses the vector-tau rule unchanged: one vector tau per
wall in the equation where $u$ has its highest $z$-order (momentum), and the divergence tau of the
Cartesian recipe in the continuity equation. For the stress-free Rayleigh–Bénard model of
{doc}`bases` (*Sine/cosine walls*):

```{literalinclude} _check_features.py
:language: python
:start-after: "# [sincos]"
:end-before: "# [/sincos]"
:dedent: 4
```

`m.explain(full=True)` prints the choices; with Chebyshev walls, the taus are:

```{literalinclude} _check_features.py
:language: python
:start-after: "# [sincos-explain]"
:end-before: "# [/sincos-explain]"
:dedent: 4
```

```{literalinclude} outputs/sincos_free_slip.txt
:language: text
:lines: 9-
```

Validation: the Rayleigh–Bénard onset between stress-free
isothermal walls, $Ra_c = 27\pi^4/4$ at $k_c = \pi/\sqrt{2}$ (Rayleigh 1916; Chandrasekhar 1961),
and the exact growth rate of the mode $n = 1$ to $\sim 10^{-12}$ at $N_z = 16$; the same model
text with `cn.walls(0, 1, basis="sincos")` needs no tau at all ({doc}`bases`, *Sine/cosine
walls*).

On curved walls (disk, annulus, pipe, ball, shell) `m.at("wall", u=cn.free_slip)` (or `"walls"`,
`"inner"`, `"outer"`) means $u_r = 0$ and a vanishing tangential traction,
$r\,\partial_r(u_t/r) = 0$, written without any division by $r$ as one vector condition per wall,
$R\,P_t\cdot\partial_r u + K\cdot u = 0$ with $P_t = I - e_r e_r$ and $K = e_r e_r - P_c$ ($P_c$ projects on
the curved tangential directions; along the pipe axis the condition is $\partial_r u_z = 0$). The
taus are those of no-slip. Checks: a solid-body rotation is
a steady solution (drift $4\times10^{-15}$ in the ball, $2\times10^{-14}$ in the shell over 100 steps,
while no-slip does not even admit it), Stokes decay rates match their Bessel closed forms to
$2\times10^{-12}$ (disk, pipe, ball), $8\times10^{-13}$ (annulus) and $10^{-10}$ (shell). The rank report flags the rigid rotations (groups $\ell=1$ or
$m=0$): a physical kernel, angular momentum is conserved with stress-free walls.

## Interfaces, half lines and the whole line

On a compound interval (`cn.walls(a, c, b)`) you declare the wall conditions only. For each
unknown of order $q$ the native layer adds, at every interface, the continuity of the value and
of the derivatives up to $q-1$, written with one-sided traces, and gives each piece $q$ tau terms
in its last modes; `explain()` lists them (see {doc}`bases`). In the IR the same conditions read

```python
import numpy as np
from conatus.expert import (LBVP, CartesianCoordinates, CompoundChebyshev, Domain, Field, diff,
                            discretize, interp, lap, lift)

cs = CartesianCoordinates("x"); x = cs["x"]
b = CompoundChebyshev(x, (24, 24), (-1, 0.0, 1), dealias=1)     # two pieces, interface x = 0
D = Domain(cs, [b])
u, f = Field("u", D), Field("f", D)
t01, t02, t11, t12 = (Field(name, D.without(x)) for name in ("t01", "t02", "t11", "t12"))
pb = LBVP([u, t01, t02, t11, t12], name="Poisson on two pieces")
pb.add_equation((lap(u) + lift(t01, b, -1, piece=0) + lift(t02, b, -2, piece=0)
                 + lift(t11, b, -1, piece=1) + lift(t12, b, -2, piece=1), f))
pb.add_equation((interp(u, x, 0.0, side="above") - interp(u, x, 0.0, side="below"), 0))
pb.add_equation((interp(diff(u, x), x, 0.0, side="above")
                 - interp(diff(u, x), x, 0.0, side="below"), 0))
pb.add_equation((interp(u, x, "left"), 0))
pb.add_equation((interp(u, x, "right"), 0))
sol = discretize(pb).solve({f: lambda X: -np.pi**2 * np.sin(np.pi * X)})   # u = sin(pi x)
```

and a trace at an interface without `side=` is refused (it is one-sided there). The banded solver
keeps, for each piece, as many pinned columns as that piece has taus (otherwise its banded part
would split into non-square blocks).

`cn.half_line(...)` has one wall (`"wall"`): an equation of order $q$ takes $q/2$ conditions
there, exactly like the radial axis of a disk. `cn.line(...)` has no boundary: a condition on it
is refused (`no boundary`), and so is a trace at $\pm\infty$ in the IR (the Hermite functions
decay; there is nothing to impose).

## Boundary values that depend on time

`t` is the time of a time-dependent model. A boundary value may be any expression of `t`, the
parameters and the given fields, written as text or as a function of `t`:

```python
import numpy as np
import conatus as cn

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="moving walls")
u, T = m.scalar("u"), m.scalar("T")
m.given(S=lambda x: np.sin(x), C=lambda x: np.cos(x))
m.equations("dt(u) - lap(u) = 0", "dt(T) - lap(T) = 0")
m.at("bottom", u="S*cos(t) - C*sin(t)",          # sin(x - t): separable form with given fields
     T="cos(t)")                                 # keyword value as text
m.at("top", "u = 0.5*sin(2*t)",                  # condition text
     T=lambda t: np.exp(-t))                     # numpy functions of t build the expression
m.simulate(N=(16, 16), scheme="RK222", dt=1e-2).run(steps=2)
```

A function of `t` **and** of coordinates (`lambda x, t: np.sin(x - t)`) is refused with the
separable form to write instead: a coordinate itself is not smooth data on a periodic axis. `t` in
an implicit term (`t*u` on the left) or in a model without `dt()` is refused. Such values are
imposed by default with the stage values that the scheme gives for $g'(t)$, exact here
(`bc_time="auto"` → `"rk"`), so that the Runge–Kutta schemes keep their order (RK443 2.8 instead
of 2.0 with $g$ at the stage times). `bc_time="stage"` and `bc_time="step"` are the other
conventions. See {doc}`timesteppers`. In the expert layer the same holds for
`pb.add_bc((interp(u, y, 0), cos(t)))` with `IVPSolver(..., time_parameter="t")`.

On a disk, an annulus, a pipe or a sphere, a given field of the wall coordinate used in a wall
condition (`m.given(g=lambda phi, r: ...)`, `m.at("wall", "T = g")`) is replaced by its trace at the
wall, evaluated from its function (exact, listed by `explain()`); under a derivative (`dr(g)`) it
is refused: declare the wall value as a function of the other coordinates.
