# Coming from Dedalus

If you know Dedalus, you already know how to write a Conatus model; what changes is what you no
longer write, and what you can do with the model afterwards.

## The same problem, side by side

Two-dimensional Boussinesq convection between a hot and a cold wall, periodic sideways
({doc}`../tutorials/14_from_dedalus` runs both and compares them). On the left, the problem
part of a Dedalus 3 script written for that tutorial ([`rb_cell_dedalus.py`](../tutorials/code/rb_cell_dedalus.py));
on the right, the same cell as a Conatus model:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
**Dedalus**

```{literalinclude} ../tutorials/code/rb_cell_dedalus.py
:language: python
:start-after: "# Coordinates, bases"
:end-before: "# Fixed-step"
```
:::

:::{grid-item}
**Conatus**

```{literalinclude} ../tutorials/code/t14_from_dedalus.py
:language: python
:start-after: "# [native]"
:end-before: "# [/native]"
```
:::
::::

## What you stop writing

- **Tau fields and lifts.** Each condition is placed by a fixed rule, printed by `explain()`
  ({doc}`boundary_conditions`); the pressure gauge is recognised from the structure.
- **Bases with sizes.** The geometry fixes the bases; the number of modes is given when you solve.
- **The split into left and right sides.** Linear terms are implicit, nonlinear terms explicit,
  wherever you wrote them ({doc}`equations`).
- **The time loop.** `sim.run(until=t)` with a fixed `dt`; Conatus never adapts `dt` for you.

What stays the same: the spectral method itself (Fourier, Chebyshev and curvilinear bases with
sparse ultraspherical operators, the generalised tau method, IMEX schemes with Dedalus's
coefficients), the string syntax of the equations, and the dealiasing convention. On the problems
where both codes apply the same method, they agree to round-off.

What is added: the GPU backend, and the dynamics toolkit on the same model: eigenvalues with
residuals, adjoints, Newton–Krylov searches for invariant solutions, continuation, Floquet and
Lyapunov analysis ({doc}`dynamics/index`).

Bases and conditions you may know from Dedalus 2 are declarations of the axis in the native
layer: sine and cosine series between stress-free or fixed-value walls
(`cn.walls(0, 1, basis="sincos")`, the parity of each field read from its conditions), Hermite
functions on the whole line (`cn.line(scale=...)`), Laguerre functions on the half line
(`cn.half_line(scale=...)`) and compound Chebyshev intervals (`cn.walls(a, c, b)`, continuity at
the cuts added for you); a stress-free wall is `u=cn.free_slip`, on plane and curved walls
({doc}`bases`, {doc}`boundary_conditions`). All thirteen time steppers of Dedalus 3 are there,
with the same coefficients ({doc}`timesteppers`).

## Importing a Dedalus script

`cn.from_dedalus(script)` translates a Dedalus 3 script into Conatus. You get back:

- the Conatus problem;
- the declared time stepping;
- the initial and given data;
- a **line-by-line report**.

```python
from pathlib import Path

import conatus as cn

Path("kdv_dedalus.py").write_text("""
import numpy as np
import dedalus.public as d3
period, Nx, damping, dispersion = 8, 64, 3e-4, 5e-4
s = d3.Coordinate('x')
mesh = d3.Distributor(s, dtype=np.float64)
ring = d3.RealFourier(s, size=Nx, bounds=(0, period), dealias=1.5)
h = mesh.Field(name='h', bases=ring)
D = lambda F: d3.Differentiate(F, s)
wave = d3.IVP([h], namespace=locals())
wave.add_equation("dt(h) - damping*D(D(h)) - dispersion*D(D(D(h))) = -h*D(h)")
xs = mesh.local_grid(ring)
h['g'] = np.exp(-2*(xs - 4)**2)
run = wave.build_solver(d3.SBDF3)
run.stop_sim_time = 0.025
dt_fixed = 2.5e-3
while run.proceed:
    run.step(dt_fixed)
""")                                                 # a Dedalus 3 script (KdV-Burgers)

tr = cn.from_dedalus("kdv_dedalus.py")            # a path, or the source text
print(tr.explain())                               # counts + one line per statement
tr.problem                                        # an ordinary Conatus Problem
solver = tr.ivp_solver(startup="lower_order")     # scheme and fixed dt from the script
solver.run(tr.steps())                            # stop_sim_time / dt steps, no Python loop
print(tr.to_script())                             # a runnable program, report as comments
```

In a notebook, `tr` displays the report as a colour-coded table. {doc}`../tutorials/14_from_dedalus`
goes through a complete example and compares the result with Dedalus.

## What happens to your script

The script is **read, not run**. Python's `ast` module parses it, and a restricted interpreter
walks it statement by statement. It understands:

- literals, arithmetic, tuples and lambdas;
- the `dedalus.public` names it knows;
- a whitelist of NumPy/SciPy functions, used for constants and initial data;
- the equation strings, which go through the {doc}`equations` parser.

No statement of the script is executed by Python, and Dedalus is never imported.

Each statement gets one of four statuses:

| Status | Meaning |
|---|---|
| **translated** | same mathematics and same method; the note lists any declared rewrite (`trace(grad(u)) -> div(u)`) or Dedalus default made explicit (`dealias=1`) |
| **approximated** | translated, but the method differs in a way that can change digits; the note says how |
| **unsupported** | not translated, with the reason and what to write instead; everything that depends on it is reported as well |
| **skipped** | output only: logging, plots, reading results after the solve |

`tr.complete` is true only when a problem was translated with no line unsupported and none
approximated. A time loop whose body changes the simulation (writes field data, calls a mutating
method, stops on a condition) is unsupported: `solver.run(n)` would not do it. `**options` in a
call are expanded, and a problem's own `namespace={...}` alone resolves the names of its equations,
as in Dedalus. `tr.to_script()` is a runnable program: it translates the embedded script again,
refuses to run if a line is unsupported, then runs the IVP or solves the LBVP (an EVP or NLBVP stops
at the discretized plan: the group or the initial guess is your choice).

## The translated subset

| Dedalus | Conatus | Note |
|---|---|---|
| `Coordinate`, `CartesianCoordinates`, `PolarCoordinates`, `Distributor(dtype=...)` | coordinate systems; field dtype | process mesh and `comm` are ignored (parallel layout is decided at run time) |
| `RealFourier`, `ChebyshevT`/`Chebyshev`, `Legendre`, `Jacobi(a, b)` | same bases | `dealias` omitted means Dedalus's default of 1, **written explicitly** |
| `ComplexFourier(N)` | `ComplexFourier(N)` | **approximated**: Dedalus keeps the Nyquist mode $k=-N/2$, Conatus drops it; that group is absent |
| `DiskBasis(coords, shape=(Nphi, Nr))` | `RealFourier`/`ComplexFourier(phi)` × `DiskBasis(r)` | same polynomial space for each $m$; the pipe-flow eigenvalues equal Dedalus's to round-off |
| `AnnulusBasis` | × `AnnulusBasis(r)` | **approximated**: Conatus multiplies the equations by a declared power of $r$ |
| `SphereBasis`, `ShellBasis`, `BallBasis`, `S2Coordinates`, `SphericalCoordinates` | — | **unsupported** in the translator: the geometries exist in Conatus ({doc}`geometries`), but the shape-to-size mapping is not automatic |
| `dist.Field`, `VectorField` (named or not) | `Field`, `VectorField` | an unnamed field takes its variable's name (reported) |
| `IVP`, `LBVP`, `EVP(eigenvalue=s)`, `NLBVP`, `IVP(time=t)` | same classes | a 0-dimensional eigenvalue or time field becomes a `Parameter` |
| `add_equation("...")`, `add_equation((lhs, rhs))` | `add_equation` | `u*v` of two tensors is `outer(u, v)` (Dedalus convention, reported) |
| `Lift(A, basis.derivative_basis(q), n)` | `lift(A, basis, n, lam=q)` | Chebyshev, Legendre, Jacobi, `ChebyshevU/V`, `Ultraspherical` and disk bases (data given in coefficient space on `ChebyshevU/V` is refused: their coefficients are normalised differently) |
| `trace`, `div(ez*lift(tau))` of first-order formulations | `div`, `dot`, `dz` | exact identities, listed in the note |
| `SBDF1-4`, `CNAB1`, `CNAB2`, `MCNAB2`, `CNLF2`, `RK111`, `RK222`, `RK443`, `RKSMR`, `RKGFY` | same schemes, same start-up (lower order first) | results agree with Dedalus to round-off; time-dependent boundary data: `tr.ivp_solver()` keeps the Dedalus convention (`bc_time="step"`, said by `explain()`), while a solver built by hand imposes them so that each Runge–Kutta scheme keeps its order (`bc_time="auto"`, which is `"rk"`: the implicit tableau applied to $g'$; `"stage"` imposes $g$ at the stage times; pass either to `ivp_solver`; {doc}`timesteppers`); `SBDF2_R`, `RK443_IMEX` are unsupported |
| `solver.step(dt)` in `while solver.proceed` | `solver.run(n)` with fixed `dt` | the time loop becomes one call |
| `CFL(...)`, `compute_timestep()` | — | **unsupported**: Conatus never changes `dt` by itself; give `dt`, `conatus.expert.CFL` reports the CFL step |
| `f['g'] = ...`, `f['g'][i] += ...` with grids from `local_grids` | grid functions `tr.data`, used by `set_state` / `ncc` | NumPy/SciPy whitelist |
| `fill_random`, `low_pass_filter`, `f['c'] = ...` | — | **unsupported**: Dedalus's random layout cannot be reproduced; write an explicit initial condition |
| file handlers, `GlobalFlowProperty` | — | **unsupported**: use `conatus.runs` and the Studio |
| `solve_dense` / `solve_sparse(subproblems[i])` | `eigenvalues(plan, group)` | **approximated**: a Dedalus subproblem index is not a Conatus group; select the group by wavenumber |

Scripts that build their problem inside a function (for example `rayleigh_benard_evp.py`) are
translated with `from_dedalus(script, entry=("max_growth_rate", dict(Rayleigh=1710, ...)))`.
`overrides={"Nx": 64}` replaces top-level assignments (also a name bound by unpacking, `Nx, Nz = 256, 64`), and each override is listed in the report; an override that matches no top-level assignment is refused with the names that can be overridden.

## The official Dedalus examples

Every official Dedalus 3 example was translated unmodified, which tests the coverage of the
translator. Where the comparison is possible, both codes then ran the same script (smaller sizes,
a short time, a deterministic initial condition instead of `fill_random`, a fixed `dt` instead of
the CFL): KdV–Burgers, Rayleigh–Bénard, the 2D shear flow, the Poisson problem, the Rayleigh–Bénard
eigenvalue problem and the pipe-flow eigenvalues agree to round-off. The spherical examples
(sphere, shell, ball) use geometries that Conatus has, but their translation is not automatic yet:
write them as native models ({doc}`geometries`).
