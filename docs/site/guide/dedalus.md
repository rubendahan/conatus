# Importing Dedalus scripts

`cn.from_dedalus(script)` builds a Conatus problem from the supported subset of a Dedalus 3 script
and reports translated, approximated and unsupported constructs. It reports, line by line, what it translated. {doc}`../tutorials/14_from_dedalus` goes
through one complete example.

## Usage

The call returns:

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

In a notebook, `tr` displays the report as a colour-coded table.

## How the script is read

The translator does not execute the script. Python's `ast` module parses it, and a restricted
interpreter walks it statement by statement. It understands:

- literals, arithmetic, tuples and lambdas;
- the `dedalus.public` names it knows;
- a whitelist of NumPy/SciPy functions, used for constants and initial data;
- the equation strings, which go through the {doc}`equations` parser.

Python executes no statement of the script, and the translator does not import Dedalus.

Each statement gets one of four statuses:

| Status | Meaning |
|---|---|
| **translated** | same mathematics and same method; the note lists any declared rewrite (`trace(grad(u)) -> div(u)`) or default of the source script made explicit (`dealias=1`) |
| **approximated** | translated, but the method differs in a way that can change digits; the note says how |
| **unsupported** | not translated, with the reason and what to write instead; everything that depends on it is reported as well |
| **skipped** | output only: logging, plots, reading results after the solve |

`tr.complete` is true only when a problem was translated with no line unsupported and none
approximated. A time loop whose body changes the simulation (writes field data, calls a mutating
method, stops on a condition) is unsupported, because `solver.run(n)` cannot reproduce it.
`**options` in a call are expanded. A problem's own `namespace={...}` alone resolves the names of
its equations. `tr.to_script()` returns a runnable program. It translates the embedded script
again and refuses to run if a line is unsupported. It then runs the IVP or solves the LBVP. For an
EVP or NLBVP it stops at the discretized plan, because the group or the initial guess is your
choice.

## The translated subset

| Dedalus | Conatus | Note |
|---|---|---|
| `Coordinate`, `CartesianCoordinates`, `PolarCoordinates`, `Distributor(dtype=...)` | coordinate systems; field dtype | process mesh and `comm` are ignored (parallel layout is decided at run time) |
| `RealFourier`, `ChebyshevT`/`Chebyshev`, `Legendre`, `Jacobi(a, b)` | same bases | an omitted `dealias` takes the source default of 1, **written explicitly** |
| `ComplexFourier(N)` | `ComplexFourier(N)` | **approximated**: the source basis keeps the Nyquist mode $k=-N/2$; Conatus drops it, so that group is absent |
| `DiskBasis(coords, shape=(Nphi, Nr))` | `RealFourier`/`ComplexFourier(phi)` × `DiskBasis(r)` | same polynomial space for each $m$ |
| `AnnulusBasis` | × `AnnulusBasis(r)` | **approximated**: Conatus multiplies the equations by a declared power of $r$ |
| `SphereBasis`, `ShellBasis`, `BallBasis`, `S2Coordinates`, `SphericalCoordinates` | — | **unsupported** in the translator: the geometries exist in Conatus ({doc}`geometries`), but the shape-to-size mapping is not automatic |
| `dist.Field`, `VectorField` (named or not) | `Field`, `VectorField` | an unnamed field takes its variable's name (reported) |
| `IVP`, `LBVP`, `EVP(eigenvalue=s)`, `NLBVP`, `IVP(time=t)` | same classes | a 0-dimensional eigenvalue or time field becomes a `Parameter` |
| `add_equation("...")`, `add_equation((lhs, rhs))` | `add_equation` | `u*v` of two tensors is `outer(u, v)` (as in the source language, reported) |
| `Lift(A, basis.derivative_basis(q), n)` | `lift(A, basis, n, lam=q)` | Chebyshev, Legendre, Jacobi, `ChebyshevU/V`, `Ultraspherical` and disk bases (data given in coefficient space on `ChebyshevU/V` is refused: their coefficients are normalised differently) |
| `trace`, `div(ez*lift(tau))` of first-order formulations | `div`, `dot`, `dz` | exact identities, listed in the note |
| `SBDF1-4`, `CNAB1`, `CNAB2`, `MCNAB2`, `CNLF2`, `RK111`, `RK222`, `RK443`, `RKSMR`, `RKGFY` | same schemes, same start-up (lower order first) | time-dependent boundary data: `tr.ivp_solver()` carries them in the explicit part, as the source script does (`bc_time="step"`, said by `explain()`); pass `bc_time="auto"` or `"stage"` to `ivp_solver` for the other conventions ({doc}`timesteppers`); `SBDF2_R`, `RK443_IMEX` are unsupported |
| `solver.step(dt)` in `while solver.proceed` | `solver.run(n)` with fixed `dt` | the time loop becomes one call |
| `CFL(...)`, `compute_timestep()` | — | **unsupported**: Conatus runs with a fixed `dt`; give `dt`, and `conatus.expert.CFL` reports the CFL step |
| `f['g'] = ...`, `f['g'][i] += ...` with grids from `local_grids` | grid functions `tr.data`, used by `set_state` / `ncc` | NumPy/SciPy whitelist |
| `fill_random`, `low_pass_filter`, `f['c'] = ...` | — | **unsupported**: the random layout of the source cannot be reproduced; write an explicit initial condition |
| file handlers, `GlobalFlowProperty` | — | **unsupported**: use `conatus.runs` and the Studio |
| `solve_dense` / `solve_sparse(subproblems[i])` | `eigenvalues(plan, group)` | **approximated**: a subproblem index of the source is not a Conatus group; select the group by wavenumber |

Scripts that build their problem inside a function (for example `rayleigh_benard_evp.py`) are
translated with `from_dedalus(script, entry=("max_growth_rate", dict(Rayleigh=1710, ...)))`.
`overrides={"Nx": 64}` replaces top-level assignments (also a name bound by unpacking, `Nx, Nz = 256, 64`), and each override is listed in the report; an override that matches no top-level assignment is refused with the names that can be overridden.

## Checking a translation

The translator is tested on every official Dedalus 3 example, read unmodified. Six of them were
also run in both forms on a reduced case (smaller sizes, a short time, a deterministic initial
condition, a fixed `dt`): KdV–Burgers, Rayleigh–Bénard, the 2D shear flow, the Poisson problem, the
Rayleigh–Bénard eigenvalue problem and the pipe-flow eigenvalues. The two results agree to
rounding. The spherical examples (sphere, shell, ball) are not translated automatically; write
them as native models ({doc}`geometries`). To check your own translation, run the original script
and the translated problem on a small case and compare one output quantity, as the optional
check script of {doc}`../tutorials/14_from_dedalus` does.
