# explain() and the Methods paragraph

Every key object says, in plain text, what it is and which numerical choices it carries, and a
simulation writes the *Methods* paragraph of your paper from what it actually ran.

```python
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
res = m.linearize(about={"u": 0, "T": lambda z: -z}).eigenvalues(N=24, k=np.pi)
print(m.explain())        # the model in a few lines: equations, conditions, decisions, warnings
print(sim.explain())      # the same at the chosen resolution, plus the scheme and the checks
print(res.explain())      # an eigenvalue result: filters, growth convention, modes, residuals
print(sim.explain(full=True))   # every automatic decision, tau by tau and term by term
```

`explain()` is short by default: a summary of about ten lines in the words of your model. For a
simulation of the README's Rayleigh–Bénard problem:

```text
Simulation of 'Rayleigh-Benard 2D' at N = x: 16, z: 24: RK443, dt = 0.0005, backend cpu
Problem: IVP (a time derivative dt() appears)
Time stepping: 7 linear term(s) implicit; nonlinear terms explicit (momentum, heat)
Boundary conditions: bottom: u = 0, T = 0.5; top: u = 0, T = -0.5 (tau method, placed automatically)
Gauge: the mean of p is fixed (integ(p) = 0)
Dealiasing: x: 3/2, z: 3/2 (declared default for products of degree 2)
IVPSolver(scheme=RK443, dt=0.0005, constant step, backend=cpu)
  scheme: RK443: IMEX Runge–Kutta (4 stages), ... global order 3 (Ascher, Ruuth & Spiteri 1997, (4,4,3))
  finite-state check: every 100 steps (default), ...
(every decision, tau by tau and term by term, and the solver layout: sim.explain(full=True))
```

Warnings always stay in the short text (terms that are not polynomial in the unknowns, parameter
values given for one call, a symmetry subspace that hides symmetry-breaking instabilities, the
m = 0 projection of curvilinear solvers, spurious tau modes of a shell, the order of the scheme
with time-dependent boundary values, fallbacks and first-order modes of the right-hand side,
eigenproblem checks that were skipped). `explain(full=True)` is
the complete record.

This is how "no silent substitution" is made checkable ({doc}`../principles`): if a choice is not
in `explain(full=True)`, it is a bug. For a native model, the full text lists the term placement
(implicit or explicit) term by term, each boundary condition and the tau that imposes it, the
pressure gauge, the dealiasing factors with their origin (yours or the declared default) and the
bases chosen by the geometry; for a solver, the system layout (groups, buckets, bands), how $F$ is
evaluated and what runs where.

The split applies to the model, the compiled model (`m.compile(N)`), the `Simulation`, the
`Linearization`, the EVP result and the time steppers (`IVPSolver` and its polar, spherical and
distributed forms):

| Call | What it prints |
|---|---|
| `explain()` | problem, discretisation, scheme, the defaults and tolerances that matter, warnings |
| `explain(full=True)` | the complete record: every decision, the logs, the solver layout |

## What has `explain()`

| Object | What it prints |
|---|---|
| a basis | family, coefficients and map, dealias factor and grid size |
| an expression | node, tensor rank, domain, dtype, free fields and parameters, differential orders, hash; aliasing warning for non-polynomial pointwise functions |
| a problem (`IVP`, `EVP`, `LBVP`, `NLBVP`) | variables, parameters, equations (LHS implicit / RHS explicit), declared tau terms, time orders, the count analysis per bounded direction (`Status: CONSISTENT` or `CHECK`) |
| a plan | bases and operators, group sizes and nnz, the row space of each equation, how RHS data are projected, the discrete rank report (`PASS` / `FAIL`) |
| an EVP result | method, finite and infinite modes, pencil regularity, modes removed by the drift test, residual and drift of every kept mode |
| a verification report | each check with its value and threshold (`[PASS]` / to investigate) |
| `IVPSolver`, `CFL`, `EmbeddedError`, `TauWatch` | scheme, dt, start-up, backend, band structure, how $F$ is evaluated; what a controller does and does not do |
| a native model, a compiled model, a `Simulation` | the declarations, every automatic decision, the resolution, the solver |
| unit systems, presets, base flows | definitions, conventions, the defaults that were used |
| `conatus.expert` | what the single import contains and which names are shadowed |

`conatus.expert.explain()` lists what the expert import contains and any name that two modules
would export with different meanings ({doc}`expert`).

## A *Methods* paragraph for your paper

`sim.methods()` (in the expert layer `methods(plan, solver=None, eigen_solver=None)`, also
`plan.methods(...)`) writes a Methods paragraph **only from what the objects hold**: bases and sizes, the tau method and number
of taus, dealias per basis, NCC truncation, symmetry subspace, rank check, scheme, `dt`, start-up,
solver and backend, dtype. Nothing is paraphrased from defaults. For an EVP, pass
`eigen_solver="dense"` or `"shift-invert"`.

```python
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
mt = sim.methods()                     # Rayleigh–Bénard at N = (16, 24), RK443
print(mt.text)
print(mt.bibtex)                       # BibTeX of exactly the methods cited in the text
print(mt.explain())                    # text + citation keys + the facts it was built from
```

```text
The IVP 'Rayleigh-Benard 2D' for (u, p, T) was solved with Conatus. The domain is discretized as
follows: x ∈ [0, 2) is periodic and expanded in a real (cosine/sine) Fourier series of 16 modes;
z ∈ [-0.5, 0.5] is expanded in 24 Chebyshev polynomials of the first kind. Derivatives, basis
conversions and multiplications by non-constant coefficients use the sparse, banded ultraspherical
operators of Olver & Townsend, and boundary conditions are imposed with the generalized tau method
(4 tau variable(s), each entering through a lift polynomial), following Burns et al. The Fourier
directions decouple, giving 8 independent systems, one per wavenumber. Nonlinear terms are
evaluated pseudospectrally with dealiasing factors 3/2 in x, 3/2 in z. The structural rank of
every system was checked before factorization. Time integration uses the third-order, four-stage
IMEX Runge–Kutta scheme (4,4,3) of Ascher, Ruuth & Spiteri, whose implicit diagonal is constant
so that a single factorization serves all stages, with a constant step Δt = 0.0005. The implicit
systems are solved by banded LU factorization with a low-rank (Woodbury) correction for the tau
rows, batched over wavenumbers on CPU, in float64 arithmetic. The equations were declared in the
native layer of Conatus as written on paper; the implicit-explicit split, the 4 tau term(s) (each
in the equation where its unknown has its highest wall-normal order) and the pressure gauge and
its integral constraint were placed by fixed, declared rules (listed by explain()). The
dealiasing factor 3/2 is the declared default of the native layer.
```

The text is a starting point to edit, not a substitute for describing your study. Pair it with the
provenance record of the run ({doc}`reproducibility`) and cite Conatus itself (see the
*Citing Conatus* page).

## Tolerances: yours, or declared

A tolerance that decides when an iterative solver stops, which modes are returned, or how the time step
adapts has no default: leaving it out is an error that proposes a value and says what it controls.

```python
import numpy as np
import conatus as cn
from conatus.expert import critical

P = cn.problems.native.rayleigh_benard_evp(k=3.1, Ra=1708.0, Pr=1.0).parametric(
    24, over=("Ra", "k"), drift=32, drift_tol=1e-6)
crit = critical(P, "Ra", bracket=(1500, 2000), values={"k": 3.1}, xtol=1e-9)   # digits of Ra_c
m = cn.problems.native.orr_sommerfeld(Re=1e4, alpha=1.0, flow="poiseuille")
res = m.eigenvalues(N=64, drift=96, drift_tol=1e-6)             # which modes are kept
sim = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1).simulate(
    N=(16, 24), scheme="RK443", dt=5e-4)
dt = sim.cfl("u", safety=0.5)                                   # your margin on the advective limit
```

Thresholds that only detect or refuse, iteration caps that raise, and stopping tests at the attainable
floor (the residual is returned) keep a named default. `cn.tolerances.explain()` lists all of them with
their value and why a default is safe; the `explain()` of the object that uses one repeats it.
