# Adjoints and sensitivity

Exact tangent and adjoint of the discrete time stepper, adjoint eigenmodes in a declared inner product, the wavemaker and the sensitivity to the base flow.

## Tangent and adjoint of the time stepper

`FlowMap` turns a time integration $X_N = \Phi(X_0)$ into a differentiable map. `jvp` applies the
tangent $\mathrm d\Phi\,V$ and `vjp` its transpose $\mathrm d\Phi^{\mathsf T} W$, both exact for
the discrete scheme (same stages, same solves, same dealiasing), so that the dot-product test holds
to rounding:

```{literalinclude} _check.py
:language: python
:start-after: "[adj-flowmap]"
:end-before: "[/adj-flowmap]"
:dedent: 4
```

The transpose is taken in the plain pairing $\langle a, b\rangle = \sum_i a_i b_i$ of the
coefficients. **A transpose is not the physical adjoint.** In an inner product of weight $W$ (the
kinetic energy, for instance) the adjoint is

$$ L^\dagger = W^{-1} L^{*} W , $$

which `vjp_physical(X0, W, weight)` applies. On the spherical shell the state is complex (stored
modes $m \ge 0$ of real fields) and the step only real-linear: there the pairing is
$\langle a, b\rangle = \mathrm{Re}\sum_i \bar a_i b_i$ (`fm.pair`, declared by `explain()`); see
{doc}`optimal_perturbations`. With `f_backend="fused"` the shell adjoint follows the planned path of
the forward step: the transposes of the same radial and angular GEMMs, phi FFTs and product program
(one generated kernel per product tree on the GPU), so a `vjp` step costs 1.6 to 1.8 forward steps
on one CPU core instead of 3.0 to 3.3; the adjoint is the same up to rounding as the term-by-term
walk of the reference evaluator, which equations outside its scope keep (`fm.explain()` names
them). Long trajectories are reversed with binomial
checkpointing (`checkpoints=`). The gradient of any final-time objective costs one forward and one
backward integration.

A term that is not differentiable in the state, `abs(T)` or `sign(T)`, is refused when the tangent
is built (`UnsupportedError`, with the reason): the tangent and the adjoint do not exist where the
argument changes sign, and a trajectory crosses that set. Write a smooth approximation into the
model (`sqrt(T**2 + eps**2)`, `tanh(T/eps)` with `eps` a declared parameter), or differentiate a
model without the term. A linearization about a base keeps `abs` (its derivative `sign(T0)` exists
wherever the base does not vanish) and refuses `sign`.

## Checkpoints off the GPU and sweeps longer than one job

With Revolve (`checkpoints=k`) the checkpoints can live elsewhere than in device memory, and a
reverse sweep (or an optimal-perturbation iteration) can be cut by a time budget and resumed by the
next job. A copy moves bits unchanged: every store, and every cut-and-resume, gives the same result
bit for bit as the uninterrupted device run.

```python
import numpy as np

import conatus as cn
from conatus.adjoint import FlowMap, VJPSuspended

m = cn.problems.native.kuramoto_sivashinsky(L=22.0)
sim = m.simulate(N=32, scheme="RK443", dt=0.01)
sim.initial(u=lambda x: 0.1 * np.cos(2 * np.pi * x / 22.0))
solver, t0, X0 = sim.solver, sim.t, sim.solver.X.copy()

fm = FlowMap(solver, 200, t0=t0, checkpoints=8,  # a long run: 100_000 steps, checkpoints=400
             checkpoint_store="host")             # "device" (default) | "host" | "disk"
W = np.random.default_rng(0).standard_normal(fm.n)
try:
    g = fm.vjp(X0, W, journal="run/vjp", budget_s=0.0)   # budget_s=11 * 3600 in a 12 h job
except VJPSuspended:
    g = fm.vjp(X0, None, journal="run/vjp")      # the next job resumes the sweep
```

* `"host"` keeps the checkpoints in page-locked host memory (one device-to-host copy per stored
  state, cheap next to a step); `"disk"` writes one `.npy` per array in `checkpoint_dir` (atomic)
  and is meant for small states: Revolve stores almost every step once, so a large state written to
  disk at every step costs more than the step itself.
* `revolve_cost(N, k)` gives the forward steps recomputed (e.g. N = 100 000: 3.9 N with k = 40,
  3.0 N with k = 100 to 400, 2.0 N with k = 1000) and `explain()` prints it with the store.
* The journal (position in the Revolve action list, partial cotangents, live checkpoints) is written
  atomically: a job killed while writing it leaves the previous one, which the next job resumes. A
  journal of another computation (other X0, W, map, operators, parameters or backend) is refused.
  The journal directory may hold other files: only its own are ever removed.
* A suspension with `"host"` writes every live checkpoint to disk (up to k states, and two
  generations at once for a moment): leave that writing time between `budget_s` and the job's time
  limit. Each call takes at least one step before looking at the budget. For optimal
  perturbations, `NLOP.iterate(state, ..., journal=dir, budget_s=...)` stops inside the adjoint
  sweep or the line search, and `NLOPState.load(dir)` + the same call continue the iteration.

## Discrete and continuous adjoint modes

The discrete adjoint (adjoint of the discretised operator) and the continuous adjoint
(discretisation of the adjoint equations, derived symbolically with their boundary conditions) are
both available, and compared:

```{literalinclude} _check.py
:language: python
:start-after: "[adj-continuous]"
:end-before: "[/adj-continuous]"
:dedent: 4
```

## Wavemaker and base-flow sensitivity

From a direct mode $\hat q$ and its adjoint $\hat q^\dagger$, the structural sensitivity
$S(x) = \|\hat q^\dagger(x)\|\,\|\hat q(x)\|$ locates where a localized feedback changes the
eigenvalue most (the wavemaker). The sensitivity to the base flow gives the eigenvalue drift for
any modification of it. On the linear Ginzburg–Landau equation both have closed forms:

```{literalinclude} _check.py
:language: python
:start-after: "[adj-sensitivity]"
:end-before: "[/adj-sensitivity]"
:dedent: 4
```

`adjoint=` is a required choice (`"continuous"` or `"discrete"`), never implied.

**See also:** tutorial {doc}`../../tutorials/13_sensitivity_data` (Chomaz 2005);
{doc}`FlowMap <../../api/_generated/_stubs/conatus.expert.FlowMap>`, {doc}`compare_adjoints <../../api/_generated/_stubs/conatus.expert.compare_adjoints>`,
{doc}`InnerProduct <../../api/_generated/_stubs/conatus.expert.InnerProduct>`, {doc}`structural_sensitivity <../../api/_generated/_stubs/conatus.expert.structural_sensitivity>`,
{doc}`base_flow_sensitivity <../../api/_generated/_stubs/conatus.expert.base_flow_sensitivity>`.
