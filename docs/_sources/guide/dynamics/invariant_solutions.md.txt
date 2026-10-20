# Invariant solutions

Equilibria, travelling waves and periodic orbits found by Newton–Krylov–hookstep on the flow map of your own time stepper.

An invariant solution is a state that the flow maps to itself, up to a symmetry: an equilibrium
$v(u) = 0$, a travelling wave $u(x - ct)$, a relative periodic orbit
$\tau_\ell\,\Phi_T(u) = u$. Conatus solves these equations with a Newton–Krylov method and a
trust-region (hookstep) step. The Jacobian is never formed: each Krylov vector costs one
linearized evaluation or time integration, exact for the discrete scheme.

## The flow map

The searches use the time stepper of the model, here Kuramoto–Sivashinsky on a periodic domain of
length 22:

```{literalinclude} _check.py
:language: python
:start-after: "[inv-model]"
:end-before: "[/inv-model]"
:dedent: 4
```

The tolerance of the Newton–Krylov solver is required (it decides the digits of the solution);
its algorithm controls (iteration budget, Krylov tolerance and size, initial, minimum and maximum
trust radius) have declared defaults, printed by `newton.explain()` and marked "(default)": they
change how fast the residual reaches the tolerance, not the answer, whose residual is returned.

**Scope.** `find_equilibrium`, `find_travelling_wave` and `find_relative_periodic_orbit` work on
problems with **one periodic direction** (a real Fourier basis, no other axis): Kuramoto–Sivashinsky,
1D Swift–Hohenberg and similar. They accept the `Simulation` (or its `sim.solver`; a `cn.Model`
with `N=` for equilibria and waves, whose search does not depend on the time scheme). Anything else
is refused with this scope and the alternatives: steady states of large problems in symmetry
subspaces (plane Couette, `conatus.ecs.steady.AffineSteadyProblem`), pipe travelling waves
(`conatus.ecs.pipe.PipeTravellingWave`, experimental: {doc}`../experimental`), 2D Kolmogorov flow
(`conatus.ecs.kolmogorov`). A generalisation to every geometry is planned, not available.

The seed `x0` of every finder is the state vector of the solver (`sim.solver.X`), the simulation
itself (its current state), or the grid values of the unknown (`sim.grid("u")`), converted to the
state vector; the conversion and the size of the round trip are in the result's notes, and any
other array is refused with the accepted shapes. The stability calls (`equilibrium_eigenvalues`,
`floquet_multipliers`) take the simulation as the finders do. A Newton search converges only from
a seed near the solution: most states of a chaotic run fall back to an equilibrium or fail
(tutorial 5 keeps stored seeds). A result prints on one line (`print(tw)`); `explain()` has the
evidence.

## Equilibria

```{literalinclude} _check.py
:language: python
:start-after: "[inv-eq]"
:end-before: "[/inv-eq]"
:dedent: 4
```

An equilibrium of a translation-invariant system is not isolated (every translate is one), so
`find_equilibrium` works in the subspace of odd states $u(-x) = -u(x)$, where translations are
broken; the seed must lie in it, which is checked. `equilibrium_eigenvalues` returns the leading
eigenvalues of the Jacobian in the full space, translation mode included.

## Travelling waves

```{literalinclude} _check.py
:language: python
:start-after: "[inv-tw]"
:end-before: "[/inv-tw]"
:dedent: 4
```

The speed `c` is an unknown, fixed by a phase condition. The residual is
$\|v(U) + c\,\partial_x U\| / \|U\|$, evaluated in the full space. A solution whose speed term
$|c|\,\|\partial_x U\|$ is below the Newton accuracy (`TW_SPEED_TOL` times `tol` times the scale
of the solve) is an equilibrium, not a travelling wave: it is returned as an `Equilibrium` (kind
`"equilibrium"`, `params["c"]` kept, the measured value in the notes). A Newton run that lands on
$u = 0$ is retracted as the trivial state (`TRIVIAL_TOL`); the boundary-value `NewtonSolver` flags
the same case with `trivial=True` and a note in `explain()` (it is not refused there: $u = 0$ may be
the wanted answer).

## Periodic and pre-periodic orbits

`find_relative_periodic_orbit(solver, X0, T0, ell0, newton, steps=..., reflect=False)` solves
$\tau_\ell\,\Phi_T(u) = u$ for $(u, T, \ell)$ with time and translation phase conditions;
`reflect=True` gives pre-periodic orbits $R\,\tau_\ell\,\Phi_T(u) = u$. The number of time steps
over one period is part of the problem. `floquet_multipliers` returns the multipliers of the orbit
and identifies its marginal directions. Seeds come from near-recurrences of a long run; tutorial
{doc}`../../tutorials/05_ks_orbits` carries one to the published Floquet exponents.

## Thresholds and edge states

The boundary between initial states that decay and those that become turbulent is located by
bisection on an amplitude. Conatus returns the threshold as a bracket and never counts an
undecided run on either side:

```{literalinclude} _check.py
:language: python
:start-after: "[inv-threshold]"
:end-before: "[/inv-threshold]"
:dedent: 4
```

The verdict function is yours (a time integration and declared thresholds). `EdgeTracker` repeats
the bisection along the trajectory to follow the edge state itself. For 3D shear flows,
`NLOP.from_solver` (module `conatus.adjoint.nlop`) computes the nonlinear optimal perturbation
that seeds such a bisection.

## Validation

A solution is accepted on its residual, reported with the iteration history (`explain()`). Its
stability is a separate computation (eigenvalues or Floquet multipliers in the full space). A
solution found in a symmetry subspace may be unstable outside it (see {doc}`symmetry`).

**See also:** tutorial {doc}`../../tutorials/05_ks_orbits` (Cvitanović et al. 2010);
{doc}`NewtonKrylovHookstep <../../api/_generated/_stubs/conatus.expert.NewtonKrylovHookstep>`, {doc}`find_equilibrium <../../api/_generated/_stubs/conatus.expert.find_equilibrium>`,
{doc}`find_travelling_wave <../../api/_generated/_stubs/conatus.expert.find_travelling_wave>`,
{doc}`find_relative_periodic_orbit <../../api/_generated/_stubs/conatus.expert.find_relative_periodic_orbit>`,
{doc}`ThresholdBisection <../../api/_generated/_stubs/conatus.expert.ThresholdBisection>`, {doc}`EdgeTracker <../../api/_generated/_stubs/conatus.expert.EdgeTracker>`.
