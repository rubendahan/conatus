# Tangent, adjoint and optimal perturbations

`FlowMap` is the time-stepping map of a solver, X_0 ↦ X_N, with its **exact discrete** tangent
(`jvp`) and adjoint (`vjp`): the same schemes, the same operators, the same linear factorizations,
transposed. Nothing is approximated by a continuous adjoint, so the dot-product test holds to
rounding and the Taylor remainder is of order 2. `NLOP` builds on it: nonlinear optimal
perturbations (maximum energy growth at a horizon T for a given initial energy E0) and, in the limit
E0 → 0, the linear optimal gain.

## In the native layer

```{literalinclude} ../_check_features.py
:language: python
:start-after: "# [nlop-shell]"
:end-before: "# [/nlop-shell]"
:dedent: 4
```

```{literalinclude} ../outputs/flowmap_nlop.txt
:language: text
```

Started from the linear optimum, the nonlinear optimisation at $E_0 = 10^{-10}$ stays on it: in
this limit the NLOP gain is the linear optimal gain.

No tau, lift or gauge is written: the native layer places them, and they never enter the energy.
`sim.flowmap(T, t0=...)` and `sim.nlop(..., t0=...)` take the start time when the equations depend
on time (`time_parameter="t"`); `checkpoints=k` reverses the map with Revolve and k stored states.

## In the expert layer

`FlowMap(solver, nsteps, t0=...)` wraps an `IVPSolver`: `fm.jvp(X0, V)` is the tangent at `X0`,
`fm.vjp(X0, W)` the adjoint, and `fm.pair(a, b)` the pairing for which `vjp` is the adjoint of
`jvp`. `NLOP.from_solver(solver, T=..., E0=..., fields=(...), t0=...)` is the nonlinear optimisation
on the same map. Test an adjoint with the dot-product error normalised by |W| |JV|: the raw
relative error of a stiff map on random vectors measures its conditioning, not the adjoint. In a
Cartesian solver the energy `fields` are the scalar velocity components (`u`, `v`, `w`); on the
shell it is the vector `u`. The pairing is:

- Cartesian solvers (real state): the bilinear pairing Σ a b;
- spherical shells (complex state: stored modes m ≥ 0 of real fields, the modes −m being their
  conjugates): the **real pairing** Re Σ conj(a) b, since the synthesis conjugates the stored
  modes and the step is only real-linear. The linear solves of the adjoint use the conjugate
  transposes, with the same factorisations as the forward step.

The energy inner product of `NLOP` is exact: Parseval along Fourier directions, exact Chebyshev
Gram matrices (with r² on the shell), orthonormal spin harmonics. The admissible set is declared by
`explain()`: on the shell, exactly divergence-free fields that vanish on both walls, with real m = 0
modes (`free_taus=True` also lets the tau unknowns absorb the constraints).

## What is supported

| Geometry | Tangent / adjoint | NLOP | GPU |
|---|---|---|---|
| Cartesian (Fourier × Chebyshev) | yes, polynomial and general-form RHS, time-dependent RHS | yes | yes |
| Spherical shell (RealFourier × SphereBasis × ShellBasis) | yes: Coriolis explicit or implicit, u·∇u | yes | yes |
| Ball, S², polar, cylinder; symmetry-reduced solvers | refused with a message (transposed transforms not written yet) | — | — |

## Validation

Dot-product and Taylor tests on every path, and on the shell: the pure Stokes optimal gain against
exp(−2 λ₁ T) with λ₁ from the spherical Bessel relation (and the eigenvalue solver), the Coriolis
sign of the adjoint (the energy adjoint of the rotating map is the map with −Ω), and the NLOP gain
against the linear optimal gain as E0 → 0. The numbers are on the
[validation page](https://rubendahan.github.io/conatus/validation/#dynamics).
