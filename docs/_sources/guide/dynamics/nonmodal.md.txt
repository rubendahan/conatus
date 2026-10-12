# Non-modal analysis

Transient growth of a linear operator, measured in the kinetic-energy norm.

A stable linear flow can amplify a perturbation by orders of magnitude before it decays, because
its eigenvectors are not orthogonal. Non-modal analysis measures that amplification in a declared
norm, here the kinetic energy $E = \tfrac12\int |\boldsymbol u|^2\,\mathrm dy$ computed with its
exact Gram matrix.

## The operator

```{literalinclude} _check.py
:language: python
:start-after: "[nm-system]"
:end-before: "[/nm-system]"
:dedent: 4
```

`parallel_shear_flow` builds the linearized Navier–Stokes equations about $U(y)\,\boldsymbol e_x$
in primitive variables for one pair of wavenumbers; the base flow is a symbolic expression in `Y`.
`state_space` removes the infinite eigenvalues of this descriptor system and writes its generator
$\tilde A$ in energy-orthonormal coordinates, so that 2-norms are energy norms.

## Optimal transient growth

$$ G(t) = \max_{q_0} \frac{E(t)}{E(0)} = \|e^{t\tilde A}\|_2^2 $$

```{literalinclude} _check.py
:language: python
:start-after: "[nm-growth]"
:end-before: "[/nm-growth]"
:dedent: 4
```

A scan of `n_scan` times is followed by a bounded Brent search whose tolerance `xtol` you give.
For a time-periodic base flow, transient growth uses the propagator of the time
stepper instead (next section).

## Any time-periodic base flow

Linearize the model about a base flow that depends on $t$ and give its period. The gain over
$[t_0, t_0 + T]$ and the Floquet exponents of one wavenumber then follow from the same operator:

- The base flow $U(y, t)$ is split into time harmonics $U = \sum_m U_m(y)\, e^{2\pi i m t/P}$ (a
  discrete Fourier transform over 64 instants per period); `explain()` lists the harmonics kept
  and the size of the largest one dropped, and a base that is not smooth in $t$ is refused.
- `optimal_gain` integrates all unit initial states at once with the implicit–explicit scheme you
  name (`RK111`, `RK222`, `RK443`), the base-flow terms explicit at the exact stage times and
  $dt = T/n_\text{steps}$ exactly. The gain is maximised over **admissible** initial states (those
  satisfying continuity and the boundary conditions) in the kinetic energy with its exact Gram
  matrix; `g.seed` and `g.response` are the optimal structures.
- `floquet(method="hill", Nh=..., targets=[...], tol=1e-12)` solves the Hill problem of the same
  matrices (exponents nearest each target, residuals reported; `tol`, the Arnoldi stopping
  tolerance, is required); `method="monodromy"` takes the eigenvalues
  of the time-stepped monodromy matrix instead.
- `eigenvalues()` is refused for a time-periodic base: it has no normal modes.
- With real periodic axes (`cn.periodic(L)`), one wavenumber group holds the waves
  $(k_x, \pm k_z)$ together; `complex=True` separates them.

Expert form, with a linear problem per wave that you write yourself:
`conatus.growth.optimal_gain(build, base_harmonics, kx, kz, t0, T, steps, scheme, backend, fields=("u", "v", "w"))`,
where `build({"kx": ..., "kz": ...})` returns the IR problem and `base_harmonics` maps each base-flow
field to `{m: U_m}` (the same input as `hill_from_plan`); `floquet_exponents` takes the same
arguments ({doc}`floquet`).

## What you choose

The norm (through the weights of the system), the resolution, the time window and the tolerance of
the growth search. The threshold below which an eigenvalue counts as
infinite is an argument (`infinite_tol`), reported with the gap that separates the two families.

**See also:** {doc}`state_space <../../api/_generated/_stubs/conatus.expert.state_space>`,
{doc}`max_growth <../../api/_generated/_stubs/conatus.expert.max_growth>`, the gallery case
{doc}`../../gallery/_generated/shear_nonmodal_poiseuille` (Butler & Farrell 1992) and the
mathematics in {doc}`../../math/nonmodal`.
