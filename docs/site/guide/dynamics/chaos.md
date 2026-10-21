# Chaos and ensembles

Lyapunov exponents with error bars, covariant Lyapunov vectors, Poincaré maps, and the fate of many trajectories integrated as one batch.

## Lyapunov spectrum

The exponents are computed from the tangent dynamics of the same discrete map you integrate,
over several independent trajectories, with a standard error from block averages.

```{literalinclude} _check.py
:language: python
:start-after: "[ly-system]"
:end-before: "[/ly-system]"
:dedent: 4
```

```{literalinclude} _check.py
:language: python
:start-after: "[ly-spectrum]"
:end-before: "[/ly-spectrum]"
:dedent: 4
```

`inner=` declares the inner product of the Gram–Schmidt steps; `res.kaplan_yorke()` returns the
Kaplan–Yorke dimension with its error. With `clv_window=...` the result also carries the covariant
Lyapunov vectors. `poincare_section` records the crossings of a surface $g(x) = 0$, located to a
declared tolerance, and their return map.

## From a native simulation

`sim.lyapunov` computes the same spectrum, and the covariant vectors, from a native model; lengths are
durations of the problem (each a whole number of steps or of intervals `reortho_every * dt`):

```{literalinclude} _check.py
:language: python
:start-after: "[ly-native]"
:end-before: "[/ly-native]"
:dedent: 4
```

- `zero_mean="u"` keeps the tangent vectors in the invariant zero-mean subspace (all axes periodic);
  `project=f` declares another invariant subspace. `starts=[{...}, ...]` runs several trajectories in
  one batch (each dict as for `sim.initial`); the simulation's own state is not advanced.
- `covariant=True` computes the covariant Lyapunov vectors (Ginelli et al. 2007) over a window of
  length `clv_window` that ends `clv_backward` before the end of the run. `clv_checkpoints=s` stores
  s checkpoints instead of every interval and recomputes them once (memory and recomputation check in
  `explain()`).
- `explain()` also gives the **backward convergence** (the covariant vectors from two independent
  random starts of the backward iteration, compared: the error left by `clv_backward`), the
  **covariance residual** (each vector advanced by the exact tangent over one interval is parallel to
  the vector there, to its rounding floor) and the smallest principal angle at every split.
- `res.angles(j)`, `res.pair_angle(i, j)`: angles at every kept time; `res.clv_grid("u", i)`: a
  covariant vector as grid values.
- `res.domination(tau)`: the fraction of windows of length $\tau$ where the local growth rates of
  $\gamma_i$ and $\gamma_j$ ($i < j$) are in the wrong order; modes joined by such violations form
  clusters, the leading one is the set of physical modes (`dom.n_physical`; eight for KS $L = 22$,
  tutorial {doc}`../../tutorials/15_ks_covariant`).

## Ensembles

An ensemble integrates many members of one model as a single batched state, on CPU or GPU.
`ensemble_fate` classifies each member with your criterion. It stops when all members are decided
or the horizon is reached. At the final time, unresolved members remain marked undecided.

```{literalinclude} _check.py
:language: python
:start-after: "[ly-ensemble]"
:end-before: "[/ly-ensemble]"
:dedent: 4
```

The proportions come with confidence intervals (`proportion_ci`).

**See also:** tutorial {doc}`../../tutorials/12_chaos_ensembles` (Lorenz attractor, Sprott's
exponents); `conatus.chaos.lyapunov_spectrum`, `conatus.chaos.poincare_section`,
`Ensemble`, `ensemble_fate`.
