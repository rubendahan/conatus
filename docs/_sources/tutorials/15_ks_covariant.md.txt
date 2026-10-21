# Covariant Lyapunov vectors of Kuramoto–Sivashinsky: tangencies and the physical modes

The Lyapunov exponents say how fast perturbations grow. The covariant Lyapunov vectors say in which
directions, and the angles between them say whether the tangent space splits cleanly. For
Kuramoto–Sivashinsky on $L = 22$, the finite-time analysis of this page finds eight entangled
modes, separated from the rest. Ding et al. (2016) call these the "physical" modes. The count is
an empirical result for these trajectory lengths and thresholds; check how it changes with them.

**Goal.** Compute the twelve leading exponents and covariant vectors of KS $L = 22$ from a native
model with `sim.lyapunov`, check their covariance, and count the entangled modes from the
domination of the Oseledets splitting.
**Expected.** Cvitanović, Davidchack & Siminos (2010, p. 12) tabulate ten values, $(0.048, 0, 0,
-0.003, -0.189, -0.256, -0.290, -0.310, -1.963, -1.967)$, in the zero-mean subspace; the page
compares its first ten estimates with them; Ding, Chaté, Cvitanović,
Siminos & Takeuchi (2016): eight physical modes. **Time:** about two minutes on one CPU core.
**Script:** [`t15_ks_covariant.py`](code/t15_ks_covariant.py) · **Tests behind it:**
`tests/chaos/`.

## The model and four starts

```{literalinclude} code/t15_ks_covariant.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t15_ks_covariant.py
:language: python
:start-after: "# [model]"
:end-before: "# [/model]"
```

The starts are `sim.initial` keyword dicts: each becomes one trajectory of the same batch, and the
simulation's own state is left as it was.

## Exponents and covariant vectors

```{literalinclude} code/t15_ks_covariant.py
:language: python
:start-after: "# [lyapunov]"
:end-before: "# [/lyapunov]"
```

The durations are times of the problem: `transient` (trajectory only), `align` (the tangent vectors
align with the Oseledets directions), `T` (averaging), `clv_window` and `clv_backward` (the
covariant vectors are computed over a window that ends 40 time units before the end, the time the
backward iteration needs to converge). Each duration must be a whole number of steps and of
intervals `reortho_every * dt`; otherwise the call is refused. `zero_mean="u"` keeps the tangent vectors in the
invariant zero-mean subspace, as in the references. `clv_checkpoints=8` stores 8 checkpoints instead
of every interval and recomputes them once; `explain()` gives the memory used, the difference
between recomputed and stored factors (zero), the **backward convergence** (the vectors obtained
from two independent random starts of the backward iteration, compared: small except inside nearly
degenerate pairs such as the two zero exponents, which converge slowly) and the **covariance
residual**: each vector advanced by the tangent of the step over one interval must be parallel to
the vector there, to its rounding floor.

```{figure} media/t15_exponents.png
:alt: Twelve Lyapunov exponents with error bars, close to zero for the first eight and dropping near -2 after; the first 10 sit on red ticks of the published values.
:width: 100%

The twelve leading exponents of Kuramoto-Sivashinsky at L = 22, with their standard errors. Red ticks mark the 10 published values (Cvitanović et al. 2010), so exponents 11 to 12 have no reference tick. The first eight are the entangled modes.
```

## Tangencies and domination

```{literalinclude} code/t15_ks_covariant.py
:language: python
:start-after: "# [domination]"
:end-before: "# [/domination]"
```

`domination(tau)` compares the local growth rates of the vectors over windows of length $\tau$:
when $\Lambda_i < \Lambda_j$ for $i < j$, the splitting is not dominated there. Modes joined by
such violations form clusters; the leading cluster is the set of physical modes. The principal
angles tell the same story: the split $8 \mid 9$ keeps a finite angle, the split $4 \mid 5$ inside the
physical block has near-tangencies.

```{figure} media/t15_angles.png
:alt: Two histograms of angles on a log axis: the filled red one, inside the entangled modes, reaches very small angles; the blue outline, at the split after eight modes, stays well away from zero.
:width: 100%

Angles between the span of the first j covariant vectors and the rest, at each sample. Inside the entangled modes (j = 4) near-tangencies are frequent; at j = 8 the smallest angle stays away from zero over this run.
```

```{figure} media/t15_domination.png
:alt: A triangular matrix of probabilities: red squares among the first eight indices, near white elsewhere.
:width: 100%

Domination of Oseledets splitting: how often the local exponent i falls below j over windows of 5 time units. Violations stay among the first eight modes: they form one entangled block.
```

## Check

```{literalinclude} code/t15_ks_covariant.py
:language: python
:start-after: "# [checks]"
:end-before: "# [/checks]"
```

## Output (as executed)

```{literalinclude} outputs/t15_ks_covariant.txt
:language: text
```

## Going further

- The same call on the GPU: `m.simulate(..., backend="cuda")`, with no other change (not yet
  verified against the CPU on GPU hardware: experimental until `tests/model/test_particles_gpu.py` has run).
