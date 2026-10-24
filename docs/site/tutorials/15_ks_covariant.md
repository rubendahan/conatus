# Covariant Lyapunov vectors of Kuramoto–Sivashinsky: tangencies and the physical modes

The Lyapunov exponents say how fast perturbations grow. The covariant Lyapunov vectors say in which
directions, and the angles between them say whether the tangent space splits cleanly. For
Kuramoto–Sivashinsky on $L = 22$, the finite-time analysis of this page finds eight entangled
modes, separated from the rest. Ding et al. (2016) call these the "physical" modes. The count is
an empirical result for these trajectory lengths and thresholds; check how it changes with them.

**Goal.** Compute the twelve leading exponents and covariant vectors of KS $L = 22$ from a native
model with `sim.lyapunov`, check their covariance, and count the entangled modes from the
domination of the Oseledets splitting.
**Expected.** Cvitanović, Davidchack & Siminos (2010, §2, p. 12 of arXiv:0709.2944) give the leading exponents; the first ten are $(0.048, 0, 0,
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
the vector there; `explain()` gives the measured residual and its ratio to the rounding scale.

```{figure} media/t15_exponents.png
:alt: Left: twelve Lyapunov exponents with error bars, the first eight between about 0.04 and -0.32, then two near -2 and two near -5.6; the first 10 sit on red ticks of the published values. Right: the first 8 enlarged, with their error bars and reference ticks.
:width: 100%

The twelve leading exponents of Kuramoto-Sivashinsky at L = 22, with their standard errors (left), and the first 8 on a finer scale (right). Red ticks mark the 10 published values (Cvitanović et al. 2010); exponents 11 to 12 have no reference. The first 8 are the entangled modes.
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
:alt: A triangular matrix of probabilities: red squares among the first 8 indices and inside the pairs 9–10 and 11–12; near white across the splits between them.
:width: 100%

Domination of the Oseledets splitting: how often the local exponent i falls below j over windows of 5 time units. The first 8 modes form the leading cluster: no violation crosses the split 8 | 9. Later modes are entangled among themselves in the pairs 9–10 and 11–12, each separated from the leading cluster.
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
