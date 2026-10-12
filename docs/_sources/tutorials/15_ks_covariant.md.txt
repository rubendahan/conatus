# 15. Covariant Lyapunov vectors of Kuramoto–Sivashinsky: tangencies and the physical modes

The Lyapunov exponents say how fast perturbations grow; the covariant Lyapunov vectors say in
which directions, and the angles between them say whether the tangent space splits cleanly. In the
Kuramoto–Sivashinsky equation on $L = 22$ the first eight vectors keep touching each other while
the rest stay apart: eight "physical" modes carry the dynamics.

**Goal.** Compute the twelve leading exponents and covariant vectors of KS $L = 22$ from a native
model with `sim.lyapunov`, check their covariance, and count the entangled modes from the
domination of the Oseledets splitting.
**Expected.** Cvitanović, Davidchack & Siminos (2010, p. 12): $(0.048, 0, 0, -0.003, -0.189,
-0.256, -0.290, -0.310, -1.963, -1.967, \dots)$ in the zero-mean subspace; Ding, Chaté, Cvitanović,
Siminos & Takeuchi (2016): eight physical modes. **Time:** about two minutes on one CPU core.
**Script:** [`t15_ks_covariant.py`](code/t15_ks_covariant.py) · **Tests behind it:**
`tests/chaos/`.

## The model and four starts

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

Every length is a time of the problem: `transient` (trajectory only), `align` (the tangent vectors
align with the Oseledets directions), `T` (averaging), `clv_window` and `clv_backward` (the
covariant vectors are computed over a window that ends 40 time units before the end, the time the
backward iteration needs to converge). Each must be a whole number of steps or of intervals
`reortho_every * dt`; nothing is rounded. `zero_mean="u"` keeps the tangent vectors in the
invariant zero-mean subspace, as in the references. `clv_checkpoints=8` stores 8 checkpoints instead
of every interval and recomputes them once; `explain()` gives the memory used, the difference
between recomputed and stored factors (zero), the **backward convergence** (the vectors obtained
from two independent random starts of the backward iteration, compared: small except inside nearly
degenerate pairs such as the two zero exponents, which converge slowly) and the **covariance
residual**: each vector advanced by the exact tangent over one interval must be parallel to the
vector there, to its rounding floor.

## Tangencies and domination

```{literalinclude} code/t15_ks_covariant.py
:language: python
:start-after: "# [domination]"
:end-before: "# [/domination]"
```

`domination(tau)` compares the local growth rates of the vectors over windows of length $\tau$:
whenever $\Lambda_i < \Lambda_j$ for $i < j$ the splitting is not dominated there. Modes joined by
such violations form clusters; the leading cluster is the set of physical modes. The principal
angles tell the same story: the split $8 \mid 9$ keeps a finite angle, the split $4 \mid 5$ inside the
physical block has near-tangencies.

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

- The same call on the GPU: `m.simulate(..., backend="cuda")`, nothing else changes (not yet
  verified against the CPU on GPU hardware: experimental until `tests/model/test_particles_gpu.py` has run).
