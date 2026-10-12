# 12. Chaos and ensembles: Lyapunov spectrum, covariant vectors, Poincaré map, basins

The tangent of the discrete step is exact, so the Lyapunov spectrum of a discretized flow is the
spectrum of the map actually computed; and because many trajectories share one batch, the
statistics that make an exponent trustworthy — independent trajectories, block errors — cost
little more than one trajectory.

**Goal.** Compute the Lyapunov spectrum, the Kaplan–Yorke dimension and the covariant Lyapunov
vectors of the Lorenz attractor from 32 trajectories in one batch; build the Lorenz map from a
Poincaré section located on the scheme; map the basins of the two stable foci at $r = 20$ with
2304 members advanced together.
**Expected.** Sprott (web page *lorenzle.htm*): $(0.906, 0, -14.572)$, $D_{KY} = 2.062$; exact sum
$-(\sigma + 1 + b) = -13.667$ (constant divergence); the neutral covariant vector is the flow direction;
maxima of $z$ on a one-dimensional cusp map (Lorenz 1963, Fig. 4). **Time:** about 1 min on one CPU core.
**Script:** [`t12_chaos_ensembles.py`](code/t12_chaos_ensembles.py) · **Tests behind it:**
`tests/chaos/`, `tests/ensembles/`.

## Write the system

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [system]"
:end-before: "# [/system]"
```

An ODE has no geometry, so the Lorenz system is written with `ode_problem` from the expert layer:
the same kind of problem object a native model compiles to, advanced by the same `IVPSolver`. The
scheme and the time step are explicit choices, as always.

## Lyapunov spectrum from 32 trajectories

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [lyapunov]"
:end-before: "# [/lyapunov]"
```

Every length is an argument: trajectory transient, tangent transient (the vectors align with the
Oseledets directions before anything is averaged), averaging steps, QR interval, number of blocks
for the error, inner product, seed. `explain()` prints each exponent with the standard error of the
block means and the lag-1 correlation of those blocks (if it is large the blocks are too short).
The sum is not exactly $-13.667$: it is the mean $\log|\det J|$ of the RK443 **map**, whose
truncation at $dt = 0.01$ is $7.7\times10^{-3}$ ($9.2\times10^{-4}$ at $dt/2$: third order).

## Covariant vectors

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [clv]"
:end-before: "# [/clv]"
```

Ginelli's backward pass turns the stored $Q$ and $R$ factors into the covariant vectors. The zero
exponent belongs to time translation, so its covariant vector must be parallel to $f(x)$: a property,
not a fit.

## Poincaré section and the Lorenz map

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [poincare]"
:end-before: "# [/poincare]"
```

The section $g = xy - bz = 0$ (a maximum of $z$) is located on the scheme: a sub-step $\tau^*$ of the
same Runge–Kutta method with $g(\Phi_{\tau^*}(X_n)) = 0$ to the declared tolerance. No interpolant is
assumed; a crossing that is not located is refused.

## Basins of attraction from one ensemble

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [basins]"
:end-before: "# [/basins]"
```

`ensemble_fate` asks the classifier every `check_every` steps and latches each member's verdict;
members still undecided at the horizon stay 0 and are reported, never assigned.

## Check against the references

Exponents within four standard errors (plus the RK443 truncation) of $(0.906, 0, -14.572)$,
$D_{KY}$ of $2.062$, sum $-13.667$ to $10^{-2}$; neutral covariant vector parallel to $f$ to $10^{-6}$;
basins exchanged exactly by $(x, y) \to (-x, -y)$.

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t12_chaos_ensembles.txt
:language: text
```

## Going further

- KS $L = 22$: spectrum against Cvitanović, Davidchack & Siminos (2010) and Edson et al. (2019),
  physical/isolated split of the covariant vectors: `benchmarks/j3/lyapunov_run.py`.
- Probability of transition to turbulence in plane Couette flow from GPU ensembles:
  `benchmarks/j3/couette_transition.py`; throughput CPU vs GPU: `benchmarks/j3/ensemble_throughput.py`.
