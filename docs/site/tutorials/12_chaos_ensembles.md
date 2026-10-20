# 12. Chaos and ensembles: Lyapunov spectrum, covariant vectors, Poincaré map, basins

The tangent of the discrete step is exact, so the Lyapunov spectrum of a discretized flow is the
spectrum of the map actually computed; and because many trajectories share one batch, the
statistics that make an exponent trustworthy (independent trajectories, block errors) cost
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

```{figure} media/t12_lyapunov.png
:alt: Three panels, one per Lyapunov exponent: grey running estimates of 32 trajectories converge with time onto their red mean and the dashed published value.
:width: 100%

Running estimates of the three exponents for the 32 starts of the page (grey), their mean (red) and the published values 0.906, 0 and −14.572 (dashed).
```

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

```{figure} media/t12_lorenz_map.png
:alt: Successive maxima of z plotted against each other: the dots lie on a thin tent-shaped curve above a dotted diagonal.
:width: 100%

The Poincaré section dz/dt = 0 of the page: each maximum of z against the next falls on a one-dimensional cusp map, as in Lorenz's Fig. 4 of 1963.
```

## Basins of attraction from one ensemble

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [basins]"
:end-before: "# [/basins]"
```

`ensemble_fate` asks the classifier every `check_every` steps and latches each member's verdict;
members still undecided at the horizon stay 0 and are reported, never assigned.

```{figure} media/t12_basins.png
:alt: A square map of starting points coloured red or green by the equilibrium they end on, forming interleaved spiral bands, with C+ and C- marked.
:width: 100%

Basins of C+ (red) and C− (green) at r = 20 for the 2304 starts of the ensemble (z₀ = 19); the rare starts still undecided at the horizon are black. The map is symmetric under (x, y) → (−x, −y).
```

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
