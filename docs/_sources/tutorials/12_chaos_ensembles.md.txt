# Chaos and ensembles: the Lyapunov spectrum of Lorenz

**Goal.** Compute the Lyapunov spectrum and the Kaplan–Yorke dimension of the Lorenz attractor,
with error estimates, from 32 trajectories advanced in one batch.
**Expected.** Sprott (web page *lorenzle.htm*): $(0.906, 0, -14.572)$, $D_{KY} = 2.062$; exact sum
$-(\sigma + 1 + b) = -13.667$ for the flow (constant divergence).
**Extensions.** Three later sections reuse the same batch; each can be read alone. The covariant
vectors (for the flow, the neutral one is the flow direction $f(x)$; the RK443 map approaches it
as $dt \to 0$). The Lorenz map from a Poincaré section: maxima of $z$ on a one-dimensional cusp
map (Lorenz 1963, Fig. 4). The basins of the two stable foci at $r = 20$, from 2304 members
advanced together.
**Time:** about 1 min on one CPU core.
**Script:** [`t12_chaos_ensembles.py`](code/t12_chaos_ensembles.py) · **Tests behind it:**
`tests/chaos/`, `tests/ensembles/`.

The tangent used here is the derivative of the discrete step, so the spectrum is that of the map
actually computed. Many trajectories share one batch, so the error estimates (independent
trajectories, block errors) cost little more than one trajectory.

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
scheme and the time step are arguments of the call.

## Lyapunov spectrum from 32 trajectories

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [lyapunov]"
:end-before: "# [/lyapunov]"
```

You supply the durations, in simulation time units or steps: the trajectory transient, the
tangent transient (the vectors align with the Oseledets directions before anything is averaged)
and the averaging steps. The QR interval, the number of blocks for the error, the inner product and
the seed are arguments too. `explain()` prints each exponent with the standard error of the
block means and the lag-1 correlation of those blocks (if it is large the blocks are too short).
The sum is not exactly $-13.667$: it is the mean $\log|\det J|$ of the RK443 **map**, whose
truncation at $dt = 0.01$ is $7.7\times10^{-3}$ ($9.2\times10^{-4}$ at $dt/2$: third order).

```{figure} media/t12_lyapunov.png
:alt: Three panels, one per Lyapunov exponent: a grey band of running estimates over the starts narrows with time around a red mean that settles on the dashed published value.
:width: 100%

Finite-time estimates of the three exponents, averaged from the start of the measurement, over the 32 starts of the page: the band holds the middle 80 % of the starts, the red line is their mean and the dashed lines are the published values 0.906, 0 and −14.572. The panel titles give the final mean and its standard error over the starts. Time is in the units of the Lorenz equations.
```

## Extension: covariant vectors

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [clv]"
:end-before: "# [/clv]"
```

Ginelli's backward pass turns the stored $Q$ and $R$ factors into the covariant vectors. For the
continuous flow, the zero exponent belongs to time translation and its covariant vector is parallel
to $f(x)$. The time-stepping map satisfies this only up to its time-discretisation error. For
example, one explicit Euler step on $x' = x^2$ gives $D\Phi_h(x)\,f(x) - f(\Phi_h(x)) = -h^2 x^4$.
The check below therefore measures the angle with $f(x)$ against a tolerance.

## Extension: Poincaré section and the Lorenz map

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

## Extension: basins of attraction from one ensemble

```{literalinclude} code/t12_chaos_ensembles.py
:language: python
:start-after: "# [basins]"
:end-before: "# [/basins]"
```

`ensemble_fate` asks the classifier every `check_every` steps and latches each member's verdict.
Members undecided at the final time keep classification 0, and the result counts them.

```{figure} media/t12_basins.png
:alt: A square map of starting points coloured red or blue by the equilibrium they end on, forming interleaved spiral bands, with C+ and C- marked and a legend below.
:width: 100%

Basins of C+ (red) and C− (blue) at r = 20 for the 2304 starts of the ensemble (z₀ = 19); the rare starts still undecided at the horizon are black. The map is symmetric under (x, y) → (−x, −y).
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
