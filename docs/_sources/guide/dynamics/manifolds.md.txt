# Invariant manifolds and slices

The unstable manifold of an equilibrium as a polynomial parametrization, connections between solutions, and the reduction of a continuous symmetry by a slice.

These tools treat the reduced coordinates of a solver (its active degrees of freedom, possibly in
an invariant subspace) as a dynamical system. The examples below start from the equilibrium `E1`
and the travelling wave `TW1` of {doc}`invariant_solutions`, computed here for the
Kuramoto–Sivashinsky equation on $L = 22$:

```{literalinclude} _check.py
:language: python
:start-after: "[man-setup]"
:end-before: "[/man-setup]"
:dedent: 4
```

## Unstable manifold

The parametrization method writes the unstable manifold of an equilibrium $x_0$ as
$P(s) = \sum_\alpha a_\alpha s^\alpha$, with $P(0) = x_0$ and $DP(0)$ the unstable eigenvectors,
and solves the homological equations order by order. For a polynomial vector field (its degree is
declared and checked), the method computes the Taylor coefficients of $P$ to the requested order.
The truncated series is an approximation, even for a quadratic field: for $x' = x - x^2$ the
parametrization is $P(s) = s/(1+s)$, an infinite series. The invariance defect measures the error
of the truncation.

```{literalinclude} _check.py
:language: python
:start-after: "[man-unstable]"
:end-before: "[/man-unstable]"
:dedent: 4
```

`max_defect(rho, m)` is the largest residual of the invariance equation at `m` points of the circle
$|s| = \rho$; the example measures it at $\rho = 0.5$. It is an estimate on that circle, not a
bound inside it. `validity_radius` finds the largest radius whose defect stays below a tolerance. `globalize` integrates the points of a fundamental
domain to grow the manifold, and `find_connection` / `scan_connection` look for connections
between two solutions. Poincaré sections (`PoincareSection`, `first_return_map`) locate crossings
on the dense output of a trajectory.

## Symmetry-reduced slices

A continuous translation symmetry makes relative solutions drift. The first-Fourier-mode slice
fixes the phase of the first Fourier mode: travelling waves become equilibria and relative
periodic orbits become periodic orbits.

```{literalinclude} _check.py
:language: python
:start-after: "[man-slice]"
:end-before: "[/man-slice]"
:dedent: 4
```

The routine reports when the first Fourier mode vanishes, because the slice phase is then
undefined (the slice border). `find_rpo_in_slice` searches periodic orbits of the reduced flow.

**See also:** {doc}`unstable_manifold <../../api/_generated/_stubs/conatus.expert.unstable_manifold>`, {doc}`ecs_system <../../api/_generated/_stubs/conatus.expert.ecs_system>`,
{doc}`find_connection <../../api/_generated/_stubs/conatus.expert.find_connection>`, {doc}`FirstFourierModeSlice <../../api/_generated/_stubs/conatus.expert.FirstFourierModeSlice>`.
