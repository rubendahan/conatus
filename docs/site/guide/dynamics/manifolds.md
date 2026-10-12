# Invariant manifolds and slices

The unstable manifold of an equilibrium as a polynomial parametrization, connections between solutions, and the reduction of a continuous symmetry by a slice.

These tools treat the reduced coordinates of a solver (its active degrees of freedom, possibly in
an invariant subspace) as a dynamical system. The examples below start from the equilibrium `E1`
and the travelling wave `TW1` of {doc}`invariant_solutions`.

## Unstable manifold

The parametrization method writes the unstable manifold of an equilibrium $x_0$ as
$P(s) = \sum_\alpha a_\alpha s^\alpha$, with $P(0) = x_0$ and $DP(0)$ the unstable eigenvectors,
and solves the homological equations order by order. It is exact for a polynomial vector field,
whose degree is declared and checked.

```{literalinclude} _check.py
:language: python
:start-after: "[man-unstable]"
:end-before: "[/man-unstable]"
:dedent: 4
```

`max_defect` measures the invariance equation on a circle of radius $\rho$; `validity_radius`
finds the largest radius below a tolerance. `globalize` integrates the points of a fundamental
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

The slice border (where the first mode vanishes and its phase is undefined) is detected and
reported, never regularized. `find_rpo_in_slice` searches periodic orbits of the reduced flow.

**See also:** {doc}`unstable_manifold <../../api/_generated/_stubs/conatus.expert.unstable_manifold>`, {doc}`ecs_system <../../api/_generated/_stubs/conatus.expert.ecs_system>`,
{doc}`find_connection <../../api/_generated/_stubs/conatus.expert.find_connection>`, {doc}`FirstFourierModeSlice <../../api/_generated/_stubs/conatus.expert.FirstFourierModeSlice>`.
