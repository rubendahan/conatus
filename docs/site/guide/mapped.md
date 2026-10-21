# Channels between two surfaces

A channel whose walls are not flat: a wavy bottom, a bump, a corrugated lid. The equations are
written in the physical coordinates, exactly as for a flat box; Conatus maps the region between
the two surfaces onto a rectangle and carries the change of variables into every operator.

## A wavy wall

`cn.MappedBox` takes periodic axes and one vertical axis `cn.between(bottom=..., top=...)`. Each
surface is a number (a flat wall) or a function of the periodic coordinates, named after them
(`lambda x: ...` in 2D, `lambda x, y: ...` in 3D). The boundaries are `bottom`, `top` and `walls`.

```{literalinclude} _check_mapped.py
:language: python
:start-after: "# [mapped-stokes]"
:end-before: "# [/mapped-stokes]"
:dedent: 4
```

```{literalinclude} outputs/mapped_stokes.txt
:language: text
```

`sol.coordinate("z")` is a full array: the height of a grid point depends on its horizontal
position. The grid is $x_j$ (uniform) times the Gauss–Chebyshev points $\xi_l$ of the
terrain-following coordinate.

For a small amplitude $\varepsilon$ the flow is $u = z + \varepsilon u_1 + O(\varepsilon^2)$, with
$u_1$ from the stream function $\phi(z)\cos x$, $(\partial_z^2 - 1)^2\phi = 0$, and the no-slip
condition moved from $z = \varepsilon\cos x$ to $z = 0$. The difference with the computed field is
of second order in $\varepsilon$ (it falls by 4.0 each time $\varepsilon$ is halved; this is a test
of the library):

```{literalinclude} _check_mapped.py
:language: python
:start-after: "# [mapped-perturbation]"
:end-before: "# [/mapped-perturbation]"
:dedent: 4
```

## What the map adds

The vertical coordinate is replaced by $\xi \in [-1, 1]$,

$$ z = b(x) + \bigl(t(x) - b(x)\bigr)\,\frac{1 + \xi}{2} $$

(Gal-Chen and Somerville, 1975), and every physical derivative is written by the chain rule,

$$ \partial_z = J\,\partial_\xi, \qquad
   \partial_x\big|_z = \partial_x\big|_\xi + (A + B\,\xi)\,\partial_\xi, \qquad
   J = \frac{2}{t - b},\; A = -J\,\frac{b' + t'}{2},\; B = -J\,\frac{t' - b'}{2}, $$

with the volume element $\tfrac{t - b}{2}\,d\xi\,dx$ for integrals (the pressure gauge
$\int p\,dV = 0$ is over the physical domain). The metric is a polynomial of degree one in $\xi$:
along the vertical the operators stay banded, exactly. Along $x$ the functions $J$, $A$, $B$
multiply, so the Fourier modes are coupled: Conatus builds one sparse system for the whole domain
(all modes, the metric inside the operator) and solves it directly. The metric is not split into
an explicit correction, so there is no stability condition on the wall slope.

```{literalinclude} _check_mapped.py
:language: python
:start-after: "# [mapped-explain]"
:end-before: "# [/mapped-explain]"
:dedent: 4
```

```{literalinclude} outputs/mapped_explain.txt
:language: text
```

The Fourier coefficients of each metric function are computed once on a fine grid. Their tail on
the grid of the unknowns must be below `ncc_tail_tol` (default `1e-8`), like any non-constant
coefficient. The compiler checks this coefficient tail and rejects a wall too steep for the
horizontal resolution, with the resolution that resolves it ({doc}`errors`). The same holds for a known field `m.given(...)`,
which on a mapped domain is a function of $x$ and of the physical $z$.

## Convergence

A manufactured solution: the error falls geometrically with the resolution down to rounding.

```{literalinclude} _check_mapped.py
:language: python
:start-after: "# [mapped-convergence]"
:end-before: "# [/mapped-convergence]"
:dedent: 4
```

```{literalinclude} outputs/mapped_convergence.txt
:language: text
```

## Conditions on a curved wall

A value (`T=...`, `u=0`) is imposed on the surface itself. A derivative in a condition is the
physical one along the coordinate: `dz(T)` at `bottom` is the vertical derivative, not the
derivative along the normal. To impose a flux, write the normal as known fields of $x$:

```{literalinclude} _check_mapped.py
:language: python
:start-after: "# [mapped-neumann]"
:end-before: "# [/mapped-neumann]"
:dedent: 4
```

`cn.free_slip` and `cn.navier_slip` are refused on a curved surface: their operators use the normal
of a flat wall. On a flat surface of a `MappedBox` (`top=1.0`, say) they are accepted and are the
conditions of `cn.Box`.

## Problems, schemes and limits

- `m.solve(N)` (linear boundary-value problems), `m.eigenvalues(N, ...)` (one coupled spectrum
  for all modes) and `m.simulate(N, scheme=..., dt=...)` with `SBDF1`–`SBDF4`, `RK111`, `RK222`,
  `RK443`. The implicit part is the coupled operator; nonlinear terms on the right are evaluated on
  the dealiased physical grid. Boundary data that depend on time are imposed at each stage time
  (`bc_time="stage"`; the default `"auto"` resolves to it and `explain()` says why; `"rk"` and
  `"step"` are refused). RK443 is then of order 2: the same errors as `cn.Box` with
  `bc_time="stage"`, and the same order as its default `"rk"` on this manufactured diffusion test.
- Cost: one sparse LU of the coupled system (all Fourier modes). Measured on one CPU core: 2D
  Stokes at 128 × 32 (12 701 unknowns) about 90 s and 1.8 GB; 3D Poisson at 24 × 24 × 16 (9 522
  unknowns) about 16 s and 1.1 GB. Refused above 100 000 unknowns or 10⁷ nonzeros in the
  operators (`MAX_COUPLED_UNKNOWNS`, `MAX_COUPLED_NNZ` of `conatus.discretize.terrain_plan`,
  declared in `cn.tolerances.explain()`).
- A given field that multiplies an unknown must be resolved on the grid: a tail above
  `ncc_tail_tol` along $\xi$ or in the kept Fourier modes, or energy at the Nyquist wavenumber
  (aliasing), is refused with the fix.
- 3D: `cn.MappedBox(x=cn.periodic(Lx), y=cn.periodic(Ly), z=cn.between(bottom=lambda x, y: ...,
  top=...))`. The coupled system grows as $(N_x N_y)^2$ in the worst case: moderate resolutions.
- With flat surfaces the system is the one of `cn.Box` (identical solutions to rounding, no
  coupling between Fourier modes).
- Refused with a message: traces at other heights, horizontal averages at fixed $z$,
  `periodic(..., complex=True)`, symmetry subspaces, GPU backends.
