# Particles and dye of a simulation

Tracer particles and a numerical dye that live inside a native simulation: they advance with each
step of `sim.run`, on the CPU or the GPU, and write files in the `cn.io` layout.

## Tracers

```{literalinclude} _check_particles.py
:language: python
:start-after: "# [tracers]"
:end-before: "# [/tracers]"
:dedent: 4
```

What is computed, and declared by `explain()`:

- **Velocity at a tracer**: the velocity is evaluated from all retained spectral coefficients of
  the state at each tracer position (cos/sin along periodic axes, Chebyshev $T_n$ across the
  walls). The cost is $P \times$ the number of modes per evaluation (four per step), in chunks of
  particles.
- **Time integration**: one classical RK4 step per flow step. Its end velocities are the flow states
  before and after the step; the midpoint velocity is interpolated in time through the last four flow
  states. The order is 4 in a steady flow. In an unsteady flow it is at most the order of the flow
  scheme, whose time error the flow states carry (3 with RK443). The first step of each `run`
  interpolates linearly between two states (also third order).
- **Periodic axes**: positions are kept unwrapped (`positions(unwrapped=True)`, for dispersion) and
  also given wrapped into $[a, a + L)$.
- **Walls**: a step that ends outside the walls raises a `NumericalError` naming the particle
  (`at_walls="refuse"`, the default) or puts it back on the wall and counts it
  (`at_walls="clip"`, `wall_clips()`, in the file and in `explain()`). A tracer on a no-slip wall
  stays on it; a tracer on a moving wall moves with it. RK4 stage positions are not clipped.
- **File** (`path=`, `every=`): `position`, `position_unwrapped`, `velocity` with dimensions
  `(t, particle, component)` and `wall_clips` with `(t, particle)`; `cn.io.read(path).to_xarray()`.
- **Not covered** (refused with the reason): curved and mapped geometries, sine/cosine walls,
  half lines and lines, compound intervals, complex-valued problems (`periodic(L, complex=True)`). Positions are not in the
  simulation's checkpoints: after a restart, declare the tracers again (from the last row of the
  file to continue them).

## Dye

```{literalinclude} _check_particles.py
:language: python
:start-after: "# [dye]"
:end-before: "# [/dye]"
:dedent: 4
```

`m.dye("c", velocity="u", kappa=...)` declares one more unknown and its equation in conservative
form, $\partial_t c - \kappa \nabla^2 c = -\nabla\cdot(c\,\boldsymbol u)$, with the no-flux condition
$\partial_n c = 0$ on the walls when $\kappa > 0$ (both printed by `m.explain()`). The dye is advanced
by the simulation's own scheme: diffusion implicit, advection explicit, the same dealiasing and the
same time step, on the CPU or the GPU. `kappa` is a number or the name of a declared parameter.
With `kappa=0`, the dye equation contains only conservative advection, and no wall condition is
imposed. The dye's spectral tail then grows as its filaments thin, and the file records it.

`sim.dye(name, path, every=)` writes the dye (grid values; `field=False` for the diagnostics only)
and, at each row: `mass` ($\int c\,dV$ of the series), `variance` ($\int (c - \bar c)^2\,dV$),
`minimum` and `maximum` of the grid values (under- and overshoots of an under-resolved dye), and
`tail` (largest relative coefficient among the last two modes of each axis). Mass is conserved by
the equation; what the discretization keeps is measured: to rounding in a periodic box, to the
level of the tau correction of the no-flux condition with walls (reported in the validation table of
the paper).

## On the GPU

`m.simulate(..., backend="cuda")`: the tracers' positions and the evaluation of the series stay on
the device; `positions()` and the files copy to the host. The test
`tests/model/test_particles_gpu.py` compares GPU and CPU runs. **Not yet verified on GPU hardware:**
that test has not run yet, so treat the GPU path of tracers and dye as experimental until it has.

## Two interfaces

| | `sim.particles`, `sim.dye` (this page) | `cn.diagnostics.Particles`, `Dye` ({doc}`diagnostics`) |
|---|---|---|
| Velocity | the running simulation, its spectral series | any callable, or grid snapshots with a declared interpolation |
| Time step | one RK4 step per flow step, past flow states for the midpoint | the caller's `dt`, RK4 or RK2 |
| Walls | `at_walls="refuse"` or `"clip"` (counted) | `boundary="raise"` or `"clip"` (counted) |
| Output | HDF5 file (`cn.io.read`) and `positions()` | arrays in memory |

Use `sim.particles` while a native simulation runs; use the diagnostics module for a velocity you
own or for data after a run.

**See also:** {doc}`diagnostics` (tracers from snapshots or analytic velocities, inertial
particles, streaklines, timelines, FTLE, mixing), tutorial {doc}`../tutorials/16_particles_dye`.
