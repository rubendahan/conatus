# 16. Lagrangian tracers and a numerical dye

Tracers show where the fluid goes; a dye shows how it mixes. Both are only as good as the velocity
they are given and the way they are advanced, so both are declared: the velocity at a tracer is the
spectral series itself, the time integration has a stated order that the tutorial measures, and the
dye is one more unknown of the model whose mass is written to its file.

**Goal.** Follow tracers in decaying Taylor–Green vortices and measure the order in time against a
path integrated to $10^{-13}$; then shear a dye blob in plane Couette flow with tracers on the walls,
and read both files back with xarray.
**Expected.** Error divided by $2^3$ when $dt$ is halved (unsteady flow; $2^4$ in a steady one);
Couette tracers $x(t) = x_0 + y_0 t$ to rounding; dye mass conserved. **Time:** under a minute.
**Script:** [`t16_particles_dye.py`](code/t16_particles_dye.py) · **Tests behind it:**
`tests/model/test_particles_dye.py`, `tests/model/test_particles_gpu.py`.

## Tracers in an exact Navier–Stokes solution

```{literalinclude} code/t16_particles_dye.py
:language: python
:start-after: "# [taylor_green]"
:end-before: "# [/taylor_green]"
```

`sim.particles(X0)` registers the tracers with the simulation: each `run` step advances them by one
RK4 step. RK4 needs the velocity at the start, the middle and the end of the step; the ends are the
flow states themselves, the middle is interpolated in time through the last four flow states
(cubic, accurate to $O(dt^4)$). The order is 3 here because the flow states are only third-order
accurate in time (RK443), and because the first step of a run has only two states to interpolate
between; in a steady flow neither matters and the order is that of RK4, 4. With the velocity frozen
over a step it would be 1.

## A dye in plane Couette flow

```{literalinclude} code/t16_particles_dye.py
:language: python
:start-after: "# [couette]"
:end-before: "# [/couette]"
```

`m.dye` writes the dye's equation in conservative form and its no-flux wall condition (both shown by
`m.explain()`); the dye is then advanced with the flow, by the same scheme. `sim.dye` writes it with
its mass, variance, extrema and spectral tail; `sim.particles(..., path=, every=)` writes positions
(wrapped and unwrapped), velocities and wall clips, with the dimensions `t`, `particle`,
`component`.

## Check

```{literalinclude} code/t16_particles_dye.py
:language: python
:start-after: "# [checks]"
:end-before: "# [/checks]"
```

## Output (as executed)

```{literalinclude} outputs/t16_particles_dye.txt
:language: text
```

## Going further

- Guide: {doc}`../guide/particles` (walls, files, GPU, what is not covered).
- Streaklines, timelines, inertial particles and FTLE from snapshots: {doc}`../guide/diagnostics`.
