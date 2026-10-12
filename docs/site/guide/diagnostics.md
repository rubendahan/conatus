# Diagnostics and visualization

Eulerian, Lagrangian and mixing diagnostics computed from the fields of a run, on the CPU or the GPU.

`conatus.diagnostics` computes the classical Eulerian, Lagrangian and mixing diagnostics of a flow
from **physical values on a declared grid**, with the same code on NumPy and CuPy. It never touches
a solver: every function reads arrays and returns new ones, so a run with diagnostics is identical
to the bit to the same run without them. Anything that is
not defined by the data alone is refused with the reason, never guessed.

## Global properties during a run

`sim.properties(every=n, name=...)` evaluates global quantities of a running simulation every `n`
steps (the role of Dedalus's `GlobalFlowProperty`): a property is an unknown, a component, or a
function of the simulation; the reductions are `max`, `min`, `grid_average` (mean of the grid
values), `volume_average` and `integral` (quadrature). It never changes the run:

```python
import numpy as np
import conatus as cn

m = cn.problems.native.kuramoto_sivashinsky(L=22.0)
sim = m.simulate(N=32, scheme="RK443", dt=0.01).initial(u=lambda x: 0.1 * np.cos(2 * np.pi * x / 22))
p = sim.properties(every=10, u="u", energy=lambda s: 0.5 * s.grid("u") ** 2)
sim.run(steps=100)
print(p.max("u"), p.volume_average("energy"), p.evaluated_at)   # at iteration 100
print(p.explain())
```

## The grid

```python
import numpy as np
from conatus import diagnostics as dg

g = dg.Grid([dg.Periodic(4 * np.pi, 192),       # uniform nodes, FFT derivative
             dg.Chebyshev(-1.0, 1.0, 129),      # Gauss nodes (ascending), recurrence derivative
             dg.Periodic(2 * np.pi, 160)])
print(g.explain())
# g.on(cupy) is the same grid with CuPy arrays (GPU); the functions below run unchanged on it
```

`Grid.from_bases(domain.bases)` builds it from a Conatus domain (dealiased grid by default, the
grid of `IVPSolver.grid`). Derivatives are exact on the interpolant; for an even number of periodic
nodes the Nyquist mode of an odd derivative is set to zero (declared by `explain()`). Integrals use
the trapezoidal rule on periodic axes and Fejér's first rule on Chebyshev axes.

## Eulerian

| Function | What | Where |
|---|---|---|
| `vorticity(g, u)` | ∂ₓv − ∂ᵧu (2D), ∇×u (3D) | any grid |
| `streamfunction(g, u)` | u = ∂ᵧψ, v = −∂ₓψ, mean flow included | periodic², or periodic × Chebyshev |
| `vector_potential(g, u)` | ∇×A = u − ⟨u⟩, ∇·A = 0 | periodic 3D |
| `poloidal_toroidal(g, u, wall_axis=1)` | φ, ψ and the horizontal means | two periodic axes |
| `q_criterion`, `lambda2`, `helicity`, `enstrophy` | vortex identification and invariants | 3D (Q also 2D) |
| `dissipation(g, u, nu)` | 2ν S:S (ν required) | any grid |
| `pressure(g, u)` | ∇²p = −∂ᵢuⱼ∂ⱼuᵢ, zero mean | periodic only |
| `energy_spectrum`, `spectrum_1d` | Parseval-exact spectra | periodic axes |

`lambda2(..., method="eigvalsh")` (default) uses a batched symmetric eigensolver; the closed-form
cubic (`method="closed_form"`) avoids a copy but is only ≈ √ε accurate near repeated eigenvalues,
enough for an isosurface.

## Lagrangian

```python
import numpy as np
from conatus import diagnostics as dg

g = dg.Grid([dg.Periodic(2 * np.pi, 64), dg.Chebyshev(-1.0, 1.0, 33)])
X, Y = np.meshgrid(g.nodes(0), g.nodes(1), indexing="ij")
u_n = np.stack([np.cos(Y) * np.sin(X), -np.sin(Y) * np.cos(X)])   # (d, *grid shape) at t_n
t_n, t_np1 = 0.0, 0.1
u_np1 = 0.9 * u_n                                                # the next snapshot of the run

interp = dg.Interpolator(g, "lagrange", order=4)     # or "spectral": exact series, cost Np·N^d
vel = dg.GridVelocity(interp)                        # snapshots, linear in time between two
vel.set(u_n, t_n); vel.set(u_np1, t_np1)
wire_points = np.column_stack([np.full(20, 1.0), np.linspace(-0.8, 0.8, 20)])   # (Np, d)
bubbles = dg.Timelines(wire_points, vel, axes=g.axes, emit_every=8, capacity=40,
                       scheme="rk4", boundary="clip", t0=t_n)
dt, nsteps = 2.5e-3, 40                              # t_n -> t_np1
bubbles.advance(dt, nsteps)
lines = bubbles.polylines()                          # (lines, points, d), NaN = not emitted
```

`Particles` (tracers, or inertial with `MaxeyRiley(tau_p, beta=..., g=...)`; the neglected forces
are listed by `explain()`), `PathRecorder` (pathlines), `Streaklines` (smoke wire), `Timelines`
(hydrogen-bubble wire), `ftle` (Cauchy–Green tensor from neighbouring particles) and `Dye` (a
passive scalar with declared diffusivity on a periodic box). A non-finite velocity stops the
particles with an error; bounded axes either clip (counted) or raise.

## Mixing

`variance`, `mix_norm(g, c, s=0.5, weight="homogeneous" | "mmp")` (Mathew, Mezić & Petzold 2005),
`potential_energies(g, b, vertical_axis=...)` (E_p, E_b by sorting and APE = E_p − E_b ≥ 0 exactly,
Winters et al. 1995), `mixing_rate(t, E_b, phi_i=...)` (the boundary flux is required) and
`mixing_efficiency(phi_d, eps, t=...)`.
