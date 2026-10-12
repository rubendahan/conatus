# Conatus

Conatus solves partial differential equations with spectral methods, on CPUs and GPUs, from
equations written as on paper, and studies the dynamical systems they define.

**Status: 0.1.0.** CPU supported, GPU (CUDA 12) in beta. Install with `pip install conatus` ({doc}`install`), then
start with {doc}`tutorials/00_getting_started`.

```python
import numpy as np
import conatus as cn

box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
m = cn.Model(box, name="Rayleigh-Benard 2D", Ra=2500, Pr=1)
u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
m.equations("""
    dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
    div(u) = 0
    dt(T) - lap(T) = -u·grad(T)
""")
m.at("bottom", u=0, T=+0.5)
m.at("top", u=0, T=-0.5)

sim = m.simulate(N=(64, 32), scheme="RK443", dt=5e-4)     # backend="cuda" for a GPU
sim.initial(T=lambda x, z: -z + 1e-3*np.cos(np.pi*x)*np.cos(np.pi*z))
sim.run(until=1.0)

modes = m.linearize(about={u: 0, T: lambda z: -z}).eigenvalues(N=32, k=np.pi)
print(modes.explain())     # growth rates, with residuals
```

You write the geometry, the unknowns, the equations and the boundary conditions. Conatus chooses
the spectral bases of the geometry, places each boundary condition, splits the terms into
implicit and explicit parts, and prints every one of these decisions with `explain()`. The
numerical choices that are yours (resolution, time stepper, time step, backend) are written in the
call; nothing is changed behind your back.

The same model gives a simulation, its linear stability, a Newton search for a steady state, a
travelling wave or a periodic orbit, a continuation through folds and bifurcations, a Floquet or
Lyapunov analysis, an adjoint or an optimal perturbation.

## Documentation

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Installation
:link: install
:link-type: doc
`pip install conatus` (CPU), the CUDA 12 wheel (beta) or from source; check what your machine can run.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Executed tutorials, from getting started to bifurcations, chaos and Lagrangian tracers, each ending on a result from the literature.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc
Geometries, equations, boundary conditions, simulations, GPU, dynamics, provenance.
:::

:::{grid-item-card} Coming from Dedalus
:link: guide/dedalus
:link-type: doc
The same problem side by side, and what you stop writing.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Every public object, from its docstring.
:::

:::{grid-item-card} Mathematics
:link: math/index
:link-type: doc
The methods behind each part of the library.
:::
::::

## Principles

- **No silent substitution.** Basis, resolution, boundary treatment, dealiasing, time stepper,
  tolerances and inner product are written by you or printed by `explain()`; Conatus suggests,
  it never modifies ({doc}`principles`).
- **Validation by residual.** Every eigenvalue, steady state or orbit comes with its residual and
  a resolution check, never as a bare number.
- **Every capability tied to a published value.** Each method is tested against a published
  table or a closed form, and each tutorial ends on one.

## Who it is for

Researchers and students in fluid dynamics and applied mathematics who study stability,
transition, pattern formation and invariant solutions, and who want to go from an equation to a
validated number, then to a GPU simulation, with the same script.

```{toctree}
:hidden:
:caption: Getting started

install
tutorials/index
```

```{toctree}
:hidden:
:caption: Using Conatus

guide/index
gallery/index
api/index
```

```{toctree}
:hidden:
:caption: About

principles
math/index
citing
```

---

*Unaquæque res, quantum in se est, in suo esse perseverare conatur*: each thing, as far as it is in
itself, strives to persevere in its being (Spinoza, *Ethica* III, 6). The origin of the name.
