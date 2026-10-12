# User guide

Each part of Conatus described once, with the choices you make and the ones Conatus will not make
for you.

The tutorials follow one problem from start to finish; this guide is organised by task. Every
snippet runs against the current version of the library.

## Writing a model

```{toctree}
:caption: Writing a model
:maxdepth: 1

geometries
shells
mapped
equations
boundary_conditions
problems
units
```

- {doc}`geometries`: boxes, channels, disks, annuli, pipes, spheres, balls and shells, with their
  named boundaries.
- {doc}`mapped`: channels between two surfaces (wavy walls, bumps): the map, the chain rule,
  convergence and limits.
- {doc}`shells`: spherical shells: simulation on a CPU or a GPU, rotating convection, adjoints,
  and what Conatus checks for you.
- {doc}`equations`: unknowns, parameters, known fields and equations as text; which terms are
  implicit.
- {doc}`boundary_conditions`: conditions by boundary name, how Conatus places them, and what it
  refuses.
- {doc}`problems`: canonical flows ready to solve, each with its published reference.
- {doc}`units`: unit systems and study presets; every conversion is a call.

## Running

```{toctree}
:caption: Running
:maxdepth: 1

timesteppers
resolution
gpu
diagnostics
particles
io
notebook
```

- {doc}`timesteppers`: `simulate`, the schemes, start-up, choosing `dt`.
- {doc}`resolution`: spectral tails per unknown and direction, and how many modes you need.
- {doc}`gpu`: the same model on a GPU, and what runs there.
- {doc}`diagnostics`: vorticity, vortex criteria, particles, dye and mixing.
- {doc}`particles`: tracers and a dye inside a simulation (`sim.particles`, `m.dye`, `sim.dye`).
- {doc}`io`: snapshots and checkpoints in HDF5, and a run split over several jobs.
- {doc}`notebook`: rich display in Jupyter.

## Dynamics

```{toctree}
:caption: Dynamics
:maxdepth: 1

dynamics/index
```

- {doc}`dynamics/index`: eigenvalues, non-modal growth, invariant solutions, continuation and
  bifurcations, Floquet and Lyapunov analysis, adjoints, symmetry subspaces.

## Trust and provenance

```{toctree}
:caption: Trust and provenance
:maxdepth: 1

explain
uncertainty
runs
reproducibility
doctor
errors
```

- {doc}`explain`: `explain()` on every key object, and the *Methods* paragraph of your paper.
- {doc}`uncertainty`: a computed number with its error estimate, and the error of a time
  average.
- {doc}`runs`: record, query, compare and reproduce runs.
- {doc}`reproducibility`: the provenance record and what is deterministic.
- {doc}`doctor`: what your installation can run.
- {doc}`errors`: how to read an error message.

## Coming from Dedalus

```{toctree}
:caption: Coming from Dedalus
:maxdepth: 1

dedalus
```

- {doc}`dedalus`: the same problem side by side, what you stop writing, and importing a Dedalus
  script.

## Expert layer

```{toctree}
:caption: Expert layer
:maxdepth: 1

expert
operators
bases
taus
```

- {doc}`expert`: the explicit problem under every model, and when to write it yourself.
- {doc}`operators`: fields, operators, problem classes, linearisation and adjoints.
- {doc}`bases`: bases, dealiasing, curvilinear domains and symmetry subspaces.
- {doc}`taus`: tau terms written by hand, and the rank report.
