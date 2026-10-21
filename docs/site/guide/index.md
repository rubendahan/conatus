# User guide

This guide is organised by task. Use it to find the supported arguments, the returned fields and
the failure conditions of each part of Conatus.

The tutorials follow one problem from start to finish. The snippets of this guide are run against
the current version of the library by the test suite.

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
  and the checks Conatus runs.
- {doc}`equations`: unknowns, parameters, known fields and equations as text; which terms are
  implicit.
- {doc}`boundary_conditions`: conditions by boundary name, how Conatus places them, and what it
  refuses.
- {doc}`problems`: canonical flows ready to solve, each with its published reference.
- {doc}`units`: unit systems and study presets; use `convert()` to translate quantities between
  declared unit systems.

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
studio
studio_agents
agents
notebook
```

- {doc}`timesteppers`: `simulate`, the schemes, start-up, choosing `dt`.
- {doc}`resolution`: spectral tails per unknown and direction, and how many modes you need.
- {doc}`gpu`: the same model on a GPU, and what runs there.
- {doc}`diagnostics`: vorticity, vortex criteria, particles, dye and mixing.
- {doc}`particles`: tracers and a dye inside a simulation (`sim.particles`, `m.dye`, `sim.dye`).
- {doc}`io`: snapshots and checkpoints in HDF5, and a run split over several jobs.
- {doc}`studio`: watch a run as a film from a browser or a phone, and control it.
- {doc}`studio_agents`: follow and pilot a run from a script or an AI agent (JSON, tokens, audit).
- {doc}`agents`: what an AI agent reads first (`AGENTS.md`), and `python -m conatus agent`: the
  library's facts and a dry run of a script, in JSON.
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
experimental
```

- {doc}`explain`: `explain()` on models, results and solvers, and a draft *Methods* paragraph.
- {doc}`uncertainty`: a computed number with its error estimate, and the error of a time
  average.
- {doc}`runs`: record, query, compare and reproduce runs.
- {doc}`reproducibility`: the provenance record and what is deterministic.
- {doc}`doctor`: what your installation can run.
- {doc}`errors`: how to read an error message.
- {doc}`experimental`: the experimental APIs, their current limits, and the warning they emit.

## Importing scripts

```{toctree}
:caption: Importing scripts
:maxdepth: 1

dedalus
```

- {doc}`dedalus`: `cn.from_dedalus` reads a Dedalus 3 script without running it and builds the
  corresponding Conatus problem.

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
