# Tutorials

Each tutorial is a plain Python script, shown piece by piece on its page. It ends by comparing its
result with a published value or a closed form (an `assert`, with the reference in a comment). The
output printed on each page is the output of that script; the table at the end says when each
script last ran and whether it passed. The tutorials are grouped by topic; within *Start here*, the
order matters.

Prerequisites: {doc}`../install` with the `tutorials` extra (`pip install "conatus[tutorials]"`: matplotlib for
the figures, xarray for the tracers tutorial), and some familiarity with spectral methods. Run a tutorial with
`python docs/site/tutorials/code/<script>.py` from a copy of the repository. The GPU part of the
Rayleigh–Bénard tutorial needs a CUDA device; the optional check of the Dedalus-import tutorial needs Dedalus.

## Start here

In this order: a first simulation, a first stability calculation, then one problem from its onset to a simulation.

| Tutorial | You will learn | Reference value |
|---|---|---|
| {doc}`first_steps` | ten minutes: one equation written as on paper, a simulation, a plot, one physical number, four guided changes | Cole (1951), Hopf (1950) |
| {doc}`01_orr_sommerfeld` | a native model, conditions by wall name, eigenvalues with residuals, critical parameters, `verify` | Orszag (1971) |
| {doc}`00_getting_started` | the whole path once: a native model, `explain()`, stability, the onset by `critical`, a simulation, the heat flux, the *Methods* paragraph, the run registry | Reid & Harris (1958) |

## Simulation

| Tutorial | You will learn | Reference value |
|---|---|---|
| {doc}`02_rayleigh_benard` | a simulation on CPU, then on GPU; time steppers, CFL, the *Methods* paragraph | linear growth rate, CPU/GPU agreement |
| {doc}`08_polar` | polar geometry: the modes of a drum on `cn.Disk`; extensions: ready pipe and Taylor–Couette problems | Bessel zeros, Meseguer & Trefethen (2003), Recktenwald et al. (1993) |
| {doc}`16_particles_dye` | Tracers with `sim.particles`, a dye declared by `m.dye` and recorded by `sim.dye`: spectral velocity at the tracers, observed order in time, a dye whose mass is measured | Taylor–Green and Couette closed forms |

## Stability and sensitivity

| Tutorial | You will learn | Reference value |
|---|---|---|
| {doc}`03_symmetries` | linearizing a native model, symmetry subspaces: the same eigenvalue with half the unknowns | Reid & Harris (1958) |
| {doc}`13_sensitivity_data` | direct and adjoint global modes, wavemaker, base-flow sensitivity; DMD in the exact norm | Chomaz (2005) closed forms, Towne et al. (2018) |

## Nonlinear dynamics

| Tutorial | You will learn | Reference value |
|---|---|---|
| {doc}`04_bratu` | a nonlinear boundary-value model, Newton on the exact Jacobian, pseudo-arclength continuation, folds | Bratu (1914), Gelfand (1963) |
| {doc}`05_ks_orbits` | equilibria, travelling waves and periodic orbits of a native model by Newton–Krylov–hookstep, Floquet exponents | Cvitanović et al. (2010), Ding & Cvitanović (2016) |
| {doc}`10_bifurcations` | the Lorenz equilibria through a pitchfork and a Hopf point; extensions: cycles, codimension 2, Landau and Ginzburg–Landau | Lorenz (1963), Kuznetsov (2004), Cross & Hohenberg (1993) |
| {doc}`12_chaos_ensembles` | the Lyapunov spectrum and Kaplan–Yorke dimension of Lorenz; extensions: covariant vectors, Poincaré map, basins | Sprott (Lorenz), exact divergence |
| {doc}`15_ks_covariant` | `sim.lyapunov` with covariant vectors: exponents with errors, covariance residual, tangencies, domination and the physical modes | Cvitanović et al. (2010), Ding et al. (2016) |

## Workflow

| Tutorial | You will learn | Reference value |
|---|---|---|
| {doc}`11_runs` | the run registry on a native simulation: record, certify, query, compare, reproduce bit for bit, cache only on request | heat equation, closed form |
| {doc}`14_from_dedalus` | a convection cell as a native model, then imported from a Dedalus 3 script: line-by-line report, run, native vs translated | the original script (optional check, needs Dedalus) |

## Execution status

```{include} _generated/status.md
```

Scripts are executed by `docs/site/run_tutorials.py`, on a CPU and, for the GPU parts, on a GPU;
only passing outputs are copied into the pages.

```{toctree}
:hidden:
:maxdepth: 1

first_steps
01_orr_sommerfeld
00_getting_started
02_rayleigh_benard
08_polar
16_particles_dye
03_symmetries
13_sensitivity_data
04_bratu
05_ks_orbits
10_bifurcations
12_chaos_ensembles
15_ks_covariant
11_runs
14_from_dedalus
```
