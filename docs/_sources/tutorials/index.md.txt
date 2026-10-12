# Tutorials

Tutorial 0 (getting started) walks the whole path once; the shorter ones after it go deeper, from a first eigenvalue problem to a GPU run, keeping a registry of runs, importing a Dedalus script, covariant Lyapunov vectors and Lagrangian tracers. Each one is a
plain Python script, shown piece by piece on its page, that **ends by checking its result against
a published value or a closed form** (an `assert`, with the reference in a comment). The output
printed on each page is the output of that script, executed as it is shown; the table below says
when it last ran.

Prerequisites: {doc}`../install` with the `tutorials` extra (`pip install "conatus[tutorials]"`: matplotlib for
the figures), and some familiarity with spectral methods. Run a tutorial with
`python docs/site/tutorials/code/<script>.py` from a copy of the repository. Tutorial 2b needs a
GPU; the check script of tutorial 14 (14b) needs Dedalus.

| # | Tutorial | You will learn | Reference value |
|---|---|---|---|
| 0 | {doc}`00_getting_started` | the whole path once: a native model, `explain()`, stability, the onset by `critical`, a simulation, the heat flux, the *Methods* paragraph, the run registry | Reid & Harris (1958) |
| 1 | {doc}`01_orr_sommerfeld` | a native model, conditions by wall name, eigenvalues with residuals, critical parameters, `verify` | Orszag (1971) |
| 2 | {doc}`02_rayleigh_benard` | a simulation on CPU, then on GPU; time steppers, CFL, the *Methods* paragraph | linear growth rate, CPU/GPU agreement |
| 3 | {doc}`03_symmetries` | linearizing a native model, symmetry subspaces: the same eigenvalue with half the unknowns | Reid & Harris (1958) |
| 4 | {doc}`04_bratu` | a nonlinear boundary-value model, Newton on the exact Jacobian, pseudo-arclength continuation, folds | Bratu (1914), Gelfand (1963) |
| 5 | {doc}`05_ks_orbits` | equilibria, travelling waves and periodic orbits of a native model by Newton–Krylov–hookstep, Floquet exponents | Cvitanović et al. (2010), Ding & Cvitanović (2016) |
| 8 | {doc}`08_polar` | polar and cylindrical geometry: a drum on `cn.Disk`, ready pipe and Taylor–Couette problems | Bessel zeros, Meseguer & Trefethen (2003), Recktenwald et al. (1993) |
| 10 | {doc}`10_bifurcations` | branch switching, cycles, codimension 2, normal forms, Landau and Ginzburg–Landau | Lorenz (1963), Kuznetsov (2004), Cross & Hohenberg (1993) |
| 11 | {doc}`11_runs` | the run registry on a native simulation: record, certify, query, compare, reproduce bit for bit, cache only on request | heat equation, closed form |
| 12 | {doc}`12_chaos_ensembles` | Lyapunov spectrum, covariant vectors, Kaplan–Yorke, Poincaré map, basins from one ensemble | Sprott (Lorenz), exact divergence |
| 13 | {doc}`13_sensitivity_data` | direct and adjoint global modes, wavemaker, base-flow sensitivity; DMD in the exact norm | Chomaz (2005) closed forms, Towne et al. (2018) |
| 14 | {doc}`14_from_dedalus` | a convection cell as a native model, then read from a Dedalus script: line-by-line report, run, native vs translated; optional check against Dedalus | Dedalus itself (same method, rounding; check script, needs Dedalus) |
| 15 | {doc}`15_ks_covariant` | `sim.lyapunov` with covariant vectors: exponents with errors, covariance residual, tangencies, domination and the physical modes | Cvitanović et al. (2010), Ding et al. (2016) |
| 16 | {doc}`16_particles_dye` | `sim.particles` and `m.dye`: exact velocity at the tracers, observed order in time, a dye whose mass is measured | Taylor–Green and Couette closed forms |

## Execution status

```{include} _generated/status.md
```

Scripts are executed by `docs/site/run_tutorials.py`, on a CPU and, for the GPU parts, on a GPU;
only passing outputs are copied into the pages.

```{toctree}
:hidden:
:maxdepth: 1

00_getting_started
01_orr_sommerfeld
02_rayleigh_benard
03_symmetries
04_bratu
05_ks_orbits
08_polar
10_bifurcations
11_runs
12_chaos_ensembles
13_sensitivity_data
14_from_dedalus
15_ks_covariant
16_particles_dye
```
