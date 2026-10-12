# Changelog

All notable changes to Conatus.

## [0.1.0] - 2026-10-12

First public release. Linux x86-64, CPython 3.11–3.13. BSD-3-Clause.

### What ships

- **Models written as on paper**: `cn.Model` on boxes, channels, disks, annuli, pipes, spheres,
  balls and spherical shells; unknowns, parameters and equations as text; boundary conditions by
  boundary name; every decision (bases, boundary-condition placement, gauge, implicit/explicit
  split, dealiasing) printed by `explain()`.
- **Simulation**: IMEX time steppers (SBDF1–4, Crank–Nicolson family, RK222, RK443) on the CPU,
  and on one NVIDIA GPU (beta, `backend="cuda"`); checks for non-finite states, spectral tails
  and aliased initial data while stepping; self-describing HDF5 outputs and checkpoints.
- **Linear stability**: eigenvalues with residuals and resolution drift, adjoint modes,
  critical parameters (`critical`, `least_stable`), non-modal growth of steady flows.
- **Floquet–Hill** analysis of time-periodic linear systems, with a residual for every exponent.
- **Invariant solutions and continuation**: Newton–Krylov with hookstep, equilibria, travelling
  waves, periodic orbits of one-dimensional models, pseudo-arclength continuation, bifurcation
  detection and branch switching.
- **Dynamics and data**: Lyapunov spectra and covariant Lyapunov vectors, tracers and dye, DMD and
  SPOD.
- **Run registry** (`cn.runs`): record, query and compare runs.
- Documentation: tutorials, user guide, API reference and the mathematics of each method.

### Install

`pip install conatus` (CPU wheel from PyPI, compiled core included). The CUDA 12 wheel (beta,
version `0.1.0+cu12`) is attached to the GitHub release only; install that file with the `[gpu]`
extra (CuPy). From source: `pip install .` with LAPACK/BLAS available.

### Known limitations

- The CPU path is a NumPy reference implementation: slower than established CPU codes on large
  grids. Speed comes from the GPU path, which is in beta (tested on NVIDIA A100 and L40S; the CUDA
  wheel needs a driver supporting CUDA 12.9).
- One CPU process or one GPU per simulation: no multi-GPU or MPI execution.
- NumPy, SciPy and the core each load OpenBLAS: set `OMP_NUM_THREADS` (about half the CPUs) to
  avoid oversubscription; `python -m conatus doctor` prints the value to set.
- The 0.1 API may still change; changes go through a `DeprecationWarning`.

Features under development are listed on the website's
[Coming next](https://rubendahan.github.io/conatus/coming-next/) page.
