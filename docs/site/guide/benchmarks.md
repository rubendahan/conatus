# Benchmarks

The benchmark suite is a fixed set of problems that covers the main code paths. Each case checks its
answer against an oracle that does not come from Conatus. A timing is only reported for a run whose
accuracy check passed in the same process. Speed figures for Conatus, on this site or in the paper,
come from this suite.

```bash
python examples/benchmarks/suite.py --threads 1 8            # all cases, default sizes
python examples/benchmarks/suite.py --cases ivp2d_rb --size large --threads 1 2 4 8
```

## Cases

| Case | Path exercised | Default size | Oracle | Tolerance |
|---|---|---|---|---|
| `ivp1d_burgers` | IVP, 1D, RealFourier, nonlinear | N = 64, RK443, dt = 1e-3, 1000 steps | Cole–Hopf closed form | 1e-8 (max over the grid) |
| `ivp2d_taylor_green` | IVP, 2D, RealFourier², Navier–Stokes with pressure | 64 × 64, RK443, dt = 1e-2, 200 steps | exact decaying Taylor–Green vortex | 1e-9 |
| `ivp2d_rb` | IVP, 2D, RealFourier × Chebyshev, Boussinesq | 64 × 32, RK443, dt = 5e-4, 1600 steps | growth rate 5.5425059 of the k = π roll at Ra = 2500 (eigenvalue problem of the same equations, also found in the streamfunction formulation) | 1e-5 |
| `ivp3d_rb` | IVP, 3D, RealFourier² × Chebyshev | 16 × 16 × 24, as above | the same growth rate (isotropy) | 1e-5 |
| `lbvp_poisson` | LBVP, RealFourier × Chebyshev | 64 × 64, 5 solves | manufactured closed form | 1e-10 |
| `evp_dense` | EVP, dense QZ, drift filter | N = 128 and 192 | Orszag (1971): c = 0.23752649 + 0.00373967i | half a unit of the 8th decimal per part |
| `evp_shift_invert` | EVP, sparse shift-invert | N = 512, one mode near 0.24 | the same value | as above |
| `continuation_bratu` | Newton and pseudo-arclength continuation | N = 64, 6 points | fold λ_c from a tanh a = 1 | 1e-10 |
| `floquet_diffusion` | Floquet–Hill exponents, dense | N = 32, 12 harmonics | closed form of a separable problem | 1e-10 |

The tolerances are written in `examples/benchmarks/cases.py` from the expected discretisation error,
before any run. A case whose error exceeds its tolerance is reported as FAIL and its timing is not
used. The eigenvalue cases also require a backward residual below 1e-8.

## What is measured

- **Wall time** of the whole case process, imports included, and its phases: `setup_s` (model,
  discretisation, factorisations, the first steps) and `run_s` (the measured part).
- **Time per step** for initial-value cases: the median over blocks of 10 steps, each block one
  `sim.run(steps=10)` call, so the per-call cost of `run` is included as a user pays it. The
  minimum and the mean over the run are recorded too.
- **Peak resident memory** of the case process (`ru_maxrss`), and the resident memory right after
  importing Conatus, so that the case's own share can be read off.
- **Threads**: the runner sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to
  the requested count before the child process starts, and records `CN_FFT_WORKERS` and
  `CONATUS_BUILD_WORKERS` as they are. The BLAS thread pools seen by the process are recorded when
  `threadpoolctl` is installed.
- **Hardware and versions**: CPU model, host, CPUs allowed to the process, Slurm partition and job;
  Conatus version and Git commit (with a flag for uncommitted changes), Python, NumPy, SciPy, and the
  hash of the sources of the compiled core.

Each case runs in a fresh process, one per (case, thread count, repetition). The records are JSON
files (`kind = "conatus_suite"`) and a summary table, `suite_summary.md`.

## How to read the numbers

A shared node gives noisy times. The suite records the minimum next to the median, and a
comparison between two versions is made on the same node type, at the same thread count, with each
case's accuracy check passing in both runs. A speed-up is stated with the two run identifiers.

## Threads

The implicit solves of a time step are batches of independent banded systems, one per Fourier
group. With more than one thread, the compiled core splits a batch over threads, with the Python
lock released. Each system is solved by one thread with the same arithmetic as the serial loop, so
the state after a step does not depend on the number of threads, bit for bit
(`tests/core/test_core_threads.py`).

The number of solve threads allowed is `CONATUS_SOLVE_THREADS` if set, else `OMP_NUM_THREADS` if
set, else 1, and never more than the CPUs the process may use. A batch of B systems of n unknowns
uses min(allowed, B, √(B n / 900)) threads, and one thread when that is below 2 (fewer than about
3600 unknowns). The rule comes from measurement (run ``,
`benchmarks/suite/tune_threads.py`, one AMD EPYC 9654 node): a batched solve costs about 30 ns per
unknown whatever the band width (3 to 12 diagonals each side), and each thread a call starts costs
about 27 µs, so that t threads take about T₁/t + 27 µs × t. Measured gains at the chosen count:
1.1 at 4096 unknowns, 2.2 at 16 384, 4 at 65 536; 16 threads were never the fastest on 16 CPUs.
`explain()` of a solver bucket says how many threads a solve uses.

**Thread safety.** The threaded solve calls OpenBLAS's LAPACK (`gbtrs`, `getrs`) from several
threads at once on separate systems; the factors are only read. The only data a solve writes is
built on the first transposed solve of each system; since the core builds it for the
whole batch with the Python lock held, before any thread starts. Before that change, eight Python
threads making the first transposed solve of one batch at the same time corrupted the heap and
aborted the process (`free(): invalid pointer`, with the core). With the change, `benchmarks/suite/stress_threads.py` found no differing bit in 30
repetitions at 2, 8, 32 and 128 threads, forward and transposed, real and complex (96 systems of
2000 unknowns, 12 diagonals each side), with OpenBLAS on 1 and on 16 threads, nor between eight
Python threads solving one batch at once.
A ThreadSanitizer build of the core running the same script (2, 8 and 32 threads, eight Python
threads) reported no race with the current core and 107 with the core all in the lazy
build of the transposed data. ThreadSanitizer sees only the
compiled core, not OpenBLAS. In the same run, eight Python threads each stepping its own
convection solver at the same time (256 × 64 with 4 solve threads, 16 × 16 × 24 with 8) gave
final states equal byte for byte to the same solvers stepped one after the other on one thread.

The grid transforms of the CPU path run on one thread. With `f_backend="fused"`,
`CN_FFT_WORKERS=N` moves its FFTs from `numpy.fft` to `scipy.fft` on N threads. The two agree to
rounding, not bit for bit, so this stays a choice you make. The default evaluator of the
right-hand side does not read this setting.

## Cost estimate before a run

`conatus.cost.estimate` predicts the time per step, the run time and the memory of a CPU run
before it starts:

```python
import conatus as cn
from conatus.cost import estimate

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=1e5, Pr=1)
est = estimate(m, N=(256, 64), scheme="RK443", dt=1e-4, steps=5000)
print(est.explain())
```

It builds the discretised system, which a run pays for anyway, and takes no step. The prediction
is a linear model in features of that system: the band entries touched by one implicit solve, the
size of the grid transforms, the number of solves per step and the size of the state. The
coefficients were fitted on one thread of an AMD EPYC 9004-series node. They were then checked on resolutions
that the fit did not use, and `explain()` states the largest error found there.
`estimate(..., calibrate=True)` times a banded solve and an FFT for about 40 ms and rescales the
coefficients to the machine at hand.

## Results

All runs: one cluster node with AMD EPYC 9004-series processors (the CPU model of
each run is in its records), Python 3.12.14, NumPy 2.5.3, SciPy 1.17.1, OpenBLAS 0.3.34. Every
accuracy check passed in every run listed. Suite times are medians over three repetitions, each in
a fresh process, with 8 CPUs allocated.

**Version 0.3.0 and the 0.4 code, three rounds on one node.** Run ``:
0.3.0 (A) and the head of the 0.4 work (B) alternate on one node (AMD EPYC 9654, 16 CPUs
allocated), A B A B A B, ms per step, one value per round:

| Case | A, 1 thread | B, 1 thread | A, 8 threads | B, 8 threads |
|---|---|---|---|---|
| `ivp1d_burgers` | 0.53, 0.54, 0.52 | 0.49, 0.48, 0.49 | 0.53, 0.52, 0.53 | 0.48, 0.49, 0.50 |
| `ivp2d_taylor_green` | 5.70, 5.65, 5.85 | 5.40, 5.42, 5.43 | 5.71, 5.72, 5.68 | 5.25, 5.24, 5.26 |
| `ivp2d_rb` | 7.43, 7.44, 7.50 | 6.70, 6.70, 6.64 | 7.47, 7.42, 7.50 | 5.89, 5.90, 5.88 |
| `ivp3d_rb` | 54.6, 55.4, 56.2 | 52.5, 53.7, 52.6 | 54.7, 55.4, 55.8 | 32.8, 33.2, 32.7 |
| `ivp2d_taylor_green` 256 × 256 | 105.1, 106.7, 107.7 | 94.3, 95.6, 95.7 | 105.6, 108.0, 105.4 | 80.7, 80.6, 80.9 |
| `ivp2d_rb` 256 × 64 | 52.0, 52.9, 60.1 | 46.9, 48.3, 46.6 | 57.2, 57.2, 56.1 | 32.4, 32.6, 32.6 |

For each case, all twelve runs (both versions, both thread counts, three rounds) end in the same
state, byte for byte (SHA-256 recorded by the suite), and every accuracy check passed. Comparing
the slowest B round with the fastest A round, B is 1.02 to 1.11 times faster on one thread and
1.04 to 1.72 times faster on 8 threads. The largest gains are the 3D and the 256 × 64 convection
cases.

**Version 0.3.0 and the threaded solves, default sizes.** Run ``
(0.3.0) and `` (threaded solves), both on the same node, an AMD EPYC
9654.

| Case | 0.3.0, 1 thread | 0.3.0, 8 threads | threaded, 1 thread | threaded, 8 threads |
|---|---|---|---|---|
| `ivp1d_burgers`, ms/step | 0.53 | 0.54 | 0.53 | 0.54 |
| `ivp2d_taylor_green`, ms/step | 5.69 | 5.74 | 5.71 | 6.18 |
| `ivp2d_rb`, ms/step | 7.45 | 7.47 | 7.48 | 7.23 |
| `ivp3d_rb`, ms/step | 56.9 | 58.3 | 56.7 | 35.8 |
| `lbvp_poisson`, ms/solve | 61.0 | 142 | 60.6 | 65.4 |

On one thread the two versions take the same time, and the errors are identical. With 8 threads,
the 3D case gains a factor 1.6 because its solve batches are large; the 2D cases at these sizes
stay below the threading threshold. The 0.3.0 times at 8 threads are no better than at one thread:
in 0.3.0, more threads only reach the BLAS calls, which a time step barely uses.

**Larger grids, before and after on one node.** Run ``: 0.3.0 and the
threaded solves alternate on the same node (AMD EPYC 9654), two rounds each, ms per step:

| Case | 0.3.0, 1 thread | threaded, 1 thread | 0.3.0, 8 threads | threaded, 8 threads |
|---|---|---|---|---|
| `ivp2d_rb` 256 × 64 | 79.5, 58.2 | 57.8, 78.4 | 77.2, 51.4 | 43.1, 41.6 |
| `ivp2d_taylor_green` 256 × 256 | 104.4, 107.3 | 107.1, 106.6 | 110.6, 116.0 | 89.1, 91.0 |
| `lbvp_poisson` 512 × 256, ms/solve | 7600, 7607 | 7587, 7609 | 6948, 6893 | 6851, 6869 |

Errors are identical in all runs. With 8 threads the threaded solves are at least 1.19 times faster
on the 256 × 64 convection case (the best 0.3.0 round against the slower threaded round) and 1.22
times on Taylor–Green; on one thread the two versions are the same within the noise, which reaches
a factor 1.35 between rounds of the same code. The Poisson solve does not use these batches.
Separate runs on other nodes of the same CPU model were 1.4 times
slower for both versions alike, so a speed-up is only read from one node.

**Right-hand side and thread rule, before and after on one node.** Run
``: the head of (A) and (B: the right-hand side
changes and the measured thread rule) alternate on one node (AMD EPYC 9654), two rounds each, ms
per step:

| Case | A, 1 thread | B, 1 thread | A, 8 threads | B, 8 threads |
|---|---|---|---|---|
| `ivp1d_burgers` | 0.53, 0.52 | 0.50, 0.48 | 0.53, 0.53 | 0.48, 0.48 |
| `ivp2d_taylor_green` | 5.65, 5.70 | 5.35, 5.41 | 5.79, 5.81 | 5.33, 5.47 |
| `ivp2d_rb` | 7.38, 7.38 | 6.93, 6.67 | 7.00, 6.98 | 5.98, 5.88 |
| `ivp3d_rb` | 56.2, 55.1 | 53.5, 53.2 | 33.2, 33.1 | 33.5, 31.8 |
| `ivp2d_taylor_green` 256 × 256 | 105.3, 105.9 | 95.9, 96.0 | 92.5, 92.6 | 81.9, 81.8 |
| `ivp2d_rb` 256 × 64 | 77.9, 77.9 | 54.0, 72.0 | 37.6, 39.1 | 32.8, 32.3 |

The final state of every run is the same, byte for byte (SHA-256 of the state recorded by the
suite), and every accuracy check passed. Comparing the slower B round with the faster A round, B
is 1.04 to 1.10 times faster on one thread, and 1.13 to 1.17 times on 8 threads for the 2D
convection and the large Taylor-Green cases, where the thread rule now threads batches the fixed
threshold left serial; the 3D case is unchanged at 8 threads. The 256 × 64 convection case on one
thread varied between rounds of the same code (54 and 72 ms) and is not read as a gain.

**Where a step goes (0.3.0, one thread).** Run ``,
`benchmarks/suite/profile_step.py`: at 64 × 32 the right-hand side takes 61 % of a step (grid
transforms and the Python code around them) and the banded solves 27 %; at 256 × 64 the solves
take 55 %.

**Cost estimate.** Calibration run `v04_cost_c`: on the nine held-out configurations, the predicted
time per step is within a factor 1.38 of the measured one, and the predicted memory within a
factor 2.22 (1.37 for the 2D and 3D cases).

**MPI on CPU.** Run `` (AMD EPYC 9654, 32 cores allocated), plane channel 64 × 49 × 64, RK443, one thread per
rank, slab decomposition: 2.49 s/step on 1 rank, 0.80 on 4, 0.28 on 16 and 0.144 on 32 (17 times
faster than 1 rank). Every rank count gives the single-process solution to 1.8e-15 on a velocity of
order 0.5. Weak scaling with 16 × 49 × 32 per rank: 0.34 s/step on 1 rank, 0.43 on 4, 0.69 on 8 and
0.68 on 32, single samples on a shared node.

**MPI on CPU, other combinations.** Run `` (AMD EPYC 9654, 32 cores
allocated, `benchmarks/suite/mpi_scaling.py`), the same channel at 64 × 49 × 64, 10 steps, one
thread per rank, s/step; every run is compared with the single-process solver from the same state:

| Combination | 1 rank | 4 ranks | 16 ranks | 32 ranks | largest difference from one process |
|---|---|---|---|---|---|
| ComplexFourier in x and z, slab | 3.80 | 2.32 | 0.670 | 0.188 | 2.8e-15 |
| RealFourier computed as complex (`complexify`), slab | 2.03 | 1.37 | 0.272 | 0.102 | 2.6e-15 |
| RealFourier, pencil grids 2 × 2, 4 × 4, 8 × 4 | | 1.52 | 0.329 | 0.122 | 1.2e-15 |
| RealFourier, SBDF2 instead of RK443, slab | | | 0.210 | | 1.1e-16 |

The velocity is of order 0.5. The 4-rank runs of this job were slow in every combination (single
samples on a shared node).

The poloidal-toroidal form (decoupled mean flow, potentials φ and ψ) at dt = 5e-3 became
non-finite after about ten steps at this resolution in the distributed runs, one rank included.
The cause is the initial state, not the distributed path. Its
potentials contained sin z and cos(x + z) on a box of length π in z, where they are not periodic.
Their slowly decaying Fourier tails gave an initial wall-normal velocity of 1.9, 7.9 and 32.7 at
16, 32 and 64 points per direction, and an advective CFL number of 2.7 at 64 × 49 × 64 and
dt = 5e-3. From that state the single-process solver, the distributed solver on one rank and the
primitive form all became non-finite after 8 or 9 steps. Until then the distributed and
single-process states differed by at most 8e-12. From the same state, dt = 4e-3, 3e-3 and 2.5e-3
also diverged, while the smaller grids and a state of amplitude 10⁻⁶ ran 40 to 60 steps without
trouble. The benchmark now starts the poloidal-toroidal form from potentials periodic in z, and it
refuses an initial state whose advective CFL number exceeds 1. From the periodic state at
dt = 5e-3, 20 steps on 1, 4 and 16 ranks agree with one process to 1.2e-14 (run
``; 1.35, 0.50 and 0.13 s/step on a node shared with other jobs,
with ranks not bound to cores). At dt = 1e-3 (same node type), s/step:

| Combination | 1 rank | 4 ranks | 16 ranks | 32 ranks | largest difference from one process |
|---|---|---|---|---|---|
| poloidal-toroidal, per-axis transforms, slab | 1.36 | 1.70 | 0.144 | 0.060 | 3.8e-12 |
| poloidal-toroidal, grouped transforms, slab | | 1.55 | 0.177 | 0.048 | 3.0e-12 |
| poloidal-toroidal, pencil 4 × 4 | | | 0.141 | | 3.8e-12 |

There φ is of order 0.026, so the difference is 1.5e-10 relative, the same at every rank count
including one rank: it separates the distributed right-hand side of this form from the
single-process one, not one rank count from another. 32 ranks run 23 to 28 times faster than one.

The distributed solver takes only three-axis domains with one Chebyshev axis and two Fourier axes
(RealFourier or ComplexFourier), and only the right-hand side -u·grad(u) plus constant or
time-dependent forcing terms (primitive form), or the poloidal-toroidal form. It refuses, with a
typed error, every other domain (one or two axes, all-Fourier boxes, two bounded axes, the
Legendre, Hermite, Laguerre and compound bases, curvilinear domains), any other right-hand side
(for example the temperature advection of a Boussinesq problem), and symmetry subspaces. The
grouped transforms take the poloidal-toroidal form on slab grids only.
