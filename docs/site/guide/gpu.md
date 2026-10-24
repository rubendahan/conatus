# GPU

The same model runs on an NVIDIA GPU by adding `backend="cuda"`. The GPU path uses double
precision. It is experimental: it was not retested on GPU hardware for this release (status
table: {doc}`../install`, "Release status and extras").

```python
# requires: gpu
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(256, 128), scheme="RK443", dt=1e-4, backend="cuda")
sim.initial(T=lambda x, z: -z + 1e-3 * (1 - 4 * z**2))
sim.run(until=0.01)
```

The run uses the backend you pass. If the GPU cannot be used, the call raises an error that says
what is missing; it does not switch to the CPU.

## Is there a GPU here?

```python
import conatus.expert as ex
ex.gpu_available()        # True when backend="cuda" can run
print(ex.gpu_report())    # one line per requirement, with the reason when one is missing
```

`python -m conatus doctor` gives the same information with the rest of the installation
({doc}`doctor`).

## Faster steps

Two options of the time stepper make each step cheaper; both compute the same arithmetic (in
spherical geometries some sums are taken in another order: same numbers up to round-off, declared
by `explain()`):

| Option | What it does | Refused when |
|---|---|---|
| `f_backend="fused"` | box and channel: the nonlinear terms are compiled into a few fused kernels | the right-hand side depends on time |
| `f_backend="fused"` | sphere, ball, shell: the right-hand side is planned once (shared terms computed once, coefficient fields kept on the GPU, transforms that use the reality of the fields, one generated kernel per product); time-dependent right-hand sides are allowed | complex fields, `ComplexFourier`, a symmetry subspace |
| `sim.solver.run_graph(n)` | captures one step as a CUDA graph and replays it `n` times: box and channel; shell with a Runge–Kutta scheme, also when the right-hand side depends on time ({doc}`shells`) | on the CPU; box and channel: a right-hand side that depends on time |

```python
# requires: gpu
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(256, 128), scheme="RK443", dt=1e-4, backend="cuda", f_backend="fused")
sim.initial(T=lambda x, z: -z + 1e-3 * (1 - 4 * z**2))
sim.solver.run_graph(1000)
```

In a shell ({doc}`shells`):

```python
# requires: gpu
import conatus as cn

m = cn.problems.native.rotating_shell_convection(eta=0.35, Ra=100, E=1e-3, Pr=1)
sim = m.simulate((64, 48, 48), scheme="RK443", dt=1e-3, backend="cuda", f_backend="fused")
print(sim.explain())           # what is planned, and which sums change order (rounding only)
```

`run` and `run_graph` can be mixed. In the last GPU test run (A100 and L40S, 9 October 2026), a
run restarted from a checkpoint taken on the GPU matched the uninterrupted run bit for bit; the
GPU path was not retested for this release. Where a GPU path has been run on GPU hardware its
results agreed with the CPU to round-off; which paths have been run is listed below.

## What runs on the GPU

| Capability | On the GPU |
|---|---|
| Simulations in a box or a channel (2D and 3D) | yes, with fused kernels and CUDA graphs |
| Simulations in a disk, an annulus, a pipe, on a sphere, in a ball or a shell | yes (`backend="cuda"`) |
| Tangent and adjoint of the time-stepping map | yes |
| Floquet–Hill determinants | yes, batched over many trial exponents |
| Transient growth (propagator) | yes |
| Several GPUs | not in this release |
| Eigenvalue problems, steady Newton solves, continuation | CPU |

## What has been tested on a GPU

Each GPU path has tests that compare it with the CPU (`pytest -m gpu`; they are skipped on a
machine without a GPU). This release has not been re-run on a GPU: the last complete GPU suite
(`pytest -m "gpu and not slow"`, October 2026, on a development tree before this release) passed
276 tests and failed 11 on an A100 and on an L40S; the 11 were fixed and re-run as targeted tests
on an A100, where one memory-prediction test still predicts 3 % more than measured. Paths whose
tests were written after that suite have not been run on a GPU: treat them as experimental there.
Some GPU tests use development infrastructure and are not in the public source tree; the table
says "development suite only" for those.

| Capability | GPU tests in the public tree | Run on GPU hardware |
|---|---|---|
| Box and channel simulations, fused kernels, CUDA graphs | `tests/ivp/test_gpu_backend.py`, `tests/cross_backend/` | A100 and L40S |
| HDF5 checkpoints and restarts on the GPU | `tests/ivp/test_checkpoint_series_gpu.py` | A100 and L40S |
| Disk, annulus, pipe, sphere, ball, shell | `tests/spherical/`; disk to pipe: development suite only | A100 (on the L40S the comparison ran before a fix of the test's own assertions and was not re-run) |
| Hermite, Laguerre and compound bases | development suite only | A100 and L40S |
| Legendre, Jacobi and ultraspherical bases | development suite only | A100 |
| User functions (`m.function`) | `tests/model/test_pointwise_function_gpu.py` | A100 |
| Tangent and adjoint of the time-stepping map | development suite only | A100 and L40S |
| Floquet–Hill determinants | development suite only | A100 and L40S |
| Ensembles | `tests/ensembles/test_gpu_ensemble.py` | A100 and L40S |
| Tracers, dye, Lyapunov exponents | `tests/model/test_particles_gpu.py` | not run on a GPU |
| Symmetry subspaces of the disk, cylinder and pipe | development suite only | L40S |

## How much GPU memory?

The device memory of a run is predicted before any GPU is used, from the compiled objects:

```python
import conatus as cn
from conatus.runtime.memory import plan_memory

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
rep = plan_memory(m.discretize((256, 128)), "RK443")   # no GPU needed
print(rep.explain())
```

The report lists the bytes held by the two allocators (CuPy's pool and the compiled core's pool,
buffer by buffer), then, on its own line, the **CUDA process floor**: the CUDA context, kernel images
and cuFFT/cuBLAS handles that every GPU process holds and that no buffer model predicts. It is a
declared constant, `CUDA_PROCESS_FLOOR_BYTES` = 512 MiB, the largest floor measured on A100 and
L40S (439 to 504 MiB over 24 cases, CUDA 12.9) rounded up; the `TOTAL` line includes it.
`sim.solver.memory_report()` gives the same report for a solver built with `backend="cuda"`; on a
CPU solver it refuses and points to `plan_memory`.

## On a cluster

Submit GPU work as a batch job on an allocated GPU node, under the cluster's scheduling policy. Build the compiled core once
for the architectures of your cards ({doc}`../install`, GPU section: tested CUDA and CuPy versions
and the exact commands; A100 and L40S are the tested targets), then
run your script in a job that requests one GPU. Recording the run ({doc}`runs`) keeps the host,
the GPU and the job id with the results.
