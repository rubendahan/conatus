# Reproducibility and provenance

A number is only useful if you can say which code, which numerics and which machine produced it;
Conatus records this when the computation runs, not afterwards.

For runs you want to keep and query, use the {doc}`run registry <runs>`, which stores the same
record with the results. This page describes the record itself.

## Provenance records

```python
import conatus as cn
from conatus.expert import provenance, provenance_record

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)
rec = provenance_record(sim.solver, seed=0, tolerances={"newton": 1e-12}, out="run_dir")
print(sorted(rec))
# ['code', 'environment', 'extra', 'numerics', 'recorded_utc', 'schema', 'seed', 'started_utc']
print(provenance.problem_hash(sim.solver.plan.problem))   # structural hash of the problem
```

`provenance_record` is `conatus.provenance.record`. The bare name `conatus.expert.record` is refused
(`AmbiguousNameError`): `conatus.telemetry.record` (a telemetry frame, `telemetry_record` in
`conatus.expert`) is a different function.

`record(run)` accepts a `Problem`, a `Plan`, an `IVPSolver`, or `None` (environment only), and
writes `provenance.json` atomically when `out=` is given.

| Block | Content |
|---|---|
| `code` | git HEAD, modified paths under `python/` and `core/`, sha256 of the `conatus` tree **actually imported**, compiled module (path, CUDA, GPU visible, older than `core/` sources?); `code_matches_commit` |
| `environment` | host, platform, Python/NumPy/SciPy/Conatus versions, Slurm job and partition, `CN_RUN_ID`, threads, GPU |
| `numerics.problem` | kind, name, structural hash, readable equations |
| `numerics.discretization` | bases (type, N, bounds, dealias, grid), groups, NCC reports, parameters |
| `numerics.time_stepping` | scheme, dt, start-up, backend, ordering, reduced subspace, dtype, iteration, t |
| `tolerances`, `inner_product`, `seed`, `extra` | given by the caller, never invented |

The executed commit is checked *when* `record` is called (a logged commit is not proof of the
executed code). `record(..., require_clean=True)` raises if the code does not match a commit
exactly (dirty tree, package outside the repository, or a compiled core older than its sources).

## Determinism

- **Restart**: checkpoints hold the multistep history; a restarted run is bit-for-bit identical
  to an uninterrupted one, on CPU and GPU, and a checkpoint loaded with different numerics
  (`dt`, scheme, start-up) is refused ({doc}`timesteppers`).
- **Cross-backend**: fast paths (compiled core, CUDA, fused kernels, graphs) are compared with the
  NumPy reference backend at round-off tolerance, not bit for bit.

## Tutorials

The outputs shown in the tutorials are those of the scripts, executed as shown and included
verbatim; the first line of each output says when and on which commit it ran.
