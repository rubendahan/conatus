# The run registry

`cn.runs` records the computations you may want to find, compare, cite or re-run later, with
their numerics, code and machine.

It needs no service: an SQLite index plus one directory per run (a JSON record, an HDF5 file for
series and fields, solver checkpoints) under a root you declare. Tutorial:
{doc}`../tutorials/11_runs`.

## Where runs go

Set `$CN_RUNS` to the directory you choose (or pass `root=`); without it, a scratch directory
of the project is used. A default that resolves under your home directory is refused: runs
belong on scratch storage. `cn.runs.explain()` prints the root and every declared constant.

## Recording

```python
import numpy as np
import conatus as cn


def heat(kappa):
    """u_t = kappa u_xx, u(x, 0) = sin(3x): the amplitude decays as exp(-9 kappa t)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", kappa=kappa)
    m.scalar("u")
    m.equations("dt(u) = kappa*lap(u)")
    return m.simulate(N=32, scheme="RK443", dt=1e-3).initial(u=lambda x: np.sin(3 * x))


def decay_rate(sim):
    g = sim.grid("u")
    x = np.linspace(0, 2 * np.pi, g.size, endpoint=False)
    return -np.log(2 * np.mean(g * np.sin(3 * x))) / sim.t

sim = heat(0.2)
with cn.runs.track("heat_k0.2", tags=["decay"], params={"note": "first try"}) as run:
    sim.solver.run(200, record=run)       # problem hash, numerics, explain(), start/end state
    rate = decay_rate(sim)
    run.log(t=sim.t, rate=rate)           # a row of series (in memory, written at the end)
    run.result(rate=rate, rel_error=abs(rate - 1.8) / 1.8)   # scalar results, queryable
    run.save_field("u", sim.grid("u"), compression="gzip")   # compression has no default
run.certify("decay rate = 9 kappa (closed form)", evidence={"rel_error": run.results["rel_error"]})
```

Other ways in:

| Call | Records |
|---|---|
| `solver.run(n, record=True)` (or a name, or a dict of options) | one run per call: numerics, problem hash, parameters, code, hardware, Slurm job, start/end state; returns the run |
| `sim.run(until=..., record=True)` (native layer; also `cache=True`) | the same, through the native simulation; returns the run |
| `cn.runs.record(obj, name=..., results=..., fields=..., compression=...)` | a finished object in one call; an `EVPResult` gives eigenvalues, residuals, `growth_max`, `residual_max` |
| `@cn.runs.track("name")` on a function | one run per call: arguments as parameters, the return value as results / fields |

What is recorded, always: the structural hash of the problem, the **values** of its parameters, the
full numerical configuration and `explain()` text, the executed code (commit, dirty paths, sha256 of
the imported tree and of the compiled core), versions, host, GPU, batch job, the command line. Nothing is guessed: an unreadable field is recorded as such.

**Recording never blocks or breaks the computation.** Scalars and series are kept in memory; fields,
files and states are copied and written by a background thread through a bounded queue (a full
queue drops and counts, never waits); a disk error is recorded and reported by a
`RegistryWriteWarning` when the run closes, and the numbers are bit-for-bit those of an unrecorded
run (tested with an unwritable root).

Statuses: `provisional` (default), `certified`, `retracted`. Changing one needs a reason (and
optional evidence such as residuals); the history is kept, a retracted run is never deleted and never
reused. A run failing inside a `with` block is recorded as `failed` and the exception propagates.

## Querying

```python
import numpy as np
import conatus as cn


def heat(kappa):
    """u_t = kappa u_xx, u(x, 0) = sin(3x): the amplitude decays as exp(-9 kappa t)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", kappa=kappa)
    m.scalar("u")
    m.equations("dt(u) = kappa*lap(u)")
    return m.simulate(N=32, scheme="RK443", dt=1e-3).initial(u=lambda x: np.sin(3 * x))


def decay_rate(sim):
    g = sim.grid("u")
    x = np.linspace(0, 2 * np.pi, g.size, endpoint=False)
    return -np.log(2 * np.mean(g * np.sin(3 * x))) / sim.t

import socket

for kappa in (0.2, 0.4):                             # two recorded runs to query
    sim = heat(kappa)
    with cn.runs.track(f"heat_k{kappa}", tags=["decay"]) as run:
        sim.solver.run(100, record=run)
        run.log(t=sim.t, rate=decay_rate(sim))
        run.result(rate=decay_rate(sim))
    run.certify("decay rate = 9 kappa", evidence={"rate": run.results["rate"]})

print(cn.runs.find(problem=sim.solver.plan.problem, kappa__gt=0.15, status="certified",
                   tags=["decay"]))
print(cn.runs.find(rate__ge=1.0, host__contains=socket.gethostname()))   # results, index columns
a = cn.runs.find(name="heat_k0.2")[0]                 # or cn.runs.get(id or unique id prefix)
b = cn.runs.find(name="heat_k0.4")[0]
print(cn.runs.compare(a, b).explain())               # parameters, numerics, results, code
cn.runs.compare(a, b).plot("rate", x="t")            # both curves, each on its own abscissae
print(cn.runs.lineage(a).explain())                  # parents, forks, reproductions, cache reuse
```

`cn.runs.to_dataframe(tags="decay")` gives the same rows as a pandas table (pandas is optional).

Filters are `key=value` or `key__op=value` with `op` in `eq, ne, gt, ge, lt, le, in, contains`. A key
is an index column (`host`, `gpu`, `git_head`, `slurm_job`, `telemetry_id`, ...), else a parameter,
else a result. A key that is both a parameter and a result is refused (write `param_Re` or
`result_Re`), and an unknown key raises with a suggestion: a typo never returns an empty list.

## Reproducing

```python
import numpy as np
import conatus as cn


def heat(kappa):
    """u_t = kappa u_xx, u(x, 0) = sin(3x): the amplitude decays as exp(-9 kappa t)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", kappa=kappa)
    m.scalar("u")
    m.equations("dt(u) = kappa*lap(u)")
    return m.simulate(N=32, scheme="RK443", dt=1e-3).initial(u=lambda x: np.sin(3 * x))


def decay_rate(sim):
    g = sim.grid("u")
    x = np.linspace(0, 2 * np.pi, g.size, endpoint=False)
    return -np.log(2 * np.mean(g * np.sin(3 * x))) / sim.t

run = heat(0.2).run(steps=200, record="heat_k0.2")   # a recorded run
solver = heat(0.2).solver             # solver built by your script for the same problem
report = run.reproduce(solver)
print(report.explain())               # bit-for-bit identical?  code identical?
```

The solver must have the same problem hash, parameter values and numerics; any difference is
refused with the list, never adapted. The recorded start state is loaded, the same number of steps
is taken in a new run (`relation="reproduce"`), and the whole end state (the state, every multistep
history array, the time and the iteration) is compared bit for bit: same dtype and same bytes, so
`-0.0` differs from `0.0`. A run recorded by 0.1.0 does not hold the given-field data, the user
functions and six numerical choices that 0.1.1 records: an `IdentityWarning` names them (they are
not compared) and the end state is still compared bit for bit. If the
executed code changed since the run, a `CodeChangedWarning` says how (`require_same_code=True`
makes it an error). A decorated function is re-imported and called with the recorded arguments.

## The cache: only when you ask

```python
import numpy as np
import conatus as cn


def heat(kappa):
    """u_t = kappa u_xx, u(x, 0) = sin(3x): the amplitude decays as exp(-9 kappa t)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", kappa=kappa)
    m.scalar("u")
    m.equations("dt(u) = kappa*lap(u)")
    return m.simulate(N=32, scheme="RK443", dt=1e-3).initial(u=lambda x: np.sin(3 * x))


def decay_rate(sim):
    g = sim.grid("u")
    x = np.linspace(0, 2 * np.pi, g.size, endpoint=False)
    return -np.log(2 * np.mean(g * np.sin(3 * x))) / sim.t

heat(0.4).solver.run(200, record=True)                # computed and recorded
again = heat(0.4).solver.run(200, record=True, cache=True)   # read back: CacheReuseWarning
print(again.cache_report["hit"])
```

With `cache=True`, an identical computation already in the registry is read instead of computed,
and a `CacheReuseWarning` says which run was reused and what was read (`run.cache_report` has the
key and its parts). Identical means the same identity of the computation (the one the checkpoint
guards and the HDF5 append also compare: problem hash, parameter values, the data of every given
field by content, user functions by code, every numerical choice including `time_parameter`,
`bc_time` and `f_backend`), exact start state (and multistep history), number of steps, the
`series` functions and `series_every` (a hit returns the source run's series), and executed
code (tree and compiled core hashes, Python/NumPy/SciPy versions, architecture). For a decorated
function, identical also covers what it reads besides its arguments: its closure values and the
module globals its code names (by content; library modules, functions and classes by name and
version; the attributes it names of a user module or a user class, such as `cfg.A` or `Cfg.A`, by
content), followed into the user functions it calls, so two closures `make(1)` and `make(2)` of
one factory, or one function under `@scale(2)` and `@scale(3)`, are two computations (your own
decorators under `@track` are part of what is called and of the key). The cache is off by default; it needs `record=`; it is refused
with a callback (a hit would never call it); retracted runs are never reused; an argument, or a
value read by the function or by a user pointwise function, without a content hash makes the
call uncachable (said with a `CacheReuseWarning` naming it, then computed). A hit returns values of
the recorded type (a NumPy scalar keeps its dtype, the string `"nan"` stays a string), and its
series and fields are read from the source run through the new run (`run.series(...)`).

## Studio and export

- The {doc}`studio` lists, filters and compares registry runs (read-only), and a live run links to
  its registry record when it was recorded with `cn.runs.track(..., telemetry=tel)`.
- `cn.runs.export(dir)` writes runs as small JSON files (filters as in `find`); `cn.runs.export_mlflow(runs)`
  copies runs to MLflow if you have it (never installed by Conatus).

## Reference

```{eval-rst}
.. autofunction:: conatus.runs.track
.. autofunction:: conatus.runs.record
.. autofunction:: conatus.runs.find
.. autofunction:: conatus.runs.get
.. autofunction:: conatus.runs.compare
.. autofunction:: conatus.runs.lineage
.. autofunction:: conatus.runs.reproduce
.. autofunction:: conatus.runs.export_mlflow
.. autoclass:: conatus.runs.Run
   :members: log, result, save_field, save_file, attach, certify, retract, close, series, load_field, reproduce, explain
```
