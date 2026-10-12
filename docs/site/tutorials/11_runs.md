# 11. Record, query, reproduce: the run registry

Every run can be found again, compared, re-run to the bit, and reused only when you ask.

**Goal.** Record a small parameter scan, certify the runs that pass a closed-form check, query them,
compare two of them, reproduce one bit for bit, and see the cache reuse an identical computation
only when `cache=True` is given.
**Expected.** Heat equation $u_t = \kappa u_{xx}$ with $u(x,0) = \sin 3x$: the measured decay rate
equals $9\kappa$ (closed form) to better than $10^{-9}$ relative for every run; the query returns
exactly the two certified runs; `reproduce()` is bit-identical; with the cache off an identical call
is computed again, with `cache=True` it takes no step and returns the recorded end state to the bit.
**Time:** a few seconds of CPU.
**Script:** [`t11_runs.py`](code/t11_runs.py) · **Tests behind it:** `tests/runs/` (including
several processes writing at once, and an unwritable registry leaving the run bit-identical).
**Guide:** {doc}`../guide/runs`.

## The problem

```{literalinclude} code/t11_runs.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

The tutorial keeps its registry in its working directory (`CN_RUNS=registry`). Point `CN_RUNS` at a
scratch disk for your own work: a registry holds states, and states can be large.

```{literalinclude} code/t11_runs.py
:language: python
:start-after: "# [problem]"
:end-before: "# [/problem]"
```

The heat equation is a native model on a periodic line; `kappa` is a model parameter. Its value is
not part of the problem's structural hash, but it is recorded as a run parameter and enters the cache
key (two runs at different $\kappa$ are never confused).

## Record and certify

```{literalinclude} code/t11_runs.py
:language: python
:start-after: "# [record]"
:end-before: "# [/record]"
```

- `sim.solver.run(n, record=run)` records into the open run: problem hash, parameter values,
  numerics and `explain()`, the executed code, host, GPU, Slurm job, and the start and end states.
  `sim.solver` is the time stepper the simulation wraps; recording and the cache are options of its
  `run`.
- `run.result(...)` stores scalars you can query. `certify` needs a reason, and here carries the
  evidence (the relative error against the closed form); the history of statuses is kept.

## Query and compare

```{literalinclude} code/t11_runs.py
:language: python
:start-after: "# [query]"
:end-before: "# [/query]"
```

`kappa__gt=0.15` filters on a parameter; results, tags, statuses and index columns (`host`, `gpu`,
`git_head`, `slurm_job`...) work the same way. An unknown key is an error with a suggestion, never an
empty list.

## Reproduce

```{literalinclude} code/t11_runs.py
:language: python
:start-after: "# [reproduce]"
:end-before: "# [/reproduce]"
```

The solver you pass must be the same problem with the same parameter values and numerics (checked,
never adapted); the recorded start state is loaded and the end state compared bit for bit.

## The cache, only when asked

```{literalinclude} code/t11_runs.py
:language: python
:start-after: "# [cache]"
:end-before: "# [/cache]"
```

## Check

Every decay rate equals $9\kappa$ to $< 10^{-9}$ relative (RK443, $\Delta t = 10^{-3}$, $t = 0.2$); the
query returns the two certified runs in order; the reproduction and the cache hit give the recorded
end state bit for bit, and the identical call without `cache=True` was computed again.

```{literalinclude} code/t11_runs.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

## Output (as executed)

```{literalinclude} outputs/t11_runs.txt
:language: text
```

## Going further

- `@cn.runs.track("name", cache=True)` records every call of a function (arguments as parameters,
  return value as results) and reuses identical calls on request.
- In a Slurm job the record also carries the job id and the nodes.
- {doc}`../guide/runs` describes every query key, statuses and lineage.
