"""Native run registry (``cn.runs``): record, query, compare, reproduce and (on request) reuse runs.

No external service: SQLite (index) + one directory per run (JSON record, HDF5 series and fields,
solver checkpoints) under a declared root (``$CN_RUNS``, else ``$CN_TMP/runs``, else
``<repo>/scratch/runs``; never the home directory by default).

Recording (never blocks the computation, never raises into it; see :mod:`conatus.runs.run`)::

    import conatus as cn
    with cn.runs.track("rb_2d", tags=[""], params={"Ra": 1e6}) as run:
        solver.run(2000, record=run)                  # numerics, problem hash, start/end state
        run.log(t=solver.t, Nu=nu)                    # series
        run.result(Nu=nu)                             # scalar results (queryable)
        run.save_field("T", solver.grid("T"), compression="gzip")
    run.certify("Nu within 1e-3 of Clever & Busse 1974, residual 2e-12")

    solver.run(2000, record=True)                     # one run per call (IVPSolver)
    cn.runs.record(evp_result, name="os_5772", compression=None)   # a finished object

    @cn.runs.track("eig", cache=True)                 # one run per call; identical calls reused
    def least_stable(Re: float, N: int) -> dict: ...

Queries: ``find(problem=..., Re__gt=1000, status="certified", tags=[""])``, ``get(id)``,
``compare(a, b)``, ``lineage(id)``, ``to_dataframe()``; ``run.reproduce(solver)``; exports
``export(dir)`` (versioned JSON for ``results/`` and the paper) and ``export_mlflow`` (optional).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .analysis import Comparison, Lineage, RunList
from .analysis import compare as _compare
from .analysis import lineage as _lineage
from .analysis import to_dataframe as _to_dataframe
from .integrations import (CacheReuseWarning, CodeChangedWarning, ReproduceReport, Tracker, export_mlflow,  # noqa: F401 -- importable here, not in __all__ (M5)
                           load_return, record, reproduce, track)  # noqa: F401 -- importable here, not in __all__ (M5)
from .integrations import export as _export
from .keys import code_fingerprint  # noqa: F401 -- importable here, not in __all__ (M5)
from .run import CLOSE_TIMEOUT_S, COMPRESSIONS, QUEUE_CAPACITY, RegistryWriteWarning, Run, current  # noqa: F401 -- importable here, not in __all__ (M5)
from .store import BUSY_TIMEOUT_S, SCHEMA, STATES, STATUSES, Registry, default_root  # noqa: F401 -- importable here, not in __all__ (M5)

# Documented names only (M5, 0.2); the others stay importable from this module.
__all__ = ["track", "record", "find", "get", "compare", "lineage", "to_dataframe", "reproduce",
           "export", "export_mlflow", "registry", "explain", "Run", "RunList", "Registry",
           "CacheReuseWarning", "CodeChangedWarning", "RegistryWriteWarning"]


def registry(root: Any = None) -> Registry:
    """The registry at ``root`` (default: :func:`default_root`)."""
    return root if isinstance(root, Registry) else Registry(root)


def find(problem: Any = None, *, name: str | None = None, status: str | Iterable[str] | None = None,
         state: str | Iterable[str] | None = None, tags: str | Iterable[str] | None = None,
         parent: Any = None, since: Any = None, until: Any = None, limit: int | None = None,
         order: str = "-created", root: Any = None, **filters: Any) -> RunList:
    """Runs matching every criterion, newest first.

    ``problem``: a Problem / Plan / solver (its structural hash), a hash prefix (>= 8 hex digits)
    or a problem name. ``filters``: ``key=value`` or ``key__op=value`` with op in eq, ne, gt, ge,
    lt, le, in, contains. A key is an index column (host, gpu, git_head, slurm_job, ...), else a
    parameter, else a result; a key that is both a parameter and a result is refused (write
    ``param_<key>`` or ``result_<key>``), and an unknown key is an error (typos never return an
    empty list silently), except in an empty registry."""
    reg = registry(root)
    ids = reg.query_ids(problem=problem, name=name, status=status, state=state, tags=tags,
                        parent=parent, since=since, until=until, limit=limit, order=order,
                        filters=filters)
    return RunList(Run.load(reg, i) for i in ids)


def get(run_id: Any, *, root: Any = None) -> Run:
    """One run by id or unique id prefix."""
    reg = registry(root)
    return Run.load(reg, reg.resolve_id(run_id))


def compare(a: Any, b: Any, *, root: Any = None) -> Comparison:
    """Differences between two runs (ids or Run objects): parameters, numerics, results, code."""
    ra = a if isinstance(a, Run) else get(a, root=root)
    rb = b if isinstance(b, Run) else get(b, root=root)
    return _compare(ra, rb)


def lineage(run: Any, *, root: Any = None) -> Lineage:
    """Parent chain and descendants (fork, restart, reproduce, cache reuse) of a run."""
    reg = run.registry if isinstance(run, Run) and root is None else registry(root)
    return _lineage(run, reg)


def to_dataframe(runs: Iterable[Run] | None = None, *, root: Any = None, **filters: Any):
    """pandas DataFrame of ``runs`` (default: ``find(**filters)``); pandas is optional."""
    return _to_dataframe(find(root=root, **filters) if runs is None else runs)


def export(dest: Any, runs: Iterable[Run] | None = None, *, root: Any = None, **filters: Any):
    """Small JSON per run in ``dest`` (default selection: ``status="certified"``)."""
    if runs is None:
        filters.setdefault("status", "certified")
        runs = find(root=root, **filters)
    return _export(dest, runs)


def reindex(root: Any = None) -> dict:
    """Rebuild the SQLite index from the run directories."""
    return registry(root).reindex()


def explain(root: Any = None) -> str:
    """Where the registry is, what it holds, and every declared constant."""
    return (registry(root).explain()
            + f"\n  schema {SCHEMA}; statuses {STATUSES}; states {STATES}"
            + f"\n  writer queue {QUEUE_CAPACITY} items (fields, files, states; dropped and counted "
              f"beyond), close waits <= {CLOSE_TIMEOUT_S:g} s; field compressions {COMPRESSIONS} "
              "(lossless, declared per field)"
            + "\n  cache: off unless cache=True; key = problem + params + numerics/input + executed code"
              " (tree sha256, core sha256, Python/NumPy/SciPy, machine); retracted runs never reused")
