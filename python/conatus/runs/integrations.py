"""Entry points that create runs: ``track`` (context / decorator), ``record``, ``solver.run(record=)``;
``reproduce``, the declared cache, and the exports (paper tables, MLflow).

The cache is **off unless asked** (``cache=True``) and never silent: a hit emits a
:class:`CacheReuseWarning` saying which run was reused and what was read instead of computed, and
the new run records ``reused_from`` and the full report (``run.cache_report``).
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import MissingDependencyError, OptionError, UnsupportedError, check_count, did_you_mean
from .._explain import full_explain
from ._util import atomic_write_json, diff_dicts, from_jsonable, is_exact, jsonable
from .keys import (_clockless, code_fingerprint, describe_parts, exact_hash, finish_key, function_parts,
                   ivp_parts)
from .run import QUEUE_CAPACITY, Run, _to_host, problem_info
from .store import Registry


class CacheReuseWarning(UserWarning):
    """A computation was read from the registry instead of being computed (``cache=True``)."""


class CodeChangedWarning(UserWarning):
    """``reproduce()`` runs code that differs from the code recorded with the original run."""


from ..tolerances import REQUIRED as _REQUIRED  # noqa: E402  (help() shows REQUIRED)
_RECORD_KEYS = ("name", "tags", "params", "status", "parent", "relation", "root", "state",
                "series", "series_every", "notes", "telemetry", "capacity")


def _registry(root: Any) -> Registry:
    return root if isinstance(root, Registry) else Registry(root)


def _options(record: Any) -> dict:
    """``record=`` of ``solver.run``: True, a run name, or a dict of the keys in _RECORD_KEYS."""
    if record is True:
        return {}
    if isinstance(record, str):
        return {"name": record}
    if isinstance(record, Mapping):
        bad = [k for k in record if k not in _RECORD_KEYS]
        if bad:
            raise OptionError(f"record=: unknown option(s) {bad}",
                              fix=f"options: {', '.join(_RECORD_KEYS)}" + did_you_mean(bad[0], _RECORD_KEYS))
        return dict(record)
    raise OptionError(f"record= must be True, a run name, a dict of options or a Run, not "
                      f"{type(record).__name__}")


# ------------------------------------------------------------------------------------ IVP runs
def ivp_run(solver: Any, nsteps: int, callback: Callable | None, every: int | None, *,
            record: Any, cache: bool) -> Run | None:
    """``IVPSolver.run(nsteps, callback, every, record=..., cache=...)`` (arguments already checked
    by the solver). Returns the run (None only if ``record`` is None and ``cache`` is False)."""
    if record is None:
        if cache:
            raise OptionError("solver.run: cache=True needs record=",
                              why="the cache reads the registry, and a computed result must be "
                                  "recorded to be reusable later",
                              fix="solver.run(n, record=True, cache=True)")
        return None
    if isinstance(record, Run):
        if cache:
            raise OptionError("solver.run: cache=True with record=<open run> is not supported",
                              fix="record=True (one registry run per call) to use the cache")
        _ivp_segment(record, solver, nsteps, callback, every, None, None, True)
        return record
    o = _options(record)
    series, series_every = o.get("series"), o.get("series_every")
    if (series is None) != (series_every is None):
        raise OptionError("record=: 'series' and 'series_every' go together",
                          why="how often series are evaluated is part of the run's cost",
                          fix="record=dict(series={'energy': f}, series_every=10)")
    if series_every is not None:        # a float used to be truncated for the callback period and
        check_count(series_every, "series_every", "record=")   # kept in the modulo (2.5: every 10 steps)
    store_state = o.get("state", True)
    if cache and not store_state:
        raise OptionError("cache=True needs state=True (the end state is what is reused)")
    if cache and callback is not None:
        raise OptionError("solver.run: cache=True with a callback",
                          why="a cache hit computes nothing, so the callback would never be called",
                          fix="drop the callback, or cache=False")
    reg = _registry(o.get("root"))
    name = o.get("name") or solver.plan.problem.name or "ivp"
    parts = ivp_parts(solver, nsteps) if store_state else None
    if parts is not None and series:
        # the series a hit returns are those of the source run: their functions and cadence are
        # part of the key (identity review: a hit with other series functions or another
        # series_every returned the source run's series under this run's request)
        from ..io._identity import Undeclared, callable_identity
        try:
            parts["series"] = {"every": int(series_every), "functions": {
                str(k): callable_identity(f, f"series {k}") for k, f in sorted(series.items())}}
        except Undeclared as why:
            parts["undeclared"] = [*parts.get("undeclared", []), str(why)]
    undeclared = (parts or {}).get("undeclared")
    if undeclared:
        parts = None             # no key: never a cache source, never a hit (review finding 1)
    kw = dict(tags=o.get("tags", ()), params=o.get("params"), parent=o.get("parent"),
              relation=o.get("relation"), status=o.get("status", "provisional"),
              telemetry=o.get("telemetry"), capacity=o.get("capacity", QUEUE_CAPACITY),
              notes=o.get("notes"))
    report = None
    if cache and undeclared:
        why = "; ".join(undeclared)
        report = {"enabled": True, "hit": False, "miss_reason": f"no cache key: {why}"}
        warnings.warn(CacheReuseWarning(
            f"conatus.runs: cache not used for {name}: {why} (a dependency without a content "
            "hash could have changed since an earlier run); computing"), stacklevel=3)
    elif cache:
        key, full = finish_key(parts, code_fingerprint())
        report = {"enabled": True, "hit": False, "key": key, "parts": describe_parts(full)}
        src_id = _lookup(reg, key, report, full)
        if src_id is not None:
            src = Run.load(reg, src_id)
            end = src.path / "state_end.npz"
            if end.exists():
                run = Run(reg, name, obj=solver, key_parts=parts, **kw)
                t0 = time.perf_counter()
                solver.load_checkpoint(end)
                report.update(hit=True, reused_from=src_id,
                              reused=["end state (X, multistep history, t, iteration) from "
                                      f"{end.name}", "results (copied into this run)",
                                      f"series and fields: read from run {src_id} through "
                                      "this run (run.series(), run.load_field())"],
                              computed="nothing: 0 steps taken",
                              source_duration_s=src.rec.get("duration_s"),
                              load_s=round(time.perf_counter() - t0, 6))
                with run._lock:
                    run.rec["reused_from"] = src_id
                    run.rec["cache"] = report
                    run.rec["inputs"] = {"kind": "ivp", "nsteps": int(nsteps), "segments": 0}
                    run.rec["results"].update(src.results)
                run.close()
                warnings.warn(CacheReuseWarning(
                    f"conatus.runs cache hit: nothing computed; state after {nsteps} steps read from "
                    f"run {src_id} (took {src.rec.get('duration_s')} s); key {key[:12]}, "
                    f"report in run {run.id}.cache_report"), stacklevel=3)
                return run
            report["miss_reason"] = f"run {src_id} has no stored end state"
    run = Run(reg, name, obj=solver, key_parts=parts, **kw)
    if report is not None:
        with run._lock:
            run.rec["cache"] = report
    with run:
        _ivp_segment(run, solver, nsteps, callback, every, series, series_every, store_state)
    return run


def _lookup(reg: Registry, key: str, report: dict, full: dict | None = None) -> str | None:
    """Cache lookup that never fails the computation: an unreadable registry is a miss (reported).
    No lookup when the package sources changed after import (``full["code"]``): the key would
    name code that does not run."""
    changed = ((full or {}).get("code") or {}).get("changed_since_import")
    if changed:
        report["miss_reason"] = ("package sources changed after import (" + ", ".join(changed[:3])
                                 + "): the code that runs is not the one the key names")
        warnings.warn(CacheReuseWarning(
            f"conatus.runs: cache not used: {report['miss_reason']}; computing (restart the "
            "interpreter to use the cache)"), stacklevel=4)
        return None
    try:
        src = reg.cache_lookup(key)
    except Exception as err:  # noqa: BLE001 -- a cache never fails the computation; reported
        report["miss_reason"] = f"registry unreadable: {type(err).__name__}: {err}"
        return None
    if src is None:
        report["miss_reason"] = "no completed, non-retracted run with this key"
    return src


def _ivp_segment(run: Run, solver: Any, nsteps: int, callback: Callable | None, every: int | None,
                 series: Mapping[str, Callable] | None, series_every: int | None,
                 store_state: bool) -> None:
    """Advance the solver inside ``run``: start state, steps (plain ``solver.run``), end state."""
    if run.rec.get("problem") is None:
        run.attach(solver)
    segs = run.rec.setdefault("segments", [])
    tag = "" if not segs else f"s{len(segs)}_"
    seg = {"nsteps": int(nsteps), "t_start": float(solver.t), "iteration_start": int(solver.iteration)}
    if store_state:
        run._save_state(f"{tag}start", solver)
    if series:
        def cb(s: Any) -> None:
            if callback is not None and s.iteration % every == 0:
                callback(s)
            if s.iteration % series_every == 0:
                run.log(step=s.iteration, t=float(s.t), **{k: f(s) for k, f in series.items()})
        g = math.gcd(int(every or 0), int(series_every))
        args = (cb, g)
    else:
        args = (callback, every)
    t0 = time.perf_counter()
    solver.run(nsteps, *args)
    seg.update(t_end=float(solver.t), iteration_end=int(solver.iteration),
               wall_s=round(time.perf_counter() - t0, 6), state_stored=bool(store_state))
    if store_state:
        run._save_state(f"{tag}end", solver)
    with run._lock:
        segs.append(seg)
        run.rec["inputs"] = {"kind": "ivp", "nsteps": int(nsteps), "segments": len(segs)}


# ------------------------------------------------------------------------------ track / decorator
class Tracker:
    """What :func:`track` returns: a context manager (``with track(...) as run``) and a decorator
    (``@track(...)``: one run per call, arguments as parameters, return value as results)."""

    def __init__(self, name: str | None, opts: dict) -> None:
        self.name, self.opts = name, opts
        self._run: Run | None = None

    def __enter__(self) -> Run:
        if self.opts.get("cache"):
            raise OptionError("cache=True applies to a decorated function or solver.run, not to a "
                              "with-block", why="a with-block cannot be skipped on a cache hit",
                              fix="decorate a function: @cn.runs.track('name', cache=True)")
        if not self.name:
            raise OptionError("track(): a with-block needs a run name", fix="with track('rb_1e6') as run:")
        o = {k: v for k, v in self.opts.items() if k not in ("root", "cache", "compression")}
        self._run = Run(_registry(self.opts.get("root")), self.name, **o)
        return self._run.__enter__()

    def __exit__(self, *exc: Any) -> bool:
        run, self._run = self._run, None
        return run.__exit__(*exc)

    def __call__(self, fn: Callable) -> Callable:
        opts = dict(self.opts)
        name = self.name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return call_recorded(fn, args, kwargs, name=name, **opts)
        wrapper.__conatus_tracked__ = fn            # what reproduce() calls (_untracked)
        return wrapper


def _untracked(fn: Callable) -> Callable:
    """The function a :func:`track` decorator wraps (with the user's own decorators, which
    ``inspect.unwrap`` would strip: identity review)."""
    while getattr(fn, "__conatus_tracked__", None) is not None:
        fn = fn.__conatus_tracked__
    return fn


def track(name: str | None = None, *, tags: Iterable[str] = (), params: Mapping[str, Any] | None = None,
          obj: Any = None, parent: Any = None, relation: str | None = None,
          status: str = "provisional", root: Any = None, telemetry: Any = None,
          capacity: int = QUEUE_CAPACITY, notes: str | None = None, cache: bool = False,
          compression: str | None = None) -> Tracker:
    """Record a block (``with track("rb", tags=["convection"]) as run: ...``) or every call of a function
    (``@track("eig", cache=True)``). ``compression`` applies to arrays returned by a decorated
    function (None, 'gzip', 'lzf'; all lossless)."""
    return Tracker(name, dict(tags=tags, params=params, obj=obj, parent=parent, relation=relation,
                              status=status, root=root, telemetry=telemetry, capacity=capacity,
                              notes=notes, cache=cache, compression=compression))


def call_recorded(fn: Callable, args: tuple, kwargs: dict, *, name: str, cache: bool = False,
                  compression: str | None = None, root: Any = None, _with_run: bool = False,
                  **run_kw: Any) -> Any:
    """One recorded call of ``fn`` (the decorator's body); ``(value, run)`` with ``_with_run``."""
    raw = inspect.unwrap(fn)
    bound = inspect.signature(raw).bind(*args, **kwargs)
    bound.apply_defaults()
    argsd = dict(bound.arguments)
    target = f"{raw.__module__}:{raw.__qualname__}"
    try:
        src_sha = hashlib.sha256(inspect.getsource(raw).encode()).hexdigest()
    except (OSError, TypeError):
        src_sha = None
    inexact = [k for k, v in argsd.items() if not exact_hash(v)]
    # what the function reads besides its arguments: closure values, module globals, the user
    # functions it calls (review finding 49: two closures of one factory shared a key)
    from ..io._identity import Undeclared, callable_identity
    try:
        # ``fn`` as called, with the user's own decorators (their parameters are part of what
        # is computed; ``raw`` names the target and gives the source)
        deps, undeclared = callable_identity(fn, target), None
    except Undeclared as why:
        deps, undeclared = None, str(why)
    parts = (None if (src_sha is None or inexact or undeclared)
             else function_parts(target, src_sha, argsd, deps))
    reg = _registry(root)
    params = {**(run_kw.pop("params", None) or {}), **argsd}
    report = None
    if cache:
        if parts is None:
            why = ("source code unavailable" if src_sha is None
                   else f"argument(s) {inexact} have no exact content hash" if inexact
                   else f"{undeclared} (a dependency without a content hash could have changed "
                        "since an earlier call)")
            report = {"enabled": True, "hit": False, "miss_reason": f"no cache key: {why}"}
            warnings.warn(CacheReuseWarning(f"conatus.runs: cache not used for {target}: {why}; "
                                            "computing"), stacklevel=3)
        else:
            key, full = finish_key(parts, code_fingerprint())
            report = {"enabled": True, "hit": False, "key": key, "parts": describe_parts(full)}
            src_id = _lookup(reg, key, report, full)
            if src_id is not None:
                src = Run.load(reg, src_id)
                try:
                    value = load_return(src)
                except Exception as err:  # noqa: BLE001 -- e.g. data.h5 missing: recompute, reported
                    report["miss_reason"] = f"run {src_id}: return value unreadable ({err})"
                else:
                    run = Run(reg, name, params=params, key_parts=parts, **run_kw)
                    report.update(hit=True, reused_from=src_id,
                                  reused=["return value", "results (copied into this run)",
                                          f"series and fields: read from run {src_id} through "
                                          "this run"],
                                  computed="nothing: the function was not called",
                                  source_duration_s=src.rec.get("duration_s"))
                    with run._lock:
                        run.rec.update(reused_from=src_id, cache=report, returns=src.rec.get("returns"),
                                       inputs=_fn_inputs(target, src_sha, argsd))
                        run.rec["results"].update(src.results)
                    run.close()
                    warnings.warn(CacheReuseWarning(
                        f"conatus.runs cache hit: {target} not called; return value read from run "
                        f"{src_id} (took {src.rec.get('duration_s')} s); key {key[:12]}"), stacklevel=3)
                    return (value, run) if _with_run else value
    run = Run(reg, name, params=params, key_parts=parts, **run_kw)
    with run._lock:
        run.rec["inputs"] = _fn_inputs(target, src_sha, argsd)
        if report is not None:
            run.rec["cache"] = report
    with run:
        # the function as decorated by the user: calling ``inspect.unwrap(fn)`` skipped the
        # user's own decorators (``@track @scale(2) def g`` returned g(x), not 2 g(x); identity
        # review)
        out = fn(*args, **kwargs)
        _store_return(run, out, compression)
    return (out, run) if _with_run else out


def _fn_inputs(target: str, src_sha: str | None, argsd: dict) -> dict:
    return {"kind": "function", "target": target, "source_sha256": src_sha,
            "args": jsonable(argsd), "args_exact": all(is_exact(v) for v in argsd.values())}


def _scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, complex, str, np.number, np.bool_))


def _store_return(run: Run, out: Any, compression: str | None) -> None:
    """Return value -> results (scalars) and fields (arrays); layout in ``rec['returns']``."""
    def one(key: str, v: Any) -> dict | None:
        if isinstance(v, np.generic) and v.dtype.kind in "biufc":
            # a NumPy number keeps its dtype (review finding 50: a cache hit returned another
            # type): stored as a 0-d field, also indexed as a result. Numbers only: HDF5 has no
            # type for np.str_ or np.datetime64, whose field was lost in the writer thread
            # (identity review); np.str_ stays a string result
            run.result(**{key: v})
            run.save_field(key, np.asarray(v), compression=None)
            return {"field": key, "scalar": True}
        if _scalar(v):
            run.result(**{key: v})
            return {"result": key}
        if isinstance(v, np.ndarray) and v.dtype != object:
            run.save_field(key, v, compression=compression)
            return {"field": key}
        return None
    if out is None:
        layout = {"kind": "none"}
    elif _scalar(out) or isinstance(out, np.ndarray):
        item = one("return", out)
        layout = {"kind": "value", "item": item} if item else {"kind": "unstored", "repr": repr(out)[:200]}
    elif isinstance(out, Mapping) and all(isinstance(k, str) for k in out):
        items = {k: one(k, v) for k, v in out.items()}
        layout = ({"kind": "dict", "items": items} if all(items.values())
                  else {"kind": "unstored", "repr": repr(out)[:200],
                        "why": f"values of type {sorted({type(v).__name__ for k, v in out.items() if not items[k]})}"})
    elif isinstance(out, (tuple, list)):
        items = [one(f"return_{i}", v) for i, v in enumerate(out)]
        layout = ({"kind": type(out).__name__, "items": items} if all(items)
                  else {"kind": "unstored", "repr": repr(out)[:200]})
    else:
        layout = {"kind": "unstored", "repr": repr(out)[:200], "why": f"type {type(out).__name__}"}
    with run._lock:
        run.rec["returns"] = layout
        if layout["kind"] == "unstored":
            run.rec["cache_key"] = None            # nothing to reuse: never a cache hit
            run._key_parts = None


def load_return(run: Run) -> Any:
    """The stored return value of a decorated call (exact: same numbers, same container)."""
    layout = run.rec.get("returns") or {}
    kind = layout.get("kind")

    def one(item: dict) -> Any:
        if "result" in item:
            return from_jsonable(run.results[item["result"]])
        a = run.load_field(item["field"])
        return a[()] if item.get("scalar") else a
    if kind == "none":
        return None
    if kind == "value":
        return one(layout["item"])
    if kind == "dict":
        return {k: one(v) for k, v in layout["items"].items()}
    if kind in ("tuple", "list"):
        vals = [one(v) for v in layout["items"]]
        return tuple(vals) if kind == "tuple" else vals
    raise UnsupportedError(f"run {run.id}: return value not stored ({layout.get('why', kind)})")


# ------------------------------------------------------------------------------------ record()
def record(obj: Any = None, *, name: str | None = None, params: Mapping[str, Any] | None = None,
           results: Mapping[str, Any] | None = None, series: Mapping[str, Any] | None = None,
           fields: Mapping[str, Any] | None = None, files: Iterable[Any] = (),
           compression: Any = _REQUIRED, tags: Iterable[str] = (), status: str = "provisional",
           parent: Any = None, relation: str | None = None, root: Any = None,
           telemetry: Any = None, notes: str | None = None) -> Run:
    """Record one finished computation in one call and return the closed run.

    ``obj``: a Problem, Plan or solver (numerics, problem hash, ``explain()``), an ``EVPResult``
    (eigenvalues and residuals stored as fields, ``n_modes``, ``growth_max``, ``residual_max`` as
    results), or any object with ``explain()``. ``series``: {name: 1-D sequence}, all the same
    length. ``fields``: {name: array}, which requires ``compression`` (None, 'gzip', 'lzf')."""
    is_evp = hasattr(obj, "values") and hasattr(obj, "residuals") and hasattr(obj, "method")
    arrays = dict(fields or {})
    res = dict(results or {})
    if is_evp:
        vals = np.asarray(obj.values)
        arrays.setdefault("eigenvalues", vals)
        arrays.setdefault("residuals", np.asarray(obj.residuals))
        res.setdefault("n_modes", int(vals.size))
        if vals.size:
            res.setdefault("growth_max", float(np.max(vals.real)))
            res.setdefault("residual_max", float(np.max(obj.residuals)))
    if arrays and compression is _REQUIRED:
        raise OptionError("record(): arrays to store but no compression= declared",
                          why="storage choices are declared explicitly",
                          fix="compression=None (as is), 'gzip' or 'lzf' (all lossless)")
    lengths = {k: len(v) for k, v in (series or {}).items()}
    if len(set(lengths.values())) > 1:
        raise OptionError(f"record(): series of different lengths {lengths}",
                          fix="record them in separate runs, or use run.log() rows in track()")
    reg = _registry(root)
    info = problem_info(obj) if obj is not None and not is_evp else None
    name = name or (info or {}).get("name") or (getattr(obj, "label", None) if is_evp else None) \
        or (type(obj).__name__ if obj is not None else "record")
    run = Run(reg, name, tags=tags, params=params, obj=obj if info is not None else None,
              parent=parent, relation=relation, status=status, telemetry=telemetry, notes=notes)
    if info is None and obj is not None and hasattr(obj, "explain"):
        with run._lock:
            run.rec["explain"]["object"] = str(full_explain(obj))
    with run:
        if res:
            run.result(**res)
        n = next(iter(lengths.values()), 0)
        for i in range(n):
            run.log(**{k: v[i] for k, v in series.items()})
        for k, a in arrays.items():
            run.save_field(k, a, compression=None if compression is _REQUIRED else compression)
        for f in files:
            run.save_file(f)
    return run


# ------------------------------------------------------------------------------------ reproduce
class ReproduceReport:
    """Outcome of :func:`reproduce`: new run, identical or not (and by how much), code identical."""

    def __init__(self, original: Run, new: Run | None, identical: bool, max_abs_diff: float,
                 code_identical: bool, code_diff: dict, details: dict) -> None:
        self.original, self.new, self.identical = original, new, identical
        self.max_abs_diff, self.code_identical, self.code_diff = max_abs_diff, code_identical, code_diff
        self.details = details

    def explain(self) -> str:
        lines = [f"reproduce {self.original.id} -> {self.new.id if self.new else '(not recorded)'}: "
                 + ("bit-for-bit identical" if self.identical
                    else f"DIFFERENT (max |diff| = {self.max_abs_diff:.3e})"),
                 "  code: " + ("identical to the recorded code" if self.code_identical else
                               "CHANGED since the original run: " + ", ".join(
                                   f"{k}: {str(a)[:12]} -> {str(b)[:12]}"
                                   for k, (a, b) in self.code_diff.get("changed", {}).items()))]
        lines += [f"  {k}: {v}" for k, v in self.details.items()]
        return "\n".join(lines)

    __repr__ = explain


def reproduce(run: Run, solver: Any = None, *, fn: Callable | None = None, record: bool = True,
              require_same_code: bool = False) -> ReproduceReport:
    """Run the recorded computation again and compare with the recorded output.

    IVP runs: pass a ``solver`` built for the same problem and numerics (checked: problem hash,
    parameters, bases, scheme, dt... ; any difference is refused); the recorded
    start state is loaded, the same number of steps is taken, the end state is compared bit for bit.
    Decorated calls: the function is re-imported (or ``fn=``) and called with the recorded
    arguments. If the executed code differs from the recorded one, a :class:`CodeChangedWarning`
    says how (or an error with ``require_same_code=True``)."""
    inputs = run.rec.get("inputs") or {}
    inv = run.rec.get("invocation") or {}
    if inputs.get("kind") not in ("ivp", "function"):
        raise UnsupportedError(
            f"run {run.id} has no replayable input (recorded by a with-block or record())",
            fix=f"re-run the script: cd {inv.get('cwd')} && {' '.join(inv.get('argv') or [])} "
                f"at commit {str((run.rec.get('code') or {}).get('git_head'))[:10]}")
    recorded = run.rec.get("code_fingerprint") or {}
    now = code_fingerprint()
    code_diff = diff_dicts(recorded, now)
    code_same = not (code_diff["changed"] or code_diff["only_a"] or code_diff["only_b"])
    if not code_same:
        msg = (f"reproduce({run.id}): the executed code differs from the recorded one: "
               + ", ".join(f"{k} {str(a)[:12]} -> {str(b)[:12]}" for k, (a, b) in code_diff["changed"].items()))
        if require_same_code:
            raise OptionError(msg, fix="check out the recorded commit "
                              f"{str((run.rec.get('code') or {}).get('git_head'))[:10]}")
        warnings.warn(CodeChangedWarning(msg), stacklevel=2)
    rec_opts = dict(name=run.name, tags=sorted(set(run.tags) | {"reproduce"}), parent=run.id,
                    relation="reproduce", root=run.registry)
    if inputs["kind"] == "ivp":
        return _reproduce_ivp(run, solver, record, rec_opts, code_same, code_diff)
    if fn is None:
        mod, _, qual = inputs["target"].partition(":")
        if mod == "__main__":
            raise OptionError(f"reproduce({run.id}): {inputs['target']} was defined in a script",
                              fix="pass the function: run.reproduce(fn=my_function)")
        problem = _untrusted_record(run)
        if problem is not None:
            raise OptionError(
                f"reproduce({run.id}): the record names the function {inputs['target']!r} to "
                f"import and call, but {problem}",
                why="reproduce() imports the module and calls the function a run record names, "
                    "with the recorded arguments: a record that another account can write could "
                    "make it run any code (security)",
                fix="pass the function yourself: run.reproduce(fn=my_function); or, if the record "
                    "is yours, chmod go-w it and its directories")
        obj = importlib.import_module(mod)
        for part in qual.split("."):
            obj = getattr(obj, part)
        fn = obj
    if not inputs.get("args_exact"):
        raise UnsupportedError(f"reproduce({run.id}): some arguments were stored as summaries "
                               "(large arrays or objects) and cannot be replayed exactly")
    raw = inspect.unwrap(fn)
    call = _untracked(fn)                        # the user's decorators kept, track's removed
    src_sha = hashlib.sha256(inspect.getsource(raw).encode()).hexdigest()
    pos, kw = _call_args(raw, {k: from_jsonable(v) for k, v in inputs["args"].items()})
    if record:
        opts = {k: v for k, v in rec_opts.items() if k != "name"}
        out, new = call_recorded(call, pos, kw, name=run.name, _with_run=True, **opts)
    else:
        out, new = call(*pos, **kw), None
    ref = load_return(run)
    same, maxd = _same(ref, out)
    details = {"source_changed": src_sha != inputs.get("source_sha256")}
    return ReproduceReport(run, new, same, maxd, code_same, code_diff, details)


def _untrusted_record(run: Run) -> str | None:
    """Why the record of ``run`` may not choose code to import and call, or None: ``run.json``,
    its run directory and the registry's ``runs/`` directory must belong to this user and be
    writable by nobody else (the rule of the Studio control files)."""
    uid = getattr(os, "getuid", lambda: None)()
    for p in (run.path / "run.json", run.path, run.path.parent):
        try:
            st = os.stat(p)
        except OSError as err:
            return f"{p} cannot be checked ({err})"
        if uid is not None and st.st_uid != uid:
            return f"{p} is owned by uid {st.st_uid}, not by you (uid {uid})"
        if st.st_mode & 0o022:
            return f"{p} is writable by group or others (mode {st.st_mode & 0o777:o})"
    return None


#: numerics keys a 0.1.0 record does not hold (0.1.1, lot A: the canonical identity added them)
_ADDED_IN_011 = ("data", "functions", "undeclared", *(f"time_stepping.{k}" for k in (
    "time_parameter", "bc_time", "f_backend", "complexify", "decouple", "pivot")))


def _added_in_011(key: str) -> bool:
    return any(key == k or key.startswith(k + ".") for k in _ADDED_IN_011)


def _unrecorded_numerics(rec: Mapping[str, Any], d: Mapping[str, Any]) -> list[str]:
    """The numerics keys of this solver that a record written before 0.1.1 (no ``identity``)
    does not hold, when they are the only difference; [] otherwise (a newer record, or any
    other difference: refused as before)."""
    if rec.get("identity") is not None or d["changed"] or d["only_a"]:
        return []
    keys = list(d["only_b"])
    if not keys or not all(_added_in_011(k) for k in keys):
        return []
    return sorted({next(k2 for k2 in _ADDED_IN_011 if k == k2 or k.startswith(k2 + "."))
                   for k in keys})


def _call_args(fn: Callable, argsd: Mapping[str, Any]) -> tuple[tuple, dict]:
    """(positional, keyword) arguments rebuilding a call from its bound arguments by name
    (``inspect.BoundArguments.arguments``): positional-only parameters and ``*args`` are passed
    by position, ``**kwargs`` expanded (review finding 51: ``fn(**args)`` broke them)."""
    P = inspect.Parameter  # noqa: N806
    pos: list = []
    kw: dict = {}
    for name, p in inspect.signature(fn).parameters.items():
        if name not in argsd:
            continue
        v = argsd[name]
        if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD):
            pos.append(v)
        elif p.kind is P.VAR_POSITIONAL:
            pos.extend(v)
        elif p.kind is P.KEYWORD_ONLY:
            kw[name] = v
        else:
            kw.update(v)
    return tuple(pos), kw


def _same(a: Any, b: Any) -> tuple[bool, float]:
    """(bitwise identical, max abs difference) of two return values of the same layout.
    Bitwise means the bits: same type of container, same dtype and shape,
    same bytes (``-0.0`` is not ``+0.0``; a NaN equals only the same NaN)."""
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        pairs = [_same(a[k], b[k]) for k in a]
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        pairs = [_same(x, y) for x, y in zip(a, b, strict=True)]
        if type(a) is not type(b):
            pairs.append((False, 0.0))
    else:
        try:
            x, y = np.asarray(a), np.asarray(b)
        except (TypeError, ValueError):
            return a == b and type(a) is type(b), 0.0 if a == b else math.inf
        if x.shape != y.shape:
            return False, math.inf
        if x.dtype == object or y.dtype == object:
            eq = type(a) is type(b) and bool(a == b)
            return eq, 0.0 if eq else math.inf
        eq = (x.dtype == y.dtype and type(a) is type(b)
              and np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes())
        if x.size and x.dtype.kind in "biufc" and y.dtype.kind in "biufc":
            with np.errstate(invalid="ignore"):
                diff = np.abs(x.astype(np.result_type(x, y)) - y)
            d = 0.0 if eq else float(np.max(np.where(np.isnan(x) & np.isnan(y), 0.0, diff)))
        else:
            d = 0.0 if eq else math.inf
        return bool(eq), d
    return all(p[0] for p in pairs), max((p[1] for p in pairs), default=0.0)


def _reproduce_ivp(run: Run, solver: Any, record: bool, rec_opts: dict, code_same: bool,
                   code_diff: dict) -> ReproduceReport:
    from .. import provenance
    inputs = run.rec["inputs"]
    if solver is None:
        raise OptionError(f"reproduce({run.id}): pass the solver", why="the IR problem is not "
                          "rebuilt from JSON; your script builds it, the registry checks it",
                          fix="build the same problem and IVPSolver, then run.reproduce(solver)")
    if inputs.get("segments") != 1:
        raise UnsupportedError(f"reproduce({run.id}): {inputs.get('segments')} solver segments "
                               "recorded; only single-segment IVP runs are replayed")
    start, end = run.path / "state_start.npz", run.path / "state_end.npz"
    if not (start.exists() and end.exists()):
        raise UnsupportedError(f"reproduce({run.id}): start/end states were not stored (state=False)")
    info = problem_info(solver) or {}
    if info.get("hash") != (run.rec.get("problem") or {}).get("hash"):
        raise OptionError(f"reproduce({run.id}): the solver's problem is not the recorded one",
                          why=f"structural hash {str(info.get('hash'))[:12]} vs "
                              f"{str((run.rec.get('problem') or {}).get('hash'))[:12]}")
    d = diff_dicts(_clockless(run.rec.get("numerics") or {}),
                   _clockless(jsonable(provenance.numerics(solver))))
    unrecorded = _unrecorded_numerics(run.rec, d)
    if unrecorded:
        # a record written before 0.1.1 lacks what the identity added (given data, user
        # functions, six numerical choices): those cannot be compared, which is said, and the
        # bitwise comparison of the whole end state stays the verdict (identity review)
        from ..io._identity import IdentityWarning
        warnings.warn(IdentityWarning(
            f"reproduce({run.id}): the run was recorded before 0.1.1 and does not hold "
            + ", ".join(unrecorded) + ": those were not compared with this solver; the end state "
            "is still compared bit for bit"), stacklevel=3)
        d = dict(d, only_b={k: v for k, v in d["only_b"].items()
                            if not _added_in_011(k)})
    if d["changed"] or d["only_a"] or d["only_b"]:
        items = [f"{k}: {a} -> {b}" for k, (a, b) in d["changed"].items()] + \
                [f"{k} missing now" for k in d["only_a"]] + [f"{k} new" for k in d["only_b"]]
        raise OptionError(f"reproduce({run.id}): numerics differ from the recorded run: "
                          + "; ".join(items[:8]), why="reproduce uses the recorded numerical settings",
                          fix="build the solver with the recorded configuration (run.explain())")
    solver.load_checkpoint(start)
    new = ivp_run(solver, inputs["nsteps"], None, None, record=rec_opts, cache=False) if record \
        else solver.run(inputs["nsteps"])
    # the whole end state (review finding 24: X only was compared): X, every multistep history
    # array and the clock, bit for bit
    with np.load(end) as z:
        ref = {k: z[k] for k in z.files if k != "meta"}
        ref_meta = json.loads(str(z["meta"]))
    meta, arrays = solver._checkpoint_payload()
    arrays = {k: _to_host(v) for k, v in arrays.items()}
    differ, maxd = [], 0.0
    for k in sorted(set(ref) | set(arrays)):
        if k not in ref or k not in arrays:
            differ.append(f"{k} ({'absent from the record' if k not in ref else 'absent now'})")
            maxd = math.inf
            continue
        same_k, d = _same(ref[k], arrays[k])
        if not same_k:
            differ.append(k)
        maxd = max(maxd, d)
    for k in ("t", "iteration"):
        if k in ref_meta and ref_meta[k] != meta[k]:
            differ.append(f"{k} ({ref_meta[k]!r} recorded, {meta[k]!r} now)")
    return ReproduceReport(run, new, not differ, maxd, code_same, code_diff,
                           {"steps": inputs["nsteps"],
                            "compared": "end state: X, multistep history (" + ", ".join(
                                k for k in sorted(ref) if k != "X") + "), t, iteration",
                            "differ": ", ".join(differ) if differ else "nothing"})


# ------------------------------------------------------------------------------------ exports
def export(dest: str | os.PathLike, runs: Iterable[Run]) -> list[Path]:
    """Small versionable JSON per run (no arrays), to keep a results registry under version control."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for r in runs:
        rec = r.rec
        slim = {k: rec.get(k) for k in ("schema", "id", "name", "tags", "created_utc", "ended_utc",
                                        "duration_s", "state", "status", "status_history", "parent",
                                        "relation", "reused_from", "params", "results", "problem",
                                        "slurm", "cache_key")}
        code = rec.get("code") or {}
        slim["code"] = {k: code.get(k) for k in ("git_head", "code_matches_commit", "git_dirty_paths",
                                                 "python_tree_sha256", "warning")}
        slim["hardware"] = {k: (rec.get("hardware") or {}).get(k) for k in ("host", "gpu")}
        p = dest / f"{r.id}.json"
        atomic_write_json(p, slim)
        out.append(p)
    return out


def export_mlflow(runs: Iterable[Run], *, experiment: str | None = None,
                  tracking_uri: str | None = None) -> dict[str, str]:
    """Copy runs to MLflow (optional; ``mlflow`` imported here only, never installed by Conatus):
    parameters, numeric results as metrics, series as metric histories, tags, status, and the
    full record as an artifact. Returns {conatus id: mlflow run id}."""
    try:
        import mlflow
    except ImportError as err:
        raise MissingDependencyError("conatus.runs.export_mlflow needs mlflow",
                                     fix="pip install mlflow in your own environment (optional; "
                                         "the native registry needs nothing)") from err
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment is not None:
        mlflow.set_experiment(experiment)
    out = {}
    for r in runs:
        with mlflow.start_run(run_name=r.id) as mr:
            mlflow.log_params({k: str(v)[:250] for k, v in r.params.items()})
            mlflow.log_metrics({k: float(v) for k, v in r.results.items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool)})
            mlflow.set_tags({"conatus.id": r.id, "conatus.status": r.status, "conatus.state": r.state,
                             "conatus.problem_hash": str((r.rec.get("problem") or {}).get("hash")),
                             **{f"tag.{t}": "1" for t in r.tags}})
            mlflow.log_dict(r.rec, "conatus_run.json")
            out[r.id] = mr.info.run_id
    return out
