"""One run of the registry: live recorder (non-blocking writer thread) and read-only view.

A live :class:`Run` never raises into the computation after its construction and never waits for
the disk while the computation is running (spec §94–97, same rule as the telemetry):

* scalars and series are appended to in-memory lists (no I/O);
* fields, files and solver states are *copied* in the calling thread (a copy is needed anyway: the
  computation will overwrite its arrays) and handed to a bounded queue drained by one background
  thread. A full queue drops the item and counts it (``dropped``); nothing waits;
* the provenance that needs subprocesses (git, nvidia-smi) is collected by the writer thread;
* a disk error is recorded in ``run.errors`` and reported by a :class:`RegistryWriteWarning` at
  :meth:`Run.close`; the computation's numbers are untouched (tested bit-for-bit).

:meth:`Run.close` (end of a ``with`` block, of a decorated call, of ``solver.run(record=...)``)
waits for the writer at most ``CLOSE_TIMEOUT_S`` seconds: after the computation, never during it.
"""
from __future__ import annotations

import contextvars
import copy
import datetime as _dt
import json
import os
import platform
import queue
import secrets
import shutil
import socket
import sys
import threading
import time
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import OptionError
from .._explain import full_explain
from ._util import atomic_write_json, file_sha256, from_jsonable, jsonable, slug
from .store import SCHEMA, STATES, STATUSES, Registry

#: items (fields, files, states) waiting for the writer thread; beyond, dropped and counted
QUEUE_CAPACITY = 256
#: seconds :meth:`Run.close` waits for the writer thread (after the computation) before giving up
CLOSE_TIMEOUT_S = 600.0
#: HDF5 compressions a field may declare (all lossless); ``None`` = stored as is
COMPRESSIONS = (None, "gzip", "lzf")

_CURRENT: contextvars.ContextVar[Run | None] = contextvars.ContextVar("conatus_run", default=None)
_STOP = object()


class RegistryWriteWarning(UserWarning):
    """Part of a run's record could not be written (disk, permissions, full queue). The
    computation was not affected; ``run.errors`` / ``run.dropped`` say what is missing."""


def current() -> Run | None:
    """The run opened by the innermost ``with cn.runs.track(...)`` or decorated call, if any."""
    return _CURRENT.get()


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def problem_info(obj: Any) -> dict | None:
    """{hash, name, kind} of the problem behind a Problem, Plan or solver (None if there is none)."""
    from .. import provenance
    try:
        problem, _plan, _solver = provenance._unwrap(obj)
    except TypeError:
        return None
    if problem is None:
        return None
    return {"hash": provenance.problem_hash(problem), "name": problem.name, "kind": problem.kind}


def _hardware() -> dict:
    """Host and accelerator facts readable without a subprocess (nvidia-smi is in environment())."""
    return {"host": socket.gethostname(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "gpu": None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "slurm_gpus": os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE")}


def _slurm() -> dict:
    return {"job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"), "run_id": os.environ.get("CN_RUN_ID"),
            "output_dir": os.environ.get("CN_OUT")}


def _to_host(a: Any) -> np.ndarray:
    """A NumPy copy of ``a`` (CuPy arrays are copied to the host with ``.get()``)."""
    if type(a).__module__.split(".")[0] == "cupy":
        return a.get()
    return np.array(a, copy=True)


class Run:
    """A registry run. Created live by :func:`conatus.runs.track` / :func:`conatus.runs.record` /
    ``solver.run(record=...)``, or read back by :func:`conatus.runs.get` / ``find``.

    Live methods: :meth:`log` (series), :meth:`result` (scalar results), :meth:`save_field`,
    :meth:`save_file`, :meth:`attach` (a Problem / Plan / solver: numerics and ``explain()``),
    :meth:`certify` / :meth:`retract`, :meth:`close`. Read methods: :attr:`params`,
    :attr:`results`, :meth:`series`, :meth:`load_field`, :meth:`file_path`, :meth:`explain`,
    :meth:`reproduce`, :meth:`lineage`.
    """

    # ------------------------------------------------------------------ construction
    def __init__(self, registry: Registry, name: str, *, tags: Iterable[str] = (),
                 params: Mapping[str, Any] | None = None, obj: Any = None, parent: Any = None,
                 relation: str | None = None, status: str = "provisional", telemetry: Any = None,
                 capacity: int = QUEUE_CAPACITY, notes: str | None = None,
                 key_parts: dict | None = None, _record: dict | None = None) -> None:
        from ..errors import check_count
        # capacity=0 made the writer queue unbounded (queue.Queue(maxsize=0)), 2.5 was truncated
        check_count(capacity, "capacity", "conatus.runs")
        self.registry = registry
        self._lock = threading.Lock()
        self.errors: list[str] = []
        self.dropped = 0
        self._rows: list[tuple[int, int | None, dict]] = []
        self._thread: threading.Thread | None = None
        self._closed = _record is not None
        self._token: contextvars.Token | None = None
        self._key_parts = key_parts          # cache key parts; the code part is added at open
        if _record is not None:                       # read-only view of a stored run
            self.rec = _record
            self.id = _record["id"]
            return
        if not isinstance(name, str) or not name:
            raise OptionError("conatus.runs: a run needs a non-empty name", fix="track('rb_ra1e6', ...)")
        if status not in STATUSES:
            raise OptionError(f"conatus.runs: status {status!r} unknown; one of {STATUSES}")
        if isinstance(tags, str):
            tags = [tags]
        now = _dt.datetime.now(_dt.UTC)
        self.id = f"{now:%Y%m%d_%H%M%S}_{slug(name)}_{secrets.token_hex(3)}"
        self._t0 = time.perf_counter()
        self.rec: dict[str, Any] = {
            "schema": SCHEMA, "id": self.id, "name": name, "tags": sorted({str(t) for t in tags}),
            "created_utc": now.isoformat(), "ended_utc": None, "duration_s": None,
            "state": "running", "error": None, "notes": notes,
            "status": status, "status_history": [{"status": status, "reason": "created",
                                                  "evidence": None, "utc": now.isoformat(),
                                                  "by": os.environ.get("USER")}],
            "parent": getattr(parent, "id", parent), "relation": relation, "reused_from": None,
            "params": {str(k): jsonable(v) for k, v in (params or {}).items()},
            "problem": None, "numerics": {}, "explain": {}, "code": None, "code_fingerprint": None,
            "environment": None, "hardware": _hardware(), "slurm": _slurm(),
            "invocation": {"argv": list(sys.argv), "cwd": os.getcwd(), "pid": os.getpid(),
                           "python": sys.executable},
            "telemetry_id": getattr(telemetry, "run_id", telemetry),
            "cache_key": None, "cache": None, "inputs": None, "returns": None,
            "results": {}, "series": {}, "fields": {}, "files": [], "outputs": {},
            "writer": {"capacity": int(capacity), "dropped": 0, "errors": []},
        }
        if parent is not None and relation is None:
            raise OptionError("conatus.runs: parent= given without relation=",
                              why="a parent link means different things (fork, restart, reproduce)",
                              fix="relation='fork' | 'restart' | 'reproduce' | 'continuation' | ...")
        self._q: queue.Queue = queue.Queue(maxsize=int(capacity))
        if obj is not None:
            self.attach(obj)
        self._thread = threading.Thread(target=self._writer, name=f"runs-{self.id}", daemon=True)
        self._thread.start()
        self._put(("open",))

    @classmethod
    def load(cls, registry: Registry, run_id: str) -> Run:
        """Read-only view of a stored run."""
        return cls(registry, "", _record=registry.load(run_id))

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> Run:
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            _CURRENT.reset(self._token)
            self._token = None
        if exc_type is None:
            self.close()
        else:
            self.close(state="failed", error=f"{exc_type.__name__}: {exc}")
        return False                                   # never swallow the exception

    # ------------------------------------------------------------------ read properties
    @property
    def name(self) -> str:
        return self.rec["name"]

    @property
    def status(self) -> str:
        return self.rec["status"]

    @property
    def state(self) -> str:
        return self.rec["state"]

    @property
    def params(self) -> dict:
        return dict(self.rec.get("params") or {})

    @property
    def results(self) -> dict:
        return dict(self.rec.get("results") or {})

    @property
    def tags(self) -> list[str]:
        return list(self.rec.get("tags") or [])

    @property
    def parent(self) -> str | None:
        return self.rec.get("parent")

    @property
    def path(self) -> Path:
        return self.registry.run_dir(self.id)

    @property
    def cache_report(self) -> dict | None:
        return self.rec.get("cache")

    def __repr__(self) -> str:
        return f"<Run {self.id} {self.rec.get('state')} {self.rec.get('status')}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Run) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    # ------------------------------------------------------------------ live API
    def _live(self, what: str) -> None:
        if self._closed:
            raise OptionError(f"conatus.runs: {what} on a closed run {self.id}",
                              why="a closed record is only changed by certify / retract (history kept)",
                              fix="record into a new run (relation='fork' or 'restart', parent=this run)")

    def _put(self, item: tuple) -> bool:
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def attach(self, obj: Any) -> Run:
        """Record the problem hash, the numerical configuration (``conatus.provenance.numerics``)
        and the ``explain()`` text of a Problem, Plan or solver. The problem's parameter values (IR
        ``Parameter`` values, overridden by ``discretize(params=...)``) join the run parameters;
        a parameter declared twice with different values is refused."""
        self._live("attach")
        from .. import provenance
        info = problem_info(obj)
        num = jsonable(provenance.numerics(obj)) if info is not None else {}
        texts = {}
        for label, o in (("object", obj), ("plan", getattr(obj, "plan", None))):
            if o is not None and hasattr(o, "explain"):
                try:
                    texts[label] = str(full_explain(o))   # the complete record
                except Exception as err:  # noqa: BLE001 -- explain() is informative: failure recorded
                    texts[label] = f"explain() failed: {type(err).__name__}: {err}"
        # effective values: Parameter values of the IR, overridden by discretize(params=...)
        plan_params = {k: v for k, v in {**((num.get("problem") or {}).get("parameters") or {}),
                                        **((num.get("discretization") or {}).get("params") or {})}.items()
                       if v is not None}
        with self._lock:
            for k, v in plan_params.items():
                old = self.rec["params"].get(k)
                if old is not None and old != v:
                    raise OptionError(f"conatus.runs: parameter {k!r} declared as {old!r} but the "
                                      f"plan has {v!r}", fix="declare the value the plan uses")
                self.rec["params"][k] = v
            if info is not None:
                if self.rec["problem"] is not None and self.rec["problem"]["hash"] != info["hash"]:
                    self.rec.setdefault("other_problems", []).append(info)
                else:
                    self.rec["problem"] = info
                    self.rec["numerics"] = num
            self.rec["explain"].update(texts)
        return self

    def log(self, step: int | None = None, **scalars: Any) -> None:
        """Append one row of series values (``run.log(t=t, energy=E)``); ``step`` is optional (the
        row number is always kept, so series logged together can be plotted against each other).
        In-memory only; written to ``data.h5`` at close."""
        self._live("log")
        self._rows.append((len(self._rows), None if step is None else int(step),
                           {k: jsonable(v) for k, v in scalars.items()}))

    def result(self, **values: Any) -> None:
        """Scalar results (numbers, strings, booleans, complex): stored in ``run.json`` and
        indexed, so ``find(Nu__gt=...)`` works."""
        self._live("result")
        with self._lock:
            self.rec["results"].update({str(k): jsonable(v) for k, v in values.items()})

    def save_field(self, name: str, array: Any, *, compression: str | None, level: int | None = None,
                   t: float | None = None, step: int | None = None) -> bool:
        """Queue a copy of ``array`` for ``data.h5`` (``fields/<name>/<k>``). ``compression`` has no
        default (None, 'gzip' or 'lzf', all lossless; ``level`` 0–9 for gzip) and is written in the
        dataset's attributes. Returns False if the queue was full (dropped, counted)."""
        self._live("save_field")
        if compression not in COMPRESSIONS:
            raise OptionError(f"save_field: compression={compression!r} unknown; one of {COMPRESSIONS}",
                              why="storage choices are declared, never chosen for you (all are lossless)")
        if level is not None and compression != "gzip":
            raise OptionError("save_field: level= applies to gzip only")
        return self._put(("field", str(name), _to_host(array),
                          {"compression": compression, "level": level, "t": t, "step": step}))

    def save_file(self, path: str | os.PathLike, name: str | None = None) -> bool:
        """Queue a copy of a file into ``files/`` (sha256 recorded). The copy is taken by the
        writer thread: a file changed right after this call may be copied changed."""
        self._live("save_file")
        return self._put(("file", Path(path), name or Path(path).name))

    def _save_state(self, tag: str, solver: Any) -> None:
        """Queue a checkpoint of an IVPSolver in its own format (``state_<tag>.npz``, loadable by
        ``solver.load_checkpoint``): same fields as ``IVPSolver.save_checkpoint``."""
        meta = dict(scheme=solver.scheme, dt=solver.dt, startup=solver.startup,
                    startup_substeps=solver.startup_substeps, n=solver.system.n,
                    iteration=solver.iteration, t=solver.t, t_origin=solver._t0,
                    it_origin=solver._it0, reduced=solver.system.reduced,
                    ordering=solver.system.ordering, problem=solver.plan.problem.name,
                    nhist=len(solver._MXh))
        from ..io._identity import checkpoint_identity       # compared at restart (Y2)
        meta["identity"] = checkpoint_identity(solver)
        arrays = {"X": _to_host(solver.X)}
        for i, (mx, f) in enumerate(zip(solver._MXh, solver._Fh, strict=True)):
            arrays[f"MX{i}"] = _to_host(mx)
            arrays[f"F{i}"] = _to_host(f)
        if self._put(("npz", f"state_{tag}.npz", meta, arrays)):
            with self._lock:
                self.rec["outputs"][f"state_{tag}"] = f"state_{tag}.npz"

    def set_status(self, status: str, reason: str, evidence: Any = None) -> Run:
        """Change the status (history kept, never rewritten). ``reason`` is required."""
        if status not in STATUSES:
            raise OptionError(f"conatus.runs: status {status!r} unknown; one of {STATUSES}")
        if not isinstance(reason, str) or not reason.strip():
            raise OptionError(f"conatus.runs: status '{status}' needs a reason",
                              why="a status without its evidence cannot be audited",
                              fix="e.g. certify('residual 3e-13, matches Orszag 1971 to 1e-8')")
        ev = jsonable(evidence)
        if not self._closed:
            with self._lock:
                self.rec["status"] = status
                self.rec["status_history"].append({"status": status, "reason": reason, "evidence": ev,
                                                   "utc": _utc(), "by": os.environ.get("USER")})
            return self
        self.rec = self.registry.update_status(self.id, status, reason, ev, _utc())
        return self

    def certify(self, reason: str, evidence: Any = None) -> Run:
        """Mark the run ``certified`` (reason and optional evidence, e.g. residuals, recorded)."""
        return self.set_status("certified", reason, evidence)

    def retract(self, reason: str) -> Run:
        """Mark the run ``retracted`` (kept, never deleted; never reused by the cache)."""
        return self.set_status("retracted", reason)

    def close(self, state: str = "completed", error: str | None = None,
              timeout: float = CLOSE_TIMEOUT_S) -> Run:
        """Finish the run: series to ``data.h5``, final ``run.json``, index. Waits for the writer at
        most ``timeout`` s. Idempotent. Warns (:class:`RegistryWriteWarning`) if anything was lost."""
        if self._closed:
            return self
        if state not in STATES:
            raise OptionError(f"conatus.runs: state {state!r} unknown; one of {STATES}")
        self._closed = True
        with self._lock:
            self.rec["state"] = state
            self.rec["error"] = error
            self.rec["ended_utc"] = _utc()
            self.rec["duration_s"] = round(time.perf_counter() - self._t0, 6)
        try:
            self._q.put(("finalize",), timeout=timeout)
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:
            self.errors.append("writer queue still full at close: record not finalized")
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                self.errors.append(f"writer thread still busy after {timeout:g} s at close")
        if self.errors or self.dropped:
            warnings.warn(RegistryWriteWarning(
                f"conatus.runs: run {self.id} recorded incompletely under {self.registry.root}: "
                f"{len(self.errors)} error(s), {self.dropped} item(s) dropped; first: "
                f"{self.errors[0] if self.errors else 'queue full'}. The computation is unaffected."),
                stacklevel=2)
        return self

    # ------------------------------------------------------------------ writer thread
    def _snapshot(self) -> dict:
        with self._lock:
            rec = copy.deepcopy(self.rec)
        rec["writer"] = {"capacity": self._q.maxsize, "dropped": self.dropped,
                         "errors": list(self.errors)}
        return rec

    def _writer(self) -> None:
        """Drain the queue; every failure is recorded, never raised."""
        h5 = None
        broken = False
        d = self.path
        while True:
            item = self._q.get()
            if item is _STOP:
                break
            kind = item[0]
            try:
                if kind == "open":
                    from .. import provenance
                    from .keys import code_fingerprint, finish_key
                    code, env = provenance.code_state(), provenance.environment()
                    fp = code_fingerprint(code)
                    with self._lock:
                        self.rec["code"] = jsonable(code)
                        self.rec["environment"] = jsonable(env)
                        self.rec["hardware"]["gpu"] = env.get("gpu")
                        self.rec["code_fingerprint"] = fp
                        if self._key_parts is not None and self.rec["cache_key"] is None:
                            self.rec["cache_key"] = finish_key(self._key_parts, fp)[0]
                    d.mkdir(parents=True, exist_ok=False)
                    atomic_write_json(d / "run.json", self._snapshot())
                    self._index()
                    continue
                if broken:
                    continue
                if kind == "field":
                    _, name, arr, meta = item
                    h5 = h5 or _h5_open(d / "data.h5.part")
                    grp = h5.require_group(f"fields/{name}")
                    k = len(grp)
                    kw = {} if meta["compression"] is None else {
                        "compression": meta["compression"],
                        **({"compression_opts": meta["level"]} if meta["level"] is not None else {})}
                    ds = grp.create_dataset(str(k), data=arr, **kw)
                    for a in ("t", "step"):
                        if meta[a] is not None:
                            ds.attrs[a] = meta[a]
                    ds.attrs["compression"] = str(meta["compression"])
                    with self._lock:
                        f = self.rec["fields"].setdefault(name, {"count": 0})
                        f.update(count=k + 1, shape=list(arr.shape), dtype=str(arr.dtype),
                                 compression=meta["compression"], level=meta["level"])
                elif kind == "file":
                    _, src, name = item
                    (d / "files").mkdir(exist_ok=True)
                    shutil.copy2(src, d / "files" / name)
                    entry = {"name": name, "source": str(src), "bytes": (d / "files" / name).stat().st_size,
                             "sha256": file_sha256(d / "files" / name)}
                    with self._lock:
                        self.rec["files"].append(entry)
                elif kind == "npz":
                    _, fname, meta, arrays = item
                    tmp = d / f".{fname}.part"
                    with open(tmp, "wb") as fh:
                        np.savez(fh, meta=json.dumps(meta), **arrays)
                    os.replace(tmp, d / fname)
                elif kind == "finalize":
                    if self._rows:
                        h5 = h5 or _h5_open(d / "data.h5.part")
                        self._write_series(h5)
                    if h5 is not None:
                        h5.close()
                        h5 = None
                        os.replace(d / "data.h5.part", d / "data.h5")
                    atomic_write_json(d / "run.json", self._snapshot())
                    self._index()
            except Exception as err:  # noqa: BLE001 -- never raised into the computation; recorded
                self.errors.append(f"{kind}: {type(err).__name__}: {err}")
                if kind == "open":
                    broken = True                    # no directory: skip the other disk writes
                if kind == "finalize" and not broken:
                    try:                             # last chance: the record itself
                        atomic_write_json(d / "run.json", self._snapshot())
                    except Exception as err2:  # noqa: BLE001 -- recorded (was silently dropped)
                        self.errors.append(f"finalize (run.json retry): {type(err2).__name__}: {err2}")
        if h5 is not None:
            try:
                h5.close()
            except Exception as err:  # noqa: BLE001 -- recorded (was silently dropped)
                self.errors.append(f"close data.h5: {type(err).__name__}: {err}")

    def _index(self) -> None:
        try:
            self.registry.index(self._snapshot())
        except Exception as err:  # noqa: BLE001 -- the directory is the truth: sync() repairs
            with self._lock:
                self.rec.setdefault("index_note", f"index not updated at write ({err}); "
                                                  "reconciled by the next query")

    def _write_series(self, h5: Any) -> None:
        names: dict[str, list] = {}
        for row, step, vals in self._rows:
            for k, v in vals.items():
                names.setdefault(k, []).append((row, -1 if step is None else step, v))
        summary = {}
        for k, pts in names.items():
            rows = np.array([p[0] for p in pts], dtype=np.int64)
            steps = np.array([p[1] for p in pts], dtype=np.int64)
            vals = [p[2] for p in pts]
            try:
                data = np.array([from_jsonable(v) for v in vals], dtype=float)
            except (TypeError, ValueError):
                data = np.array([json.dumps(v) for v in vals], dtype=object)
            g = h5.require_group(f"series/{k}")
            g.create_dataset("row", data=rows)
            g.create_dataset("step", data=steps)
            if data.dtype == object:
                import h5py
                g.create_dataset("value", data=data.astype(str), dtype=h5py.string_dtype())
            else:
                g.create_dataset("value", data=data)
            summary[k] = {"n": len(pts), "last": vals[-1]}
        with self._lock:
            self.rec["series"] = summary

    # ------------------------------------------------------------------ reading
    def _h5(self):
        p = self.path / "data.h5"
        if not p.exists():
            raise OptionError(f"conatus.runs: run {self.id} has no data.h5",
                              why="no series or field was recorded (or the run is still running)")
        return _h5_open(p, "r")

    def series(self, name: str, x: str | None = None) -> dict:
        """{"row", "step", "value"} arrays of one series; with ``x`` (another series logged in the
        same rows, e.g. "t"), also "x" aligned on the rows both have (no interpolation)."""
        with self._h5() as h5:
            if f"series/{name}" not in h5:
                raise OptionError(f"conatus.runs: run {self.id} has no series {name!r}",
                                  fix=f"series: {sorted(h5.get('series', {}).keys())}")
            g = h5[f"series/{name}"]
            out = {k: g[k][()] for k in ("row", "step", "value")}
            if x is not None:
                gx = h5[f"series/{x}"]
                xr, xv = gx["row"][()], gx["value"][()]
                common, ia, ib = np.intersect1d(out["row"], xr, return_indices=True)
                out = {"row": common, "step": out["step"][ia], "value": out["value"][ia], "x": xv[ib]}
        return out

    def load_field(self, name: str, index: int = -1) -> np.ndarray:
        """One saved snapshot of a field (default: the last one)."""
        with self._h5() as h5:
            if f"fields/{name}" not in h5:
                raise OptionError(f"conatus.runs: run {self.id} has no field {name!r}",
                                  fix=f"fields: {sorted(h5.get('fields', {}).keys())}")
            g = h5[f"fields/{name}"]
            keys = sorted(g.keys(), key=int)
            return g[keys[index]][()]

    def file_path(self, name: str) -> Path:
        p = self.path / "files" / name
        if not p.exists():
            raise OptionError(f"conatus.runs: run {self.id} has no file {name!r}",
                              fix=f"files: {[f['name'] for f in self.rec.get('files', [])]}")
        return p

    def refresh(self) -> Run:
        """Re-read ``run.json`` (e.g. after another process changed the status)."""
        self.rec = self.registry.load(self.id)
        return self

    def lineage(self):
        from .analysis import lineage
        return lineage(self, registry=self.registry)

    def reproduce(self, solver: Any = None, **kw: Any):
        """Run this computation again and compare (see :func:`conatus.runs.reproduce`)."""
        from .integrations import reproduce
        return reproduce(self, solver, **kw)

    def explain(self) -> str:
        r = self.rec
        code = r.get("code") or {}
        pb = r.get("problem") or {}
        lines = [f"Run {self.id} ({r.get('name')}): state {r.get('state')}, status {r.get('status')}"
                 + (f", {r['duration_s']:.3g} s" if r.get("duration_s") is not None else ""),
                 f"  problem: {pb.get('name')} [{pb.get('kind')}] hash {str(pb.get('hash'))[:12]}"
                 if pb else "  problem: none attached",
                 f"  code: {str(code.get('git_head'))[:10]}"
                 f"{'' if code.get('code_matches_commit') else ' (NOT exactly a commit: ' + str(code.get('warning', 'see code block')) + ')'}"
                 if code else "  code: not recorded (writer failed or still running)",
                 f"  host {r['hardware'].get('host')}, gpu {r['hardware'].get('gpu')}, slurm job "
                 f"{r['slurm'].get('job_id')} run {r['slurm'].get('run_id')}"]
        if r.get("parent"):
            lines.append(f"  parent {r['parent']} ({r.get('relation')})")
        if r.get("reused_from"):
            lines.append(f"  CACHE: nothing computed, reused {r['reused_from']}")
        if r.get("params"):
            lines.append("  params: " + ", ".join(f"{k}={v}" for k, v in sorted(r["params"].items())))
        if r.get("results"):
            lines.append("  results: " + ", ".join(f"{k}={v}" for k, v in sorted(r["results"].items())))
        if r.get("series"):
            lines.append("  series: " + ", ".join(f"{k} ({v['n']})" for k, v in sorted(r["series"].items())))
        if r.get("fields"):
            lines.append("  fields: " + ", ".join(f"{k} {v.get('shape')} x{v['count']} [{v.get('compression')}]"
                                                  for k, v in sorted(r["fields"].items())))
        for h in r.get("status_history", [])[1:]:
            lines.append(f"  status -> {h['status']} ({h['utc'][:19]}): {h['reason']}")
        w = r.get("writer") or {}
        if w.get("errors") or w.get("dropped"):
            lines.append(f"  WRITER: {w.get('dropped')} dropped, errors {w.get('errors')}")
        if r.get("error"):
            lines.append(f"  error: {r['error']}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html
        return "<pre>" + html.escape(self.explain()) + "</pre>"


def _h5_open(path: Path, mode: str = "a"):
    """h5py file without HDF5 file locking (one writer per file by construction; locking is
    unreliable on network file systems)."""
    import h5py
    try:
        return h5py.File(path, mode, locking=False)
    except TypeError:                                  # h5py built without the locking argument
        return h5py.File(path, mode)
