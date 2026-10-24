"""Non-blocking telemetry emitter used inside a computation.

The computation only ever does ``queue.put_nowait`` (records) or swaps a reference (frames). A
background thread writes the spool files. If the queue is full, the record is dropped and counted;
if the frame slot is overwritten before being written, the old frame is dropped and counted; if the
spool cannot be written (disk, permissions), the error is recorded and the computation continues.
Nothing here can raise into the computation after construction, and nothing blocks it.

Telemetry is always explicit: an object is created and passed (``IVPSolver.run(callback=...)``,
``NewtonKrylovHookstep.solve(on_iteration=...)``). There is no global or implicit telemetry.
"""
from __future__ import annotations

import json
import os
import queue
import re
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .._private import open_private, private_dir

from .protocol import (COMMANDS, FILM_ENCODING, PROTOCOL_VERSION, RESERVED_RUN_IDS, RUN_ID_RE, SCALES, STAT_KINDS,
                       VIEW_KINDS, check_controls, check_set, control_file_problem, encode_curve, encode_frame, encode_frame3d, encode_points, encode_vec2d,
                       film_record, record)

_STOP = object()
_RESERVED = ("v", "kind", "run_id", "wall", "name")    # fields every record already has
# scalars every ivp_callback record carries: a user scalar of that name would replace them silently
BUILTIN_SCALARS = ("t", "iteration", "dt", "wall_per_step", "gpu_mem_bytes")
# frame / curve names: part of a spool file name (<run_id>.frame.<name>.json), so no path separator,
# no NUL, no leading dot (a '/' made every later spool write fail, telemetry silent)
NAME_RE = re.compile(r"^[^./\\\x00][^/\\\x00]{0,127}$")


def check_name(name: Any, what: str = "frame") -> str:
    """A frame, view or statistic name usable in a spool file name; ValueError with the fix."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"{what} name {name!r} cannot name a spool file: 1-128 characters, no "
                         "'/', '\\' or NUL, not starting with '.' (e.g. 'u_x', 'E(k)')")
    return name
# Film thinning: at level j the film keeps every FILM_GROWTH**j-th frame of each name, and the level
# goes up each time half of the remaining budget is used: the film never
# exceeds its budget, and the time it covers grows geometrically with the level (each level covers
# about twice the duration of the previous one), so a run of any practical length is covered.
FILM_GROWTH = 4
FILM_POLICY = (f"every {FILM_GROWTH}**level-th frame of each name is kept; the level goes up by one "
               "each time half of the remaining budget is used; past the budget no frame is added "
               "(event film_full)")


def default_spool() -> Path:
    """Declared default spool: ``$CN_TELEMETRY``, else ``$CN_TMP/telemetry``, else the checkout's
    ``scratch/telemetry`` (development), else ``<user cache>/telemetry`` (installed package:
    ``$XDG_CACHE_HOME/conatus/telemetry``; never inside the Python environment).
    The rule lives in ``conatus._paths.telemetry_spool``."""
    from .._paths import telemetry_spool
    return telemetry_spool()


class Telemetry:
    """Emitter for one run. All knobs are explicit and shown by :meth:`explain`.

    capacity
        bounded record queue; beyond it records are dropped (counted)
    scalar_interval
        minimum wall time (s) between two evaluations of the *declared* scalars
        (t, iteration, dt, wall per step, GPU memory are always sent: they are free)
    frame_interval
        minimum wall time (s) between two frames of the same name
    max_px
        frames are strided down to at most this many points per axis
    control_interval
        minimum wall time (s) between two reads of the control file
    max_px3d
        3D frames are strided down to at most this many points per axis
    max_points
        particles / polylines frames carry at most this many points
    film_max_bytes
        budget of the film ``<run_id>.film.ndjson`` (frames kept for replay in the Studio);
        0 = no film. Thinning policy: :data:`FILM_POLICY`
    views, stats
        what the Studio may show, declared in the header (protocol VIEW_KINDS, STAT_KINDS);
        the Studio offers nothing else for this run
    controls
        parameters the Studio (or an agent) may change with the ``set`` command, declared in the
        header: ``[{"name": "dt", "min": 1e-4, "max": 1e-2}]`` (protocol ``check_controls``);
        each needs a setter in :meth:`ivp_callback` (``setters=``). Nothing else can be set.
    """

    def __init__(self, run_id: str, *, capacity: int, scalar_interval: float, frame_interval: float,
                 max_px: int,
                 spool: str | Path | None = None, control_interval: float = 1.0,
                 meta: Mapping[str, Any] | None = None, max_px3d: int = 48,
                 max_points: int = 16384, film_max_bytes: int = 0,
                 views: list[Mapping[str, Any]] | None = None,
                 stats: list[Mapping[str, Any]] | None = None,
                 header_extra: Mapping[str, Any] | None = None,
                 controls: list[Mapping[str, Any]] | None = None) -> None:
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id) or run_id in RESERVED_RUN_IDS:
            raise ValueError(f"run_id {run_id!r}: 1-200 characters among letters, digits, '.', "
                             "'_', '-', starting with a letter or digit (the relay and the hub "
                             f"refuse other ids; {', '.join(RESERVED_RUN_IDS)} is reserved), "
                             "e.g. 'kh_demo_2'")
        self.run_id = run_id
        from ..errors import check_count
        # counts are integers, never truncated (finding 66)
        self.capacity = check_count(capacity, "capacity", "Telemetry")
        self.max_px = check_count(max_px, "max_px", "Telemetry")
        self.frame_interval = float(frame_interval)
        self.scalar_interval = float(scalar_interval)
        self.max_px3d = check_count(max_px3d, "max_px3d", "Telemetry")
        self.max_points = check_count(max_points, "max_points", "Telemetry")
        self.views, self.stats = _check_views(views), _check_stats(stats)
        self.settable = check_controls(controls)     # declared settable parameters (or None)
        clash = {v["name"] for v in self.views or []} & {v["name"] for v in self.stats or []}
        if clash:          # frames, curves and the film share one namespace per run
            raise ValueError(f"names {sorted(clash)} are both a view and a statistic: frames and "
                             "curves of one run share their names; rename one of them")
        if isinstance(film_max_bytes, int) and film_max_bytes < 0:
            raise ValueError(f"film_max_bytes must be >= 0 (0 = no film), got {film_max_bytes}")
        self.film_max_bytes = check_count(film_max_bytes, "film_max_bytes", "Telemetry", minimum=0)
        # film state, touched by the writer thread only (counters read by explain/status)
        self.film_bytes = 0
        self.film_level = 0
        self.film_frames = 0
        self.film_skipped = 0            # frames that did not fit in the remaining budget
        self._film_seen: dict[str, int] = {}
        self._film_next: dict[str, int] = {}
        self._film_full = False
        self._scalar_last = -1e30
        self.control_interval = float(control_interval)
        self.spool = Path(spool) if spool is not None else default_spool()
        self.dropped_records = 0
        self.dropped_frames = 0
        self.written_records = 0
        self.error: str | None = None
        # how the run ended when it did not finish normally, sent with every later status (the
        # final one included): ``stopped`` by a Studio stop, ``failed`` (exception type and
        # message) by :meth:`fail`. ``error`` above is the telemetry's own failure.
        self.stopped: str | None = None
        self.failed: str | None = None
        self._q: queue.Queue = queue.Queue(maxsize=self.capacity)
        self._frames: dict[str, dict] = {}
        self._frame_last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._control_pos = 0
        self._control_last = 0.0
        self._control_problem: str | None = None
        self._last_step_wall: float | None = None
        self._last_step_it: int | None = None
        self.paused = False
        self._thread = threading.Thread(target=self._writer, name=f"telemetry-{run_id}", daemon=True)
        try:
            private_dir(self.spool)                 # 0700 when created here
        except OSError as err:                      # never fatal for the computation
            self.error = f"spool {self.spool} unavailable: {err}"
        # a run_id used again (a script re-run): the old film is moved aside, never appended to
        # (its frame numbers would restart at 0 and the file would exceed the budget)
        moved = None
        film_path = self.spool / f"{run_id}.film.ndjson"
        if self.film_max_bytes and film_path.exists():
            try:
                moved = film_path.with_name(f"{run_id}.film.previous.ndjson")
                os.replace(film_path, moved)
            except OSError as err:
                self.error = f"film {film_path} exists and cannot be moved aside: {err}"
                self.film_max_bytes = 0
        self._thread.start()
        head = dict(protocol=PROTOCOL_VERSION, host=socket.gethostname(), pid=os.getpid(),
                    slurm_job=os.environ.get("SLURM_JOB_ID"), settings=self.settings(),
                    meta=dict(meta or {}))
        if self.views is not None:
            head["views"] = self.views
        if self.stats is not None:
            head["stats"] = self.stats
        if self.settable is not None:
            head["controls"] = self.settable
        if self.film_max_bytes:
            head["film"] = {"file": f"{run_id}.film.ndjson", "max_bytes": self.film_max_bytes,
                            "policy": FILM_POLICY, "encoding": FILM_ENCODING}
        for k, v in dict(header_extra or {}).items():
            if k in head or k in _RESERVED:
                raise ValueError(f"header_extra field {k!r} clashes with a header field")
            head[k] = _deep_jsonable(v)
        self._put(record("header", run_id, **head))
        # a run_id used again: the commands sent to the previous run are never replayed (earlier versions: a
        # "stop" or a "set dt" left in the control file used to hit the new run at its first check)
        try:
            self._control_pos = (self.spool / f"{run_id}.control.jsonl").stat().st_size
        except OSError:
            self._control_pos = 0
        if self._control_pos:
            self.log(f"run_id {run_id!r} was used before: the {self._control_pos} bytes of commands "
                     "already in its control file are skipped", level="warning")
        if moved is not None:
            self.log(f"run_id {run_id!r} was used before: its film was moved to {moved.name} and "
                     "this run starts a new film", level="warning")
        elif self.error and "film" in self.error:
            self.log(f"no film for this run: {self.error}", level="warning")

    # ------------------------------------------------------------------ public API
    def settings(self) -> dict:
        """The sink's knobs as a dict (recorded in the header)."""
        return dict(capacity=self.capacity, scalar_interval=self.scalar_interval,
                    frame_interval=self.frame_interval, max_px=self.max_px,
                    control_interval=self.control_interval, spool=str(self.spool),
                    max_px3d=self.max_px3d, max_points=self.max_points,
                    film_max_bytes=self.film_max_bytes)

    def explain(self) -> str:
        """One-line summary of every knob and of the written/dropped counters."""
        return (f"Telemetry(run_id={self.run_id!r}, spool={self.spool}, protocol v{PROTOCOL_VERSION}, "
                f"queue capacity {self.capacity} records (drop when full), declared scalars every >= "
                f"{self.scalar_interval:g} s, frames every >= "
                f"{self.frame_interval:g} s strided to <= {self.max_px} px/axis and quantized to 8 bits, "
                f"3D frames strided to <= {self.max_px3d} px/axis (8 bits), point frames <= "
                f"{self.max_points} points (16 bits), "
                f"controls read every >= {self.control_interval:g} s; "
                + (f"film <= {self.film_max_bytes / 2**20:g} MiB ({FILM_POLICY}; level "
                   f"{self.film_level}, {self.film_frames} frames, {self.film_bytes} bytes, "
                   f"{self.film_skipped} skipped), " if self.film_max_bytes else "no film, ")
                + f"written {self.written_records}, "
                f"dropped {self.dropped_records} records / {self.dropped_frames} frames"
                + (f"; ERROR {self.error}" if self.error else "") + ")")

    def scalars(self, **values: Any) -> None:
        """Telemetry-plane scalars (floats); non-finite values are sent as strings, not hidden."""
        self._put(record("scalars", self.run_id, **self._payload(values, _jsonable)))

    def event(self, name: str, **data: Any) -> None:
        """Queue a named event with JSON-converted fields."""
        self._put(record("event", self.run_id, name=name, **self._payload(data, _deep_jsonable)))

    def _payload(self, data: Mapping[str, Any], conv: Callable[[Any], Any]) -> dict:
        """User fields of a record. A field named like a record field (v, kind, run_id, wall, name)
        would clash with it (earlier versions: a TypeError raised into the run): it is renamed ``user_<key>``
        and the renaming is logged, never dropped."""
        out = {}
        for k, v in data.items():
            if k in _RESERVED:
                self._put(record("log", self.run_id, level="warning",
                                 message=f"telemetry field {k!r} is reserved: sent as 'user_{k}'"))
                k = f"user_{k}"
            out[k] = conv(v)
        return out

    def log(self, message: str, level: str = "info") -> None:
        """Queue a log message at ``level``."""
        self._put(record("log", self.run_id, level=level, message=str(message)))

    def frame(self, name: str, data: Any, *, t: float | None = None, iteration: int | None = None,
              x: Any = None, y: Any = None, force: bool = False, scale: str | None = None,
              resample: str | None = None) -> bool:
        """Offer a 2D frame; returns False if throttled (frame_interval not elapsed).
        ``x``, ``y``: node positions (the extent is sent); ``scale``: declared display scale;
        ``resample``: text declaring a resampling done before (see ``encode_frame``)."""
        now = time.monotonic()
        if not force and now - self._frame_last.get(name, -1e30) < self.frame_interval:
            return False
        self._frame_last[name] = now
        try:
            fr = encode_frame(self.run_id, name, np.asarray(data), max_px=self.max_px, t=t,
                              iteration=iteration, x=x, y=y, scale=scale, resample=resample)
        except ValueError as err:                   # e.g. NaN field: reported, run continues
            self.log(f"frame {name!r} not sent: {err}", level="warning")
            return False
        self._offer(name, fr)
        return True

    def want_frame(self, name: str) -> bool:
        """Would a frame named ``name`` be sent now (``frame_interval`` elapsed)? Does not consume
        the slot: use it to compute an expensive diagnostic only when its frame will leave."""
        return time.monotonic() - self._frame_last.get(name, -1e30) >= self.frame_interval

    def frame3d(self, name: str, data: Any, *, extent: Any, scale: str, t: float | None = None,
                iteration: int | None = None, axes: tuple = ("x", "y", "z"), coords: Any = None,
                resample: str = "stride", force: bool = False) -> bool:
        """Offer a lossy 3D frame (protocol ``frame3d``; NumPy or CuPy data, strided on its
        device to ``max_px3d`` per axis, 8 bits). ``scale``: ``smoke`` | ``signed`` | ``dye``.
        Returns False if throttled or refused (refusals are logged, never raised)."""
        return self._encode_and_offer(name, force, lambda: encode_frame3d(
            self.run_id, name, data, max_px=self.max_px3d, extent=extent, scale=scale, axes=axes,
            coords=coords, resample=resample, t=t, iteration=iteration))

    def points3d(self, name: str, pts: Any, *, extent: Any, mode: str, values: Any = None,
                 t: float | None = None, iteration: int | None = None, force: bool = False) -> bool:
        """Offer particles ``(n, d)`` or polylines ``(n_lines, n_pts, d)`` (NaN = missing),
        protocol ``points3d``: at most ``max_points`` points (declared strides), 16-bit
        positions in ``extent``. ``mode``: particles | pathlines | streaklines | timelines."""
        return self._encode_and_offer(name, force, lambda: encode_points(
            self.run_id, name, pts, max_points=self.max_points, extent=extent, mode=mode,
            values=values, t=t, iteration=iteration))

    def vec2d(self, name: str, u: Any, v: Any, *, extent: Any, t: float | None = None,
              iteration: int | None = None, force: bool = False) -> bool:
        """Offer a lossy 2D vector field (protocol ``vec2d``, for LIC / streamlines):
        strided to ``max_px`` per axis, 8 bits on a symmetric scale."""
        return self._encode_and_offer(name, force, lambda: encode_vec2d(
            self.run_id, name, u, v, max_px=self.max_px, extent=extent, t=t, iteration=iteration))

    def curve(self, name: str, x: Any, y: Any, *, xlabel: str = "", ylabel: str = "",
              logx: bool = False, logy: bool = False, t: float | None = None,
              iteration: int | None = None, force: bool = False) -> bool:
        """Offer a 1D curve (protocol ``curve``: a spectrum, a profile), throttled like frames;
        at most 4096 points (declared stride). Refusals are logged, never raised."""
        return self._encode_and_offer(name, force, lambda: encode_curve(
            self.run_id, name, x, y, xlabel=xlabel, ylabel=ylabel, logx=logx, logy=logy,
            t=t, iteration=iteration))

    def _encode_and_offer(self, name: str, force: bool, encode: Callable[[], dict]) -> bool:
        now = time.monotonic()
        if not force and now - self._frame_last.get(name, -1e30) < self.frame_interval:
            return False
        self._frame_last[name] = now
        try:
            fr = encode()
        except Exception as err:  # noqa: BLE001 -- NaN, wrong shape, device error: logged
            self.log(f"frame {name!r} not sent: {type(err).__name__}: {err}", level="warning")
            return False
        self._offer(name, fr)
        return True

    def _offer(self, name: str, fr: dict) -> None:
        check_name(name)                            # raised to the caller, never in the writer
        with self._lock:
            if name in self._frames:
                self.dropped_frames += 1            # previous one never reached the disk
            self._frames[name] = fr

    def controls(self) -> list[dict]:
        """New commands from the Studio (throttled to control_interval). Unknown ones are logged."""
        now = time.monotonic()
        if now - self._control_last < self.control_interval:
            return []
        self._control_last = now
        path = self.spool / f"{self.run_id}.control.jsonl"
        try:
            with path.open("rb") as fh:
                problem = control_file_problem(os.fstat(fh.fileno()))
                fh.seek(self._control_pos)
                chunk = fh.read()
        except OSError:
            return []
        if not chunk.endswith(b"\n"):
            chunk = chunk[:chunk.rfind(b"\n") + 1]  # only complete lines
        self._control_pos += len(chunk)
        if problem is not None:                     # never applied, said once per problem
            if chunk and problem != self._control_problem:
                self._control_problem = problem
                self.log(f"control file {path.name} {problem}: its commands are ignored",
                         level="error")
            return []
        out = []
        for line in chunk.decode("utf-8", "replace").splitlines():
            try:
                cmd = json.loads(line)
            except ValueError:
                self.log(f"unreadable control line ignored: {line[:80]!r}", level="warning")
                continue
            if not isinstance(cmd, dict):           # a bare number or list crashed the run
                self.log(f"control line that is not an object ignored: {line[:80]!r}", level="warning")
                continue
            if cmd.get("cmd") not in COMMANDS:
                self.log(f"unknown command ignored: {cmd.get('cmd')!r}", level="warning")
                continue
            out.append(cmd)
        return out

    def close(self, timeout: float = 30.0) -> None:
        """Send the final status, stop the writer thread and wait up to ``timeout`` s. If the
        writer is still busy then (a slow shared file system), say so on stderr: the last
        records (the final status among them) may never reach the spool, and the Studio will
        show the run as stale rather than ended."""
        self.status(final=True)
        try:
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)
        if self._thread.is_alive():
            print(f"telemetry {self.run_id!r}: the spool writer did not finish within {timeout:g} s "
                  f"({self._q.qsize()} record(s) still queued, the final status among them): the "
                  "Studio may show this run as stale", file=sys.stderr, flush=True)

    def status(self, **extra: Any) -> None:
        """Queue a status record (counters, paused, error, plus ``extra``)."""
        film = ({"film_frames": self.film_frames, "film_bytes": self.film_bytes,
                 "film_level": self.film_level, "film_skipped": self.film_skipped}
                if self.film_max_bytes else {})
        self._put(record("status", self.run_id, written=self.written_records,
                         dropped_records=self.dropped_records, dropped_frames=self.dropped_frames,
                         paused=self.paused, error=self.error, **film,
                         **({"stopped": self.stopped} if self.stopped else {}),
                         **({"failed": self.failed} if self.failed else {}), **extra))

    def fail(self, err: BaseException) -> None:
        """Record that the run raised ``err``: an event ``failed`` (exception type and message) and
        a status; every later status, the final one included, carries ``failed``, so the Studio
        and the agent API show the run as failed (earlier versions recorded only a Studio stop:
        a run that diverged ended as "finished")."""
        kind, msg = type(err).__name__, str(err)
        self.failed = f"{kind}: {msg}"[:2000] if msg else kind
        self.event("failed", error_type=kind, message=msg[:2000])
        self.status()

    # ------------------------------------------------------------------ integrations
    def ivp_callback(self, *, scalars: Mapping[str, Callable[[Any], float]] | None = None,
                     frames: Mapping[str, Callable[[Any], Any]] | None = None,
                     checkpoint_dir: str | Path | None = None,
                     on_checkpoint: Callable[[Any, Path], Mapping[str, Any] | None] | None = None,
                     frames3d: Mapping[str, tuple[Callable[[Any], Any], Mapping[str, Any]]] | None = None,
                     curves: Mapping[str, tuple[Callable[[Any], Any], Mapping[str, Any]]] | None = None,
                     setters: Mapping[str, Callable[[Any, float], Mapping[str, Any] | None]] | None = None,
                     ) -> Callable[[Any], None]:
        """Callback for ``IVPSolver.run(nsteps, callback=cb, every=k)`` (or to call after each
        ``run_graph`` block). Always sends t, iteration, dt, wall time per step and GPU memory;
        user scalars and frames are **declared** callables of the solver (no implicit physics).
        Handles Studio commands: pause/resume, checkpoint (needs checkpoint_dir), stop (raises
        :class:`StopRun`), fork (recorded with a checkpoint; never changes the running run).
        ``on_checkpoint(solver, path)`` (optional, explicit) runs after each checkpoint written by
        a Studio command; the small JSON dict it returns is added to the ``checkpoint_written`` /
        ``fork_written`` event (e.g. near-recurrence period guesses for a Newton seed).
        If it raises, the checkpoint event is still sent and the error is logged.
        ``frames3d = {name: (f, opts)}``: ``f(solver)`` returns a 3D array (NumPy or CuPy),
        called **only** when the frame would be sent; ``opts`` are the keyword arguments of
        :meth:`frame3d` (``extent``, ``scale``, optionally ``axes``, ``coords``, ``resample``).
        A 2D frame may also be given as ``(f, opts)`` with ``opts`` the keyword arguments of
        :meth:`frame` (``x``, ``y``, ``scale``, ``resample``). ``curves = {name: (f, opts)}``:
        ``f(solver)`` returns ``(x, y)``, called only when the curve would be sent; ``opts``
        are those of :meth:`curve` (``xlabel``, ``ylabel``, ``logx``, ``logy``).
        ``setters = {name: f(solver, value)}``: one per declared control (``controls=``); a
        ``set`` command checked against the declared bounds calls it between two steps; it may
        refuse by raising (logged, event ``set_refused``) and may return a small dict added to
        the ``param_set`` event (e.g. the old value)."""
        setters = dict(setters or {})
        declared = {c["name"] for c in self.settable or []}
        if set(setters) != declared:
            raise ValueError(f"setters {sorted(setters)} must match the declared controls "
                             f"{sorted(declared)}: one setter per settable parameter")
        scalars, frames = dict(scalars or {}), dict(frames or {})
        frames3d, curves = dict(frames3d or {}), dict(curves or {})
        shadow = sorted(set(scalars) & set(BUILTIN_SCALARS))
        if shadow:
            raise ValueError(f"scalars {shadow} would replace the built-in {', '.join(BUILTIN_SCALARS)} "
                             "of every record; rename them (e.g. 'my_dt')")
        seen: set[str] = set()
        for n in [*frames, *frames3d, *curves]:
            check_name(n)
            if n in seen:
                raise ValueError(f"name {n!r} is given to two frames/curves: one namespace per run")
            seen.add(n)

        def cb(solver: Any) -> None:
            now, it = time.perf_counter(), int(solver.iteration)
            per_step = None
            if self._last_step_wall is not None and it > (self._last_step_it or 0):
                per_step = (now - self._last_step_wall) / (it - self._last_step_it)
            self._last_step_wall, self._last_step_it = now, it
            vals = dict(t=float(solver.t), iteration=it, dt=float(solver.dt), wall_per_step=per_step,
                        gpu_mem_bytes=_gpu_mem(solver))
            if scalars and time.monotonic() - self._scalar_last >= self.scalar_interval:
                self._scalar_last = time.monotonic()   # declared scalars may cost a host copy
                for k, f in scalars.items():
                    vals[k] = _safe(f, solver)
            self.scalars(**vals)
            for name, f in frames.items():
                f, opts = f if isinstance(f, tuple) else (f, {})
                if self.want_frame(name):
                    arr = _safe(f, solver)
                    if isinstance(arr, np.ndarray):
                        self.frame(name, arr, t=float(solver.t), iteration=it, **dict(opts))
                    else:                           # an "error: ..." string: shown, never raised
                        self._frame_last[name] = time.monotonic()
                        self.log(f"frame {name!r} not sent: {arr}", level="warning")
            for name, (f, opts) in curves.items():
                if self.want_frame(name):
                    try:                            # user callable: logged, never raised
                        x, y = f(solver)
                        self.curve(name, x, y, t=float(solver.t), iteration=it, **dict(opts))
                    except Exception as err:  # noqa: BLE001 -- user callable: logged, never raised
                        self._frame_last[name] = time.monotonic()
                        self.log(f"curve {name!r} failed: {type(err).__name__}: {err}",
                                 level="warning")
            for name, (f, opts) in frames3d.items():
                if self.want_frame(name):
                    try:                            # user callable / options: logged, never raised
                        self.frame3d(name, f(solver), t=float(solver.t), iteration=it, **dict(opts))
                    except Exception as err:  # noqa: BLE001 -- user callable: logged, never raised
                        self._frame_last[name] = time.monotonic()
                        self.log(f"frame3d {name!r} failed: {type(err).__name__}: {err}",
                                 level="warning")
            self._handle_controls(solver, checkpoint_dir, on_checkpoint, setters)
        return cb

    def newton_callback(self) -> Callable[[dict], None]:
        """For ``NewtonKrylovHookstep.solve(..., on_iteration=cb)``: one record per iteration."""
        def cb(info: dict) -> None:
            # the iteration's own 'kind' (newton | hook) is renamed: 'kind' is the record type
            data = {("step_kind" if k == "kind" else k): _jsonable(v) for k, v in info.items()}
            self._put(record("newton", self.run_id, **data))
        return cb

    def wrap_newton(self, newton: Any) -> Any:
        """A Newton solver that reports each iteration here; everything else is delegated.
        Use it where a function takes a ``newton=`` object (e.g. ``conatus.ecs`` searches)."""
        return _NewtonWithTelemetry(newton, self.newton_callback())

    # ------------------------------------------------------------------ internals
    def _handle_controls(self, solver: Any, checkpoint_dir: str | Path | None,
                         on_checkpoint: Callable[[Any, Path], Mapping[str, Any] | None] | None = None,
                         setters: Mapping[str, Callable] | None = None) -> None:
        """Apply the new Studio commands in order. While paused, the run waits here; checkpoint,
        fork and set received meanwhile are applied at once (earlier versions: they used to be dropped
        silently while paused), resume ends the wait, stop raises :class:`StopRun`."""
        pending = deque(self.controls())
        while pending:
            cmd = pending.popleft()
            c = cmd["cmd"]
            self.event("command_received", cmd=c, seq=cmd.get("seq"), by=cmd.get("from"))
            if c == "pause":
                if self.paused:
                    continue
                self.paused = True
                self.status()
                while self.paused:                  # explicit user request: blocking is intended
                    if not pending:
                        time.sleep(self.control_interval)
                        pending.extend(self.controls())
                        continue
                    nxt = pending.popleft()
                    n = nxt["cmd"]
                    if n in ("resume", "stop", "pause"):
                        self.event("command_received", cmd=n, seq=nxt.get("seq"), by=nxt.get("from"))
                    if n == "resume":
                        self.paused = False         # what follows is handled after resuming
                    elif n == "stop":
                        self.paused = False
                        self.stopped = "stopped from the Studio while paused"
                        raise StopRun(self.stopped)
                    elif n != "pause":              # already paused otherwise
                        self._apply(nxt, solver, checkpoint_dir, on_checkpoint, setters)
                self.event("resumed")
                self.status()
            elif c == "stop":
                self.stopped = "stopped from the Studio"
                raise StopRun(self.stopped)
            elif c == "resume":
                self.log("resume received while running: nothing to do", level="info")
            else:
                self._apply(cmd, solver, checkpoint_dir, on_checkpoint, setters, received=True)

    def _apply(self, cmd: dict, solver: Any, checkpoint_dir: str | Path | None,
               on_checkpoint: Callable | None, setters: Mapping[str, Callable] | None,
               received: bool = False) -> None:
        """One checkpoint, fork or set command (never raises into the run)."""
        c = cmd["cmd"]
        if not received:
            self.event("command_received", cmd=c, seq=cmd.get("seq"), by=cmd.get("from"))
        if c == "set":
            self._set(cmd, solver, setters or {})
            return
        if c not in ("checkpoint", "fork"):
            return
        if checkpoint_dir is None:
            self.log(f"{c} requested but no checkpoint_dir was declared; ignored", "warning")
            return
        path = Path(checkpoint_dir) / f"{self.run_id}_it{solver.iteration}.npz"
        try:
            path = Path(solver.save_checkpoint(path) or path)   # the solver may add .npz
            extra = {"params": cmd.get("params")} if c == "fork" else {}
        except Exception as err:  # noqa: BLE001 -- reported, the run goes on
            self.log(f"{c} failed: {err}", level="error")
            return
        if on_checkpoint is not None:       # user hook: never breaks the run
            try:
                more = {str(k): v for k, v in dict(on_checkpoint(solver, path) or {}).items()}
                # keys of the event itself (and the fork parameters) are not the hook's: a hook
                # returning {"t": ...} used to raise a TypeError (duplicate keyword) into the run
                clash = sorted(set(more) & ({"name", "path", "iteration", "t"} | set(extra)))
                if clash:
                    self.log(f"on_checkpoint keys {clash} clash with the {c}_written event fields; "
                             "left out (rename them)", "warning")
                extra.update({k: _deep_jsonable(v) for k, v in more.items() if k not in clash})
            except Exception as err:  # noqa: BLE001 -- user hook: logged
                self.log(f"on_checkpoint failed: {type(err).__name__}: {err}", "warning")
        self.event(f"{c}_written", path=str(path), iteration=int(solver.iteration),
                   t=float(solver.t), **extra)

    def _set(self, cmd: dict, solver: Any, setters: Mapping[str, Callable]) -> None:
        """A ``set`` command: checked against the declared controls, then the declared setter.
        Any refusal is an event ``set_refused`` with its reason (and a warning in the log)."""
        params = cmd.get("params")
        try:
            name, value = check_set(self.settable, params)
            more = setters[name](solver, value) or {}
        except Exception as err:  # noqa: BLE001 -- a refused setting is reported, the run goes on
            self.log(f"set refused: {err}", level="warning")
            self.event("set_refused", params=params, reason=str(err), seq=cmd.get("seq"),
                       by=cmd.get("from"), iteration=int(solver.iteration))
            return
        self.event("param_set", param=name, value=value, seq=cmd.get("seq"), by=cmd.get("from"),
                   iteration=int(solver.iteration), t=float(solver.t),
                   **{k: _deep_jsonable(v) for k, v in dict(more).items()})

    def _put(self, rec: dict) -> None:
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            self.dropped_records += 1

    def _writer(self) -> None:
        path = self.spool / f"{self.run_id}.jsonl"
        fh = None
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                item = None
            batch = [] if item is None else [item]
            while True:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            stop = any(b is _STOP for b in batch)
            batch = [b for b in batch if b is not _STOP]
            if self.error is None:
                try:
                    if batch:
                        if fh is None:
                            fh = open_private(path, "a")    # 0600 whatever the umask
                        fh.write("".join(json.dumps(b, separators=(",", ":")) + "\n" for b in batch))
                        fh.flush()
                        self.written_records += len(batch)
                    self._write_frames()
                except OSError as err:
                    self.error = f"spool write failed: {err}"
            if stop:
                if fh is not None:
                    fh.close()
                return

    def _write_frames(self) -> None:
        with self._lock:
            frames, self._frames = self._frames, {}
        for name, fr in frames.items():
            dst = self.spool / f"{self.run_id}.frame.{name}.json"
            tmp = dst.with_suffix(".json.tmp")
            with open_private(tmp, "w") as fh:
                fh.write(json.dumps(fr, separators=(",", ":")))
            os.replace(tmp, dst)                    # atomic for readers
            if self.film_max_bytes:
                self._film_append(name, fr)

    def _film_append(self, name: str, fr: dict) -> None:
        """Writer thread: keep the frame in the film if the thinning level and the remaining
        budget allow it (:data:`FILM_POLICY`). The computation never waits for this."""
        n = self._film_seen.get(name, 0)
        self._film_seen[name] = n + 1
        if n % FILM_GROWTH ** self.film_level:
            return                                  # thinned out at this level (declared)
        i = self._film_next.get(name, 0)
        line = (json.dumps(film_record(fr, i=i, level=self.film_level), separators=(",", ":"))
                + "\n").encode("utf-8")
        if self.film_bytes + len(line) > self.film_max_bytes:
            self.film_skipped += 1
            if not self._film_full:
                self._film_full = True
                self.event("film_full", t=fr.get("t"), iteration=fr.get("iteration"),
                           bytes=self.film_bytes, level=self.film_level)
            return
        with open_private(self.spool / f"{self.run_id}.film.ndjson", "ab") as fh:
            fh.write(line)
        self._film_next[name] = i + 1
        self.film_bytes += len(line)
        self.film_frames += 1
        while (self.film_level < 40 and
               self.film_bytes >= self.film_max_bytes * (1 - 0.5 ** (self.film_level + 1))):
            self.film_level += 1                    # half of the remaining budget used
            self.event("film_level", level=self.film_level, t=fr.get("t"),
                       keep_every=FILM_GROWTH ** self.film_level, bytes=self.film_bytes)


class _NewtonWithTelemetry:
    """Delegating proxy: ``solve(...)`` gets ``on_iteration`` unless the caller passed one."""

    def __init__(self, newton: Any, cb: Callable[[dict], None]) -> None:
        self._newton, self._cb = newton, cb

    def __getattr__(self, name: str) -> Any:
        return getattr(self._newton, name)

    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapped ``solve`` with ``on_iteration`` defaulting to the telemetry callback."""
        kwargs.setdefault("on_iteration", self._cb)
        return self._newton.solve(*args, **kwargs)


class StopRun(RuntimeError):
    """Raised inside the computation's callback when the Studio asks to stop the run."""


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, bool, int)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f if np.isfinite(f) else str(f)


def _deep_jsonable(v: Any) -> Any:
    """Nested lists/dicts of scalars (event payloads); non-finite floats become strings."""
    if isinstance(v, dict):
        return {str(k): _deep_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep_jsonable(x) for x in v]
    return _jsonable(v)


def _safe(f: Callable, solver: Any) -> Any:
    """User callables never break the run: failures become the string 'error: ...'."""
    try:
        v = f(solver)
    except Exception as err:  # noqa: BLE001 -- user callable: the error becomes the value shown
        return f"error: {type(err).__name__}: {err}"
    if hasattr(v, "get") and not isinstance(v, (dict, np.ndarray)):   # CuPy array -> host
        v = v.get()
    return v if isinstance(v, np.ndarray) else _jsonable(v)


def _gpu_mem(solver: Any) -> float | None:
    if getattr(solver, "backend", "cpu") != "cuda":
        return None
    try:
        import cupy
        return float(cupy.get_default_memory_pool().used_bytes())
    except Exception:  # noqa: BLE001 -- telemetry only: None means 'unknown', never 0
        return None


def _check_views(views: list[Mapping[str, Any]] | None) -> list[dict] | None:
    """Declared views (header field ``views``): known kinds and scales, unique names."""
    if views is None:
        return None
    out, seen = [], set()
    for v in views:
        v = dict(v)
        if v.get("kind") not in VIEW_KINDS:
            raise ValueError(f"view {v.get('name')!r}: kind must be one of {VIEW_KINDS}")
        if v.get("scale") is not None and v["scale"] not in SCALES:
            raise ValueError(f"view {v.get('name')!r}: scale must be one of {SCALES}")
        if not v.get("name") or v["name"] in seen:
            raise ValueError(f"views need unique non-empty names, got {v.get('name')!r}")
        check_name(v["name"], "view")
        seen.add(v["name"])
        out.append(_deep_jsonable(v))
    return out


def _check_stats(stats: list[Mapping[str, Any]] | None) -> list[dict] | None:
    """Declared statistics (header field ``stats``): known kinds, unique names."""
    if stats is None:
        return None
    out, seen = [], set()
    for v in stats:
        v = dict(v)
        if v.get("kind") not in STAT_KINDS:
            raise ValueError(f"stat {v.get('name')!r}: kind must be one of {STAT_KINDS}")
        if not v.get("name") or v["name"] in seen:
            raise ValueError(f"stats need unique non-empty names, got {v.get('name')!r}")
        check_name(v["name"], "statistic")
        seen.add(v["name"])
        out.append(_deep_jsonable(v))
    return out
