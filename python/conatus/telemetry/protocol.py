"""Conatus telemetry protocol, version 1 (spec).

Two planes only: the **telemetry plane** (small JSON records: scalars, events, logs,
status) and the **visualization plane** (lossy 2D frames, latest value only). The scientific data
plane (checkpoints, snapshots) never goes through here.

Transport v1 is a *spool directory* on the shared file system, written by the computation and read
by the Studio server (see :mod:`conatus.telemetry.reader`):

* ``<run_id>.jsonl``            — one JSON record per line, appended (telemetry plane);
* ``<run_id>.frame.<name>.json`` — latest frame of ``name``, replaced atomically (visualization plane;
  kinds ``frame`` (2D), and since ``frame3d``, ``points3d``, ``vec2d``: see FRAME_KINDS);
* ``<run_id>.control.jsonl``    — commands written by the Studio, read by the computation;
* ``<run_id>.film.ndjson``      — the film: frames kept for replay, compact encoding, bounded
  budget (see :func:`film_record`; the name does not match ``*.jsonl`` on purpose: older
  readers and relays never mistake it for the record file).

Files decouple the two sides completely: a Studio that is slow, absent or crashed has no effect on
the run, and a run can be followed from any node that mounts the spool.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import zlib
from typing import Any

import numpy as np

try:
    from ..errors import OptionError, check_count
except ImportError:      # the Studio hub image ships the protocol without conatus.errors
    class OptionError(ValueError):  # type: ignore[no-redef]
        """Stand-in of conatus.errors.OptionError (same base class, message only)."""

        def __init__(self, message: str = "", **_kw: Any) -> None:
            super().__init__(message)

    def check_count(value: Any, name: str, where: str, minimum: int = 1) -> int:  # type: ignore[misc]
        """Stand-in of conatus.errors.check_count."""
        import numbers
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
            raise OptionError(f"{where}: {name}={value!r} must be an integer >= {minimum}")
        return int(value)

PROTOCOL_VERSION = 1
# run ids: file names in the spool, keys on the relay and the hub (one rule for all of them)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
# The spool audit of the agent channel (``conatus.studio.client.SpoolClient``) also ends in
# ``.jsonl``: it is not a run file, and its stem is not a valid run id.
SPOOL_AUDIT = "studio_audit.jsonl"
RESERVED_RUN_IDS = (SPOOL_AUDIT[:-len(".jsonl")],)


def run_file_id(name: str) -> str | None:
    """The run id of a spool record file ``<run_id>.jsonl``; None for a control file, the spool
    audit or a name that is not a run id. The one rule of the readers (local server, relay,
    spool client): the audit used to be read as a run "studio_audit", and the relay pushed it
    (user names, commands) to the hub as telemetry."""
    if not name.endswith(".jsonl") or name.endswith(".control.jsonl"):
        return None
    run_id = name[:-len(".jsonl")]
    if run_id in RESERVED_RUN_IDS or not RUN_ID_RE.match(run_id):
        return None
    return run_id

# Record kinds of the telemetry plane. Every record carries: v, kind, run_id, wall (UNIX time).
KINDS = ("header", "scalars", "event", "log", "status", "newton")
# Frame kinds of the visualization plane (one file per name, latest value only). "frame" is the
# 2D frame of v1; frame3d, points3d, vec2d were added for (3D view) and ``curve`` (1D: spectra,
# profiles) for the Studio rebuild; older clients ignore the kinds they do not know.
FRAME_KINDS = ("frame", "frame3d", "points3d", "vec2d", "curve")
# Declared display scales of a 3D frame (brand charter, section III "Data").
SCALES = ("smoke", "signed", "dye")
POINT_MODES = ("particles", "pathlines", "streaklines", "timelines")
# Views and statistics a run declares in its header (``Telemetry(views=..., stats=...)``, the native
# ``Simulation.watch``): the Studio offers only what is declared (no hidden physics).
VIEW_KINDS = ("film2d", "volume3d", "particles", "flow2d")
STAT_KINDS = ("series", "spectrum", "profile")
# Film encoding of the 8-bit frames: PNG-style "Up" filter (row minus previous row, mod 256) along
# the last axis, then zlib deflate (RFC 1950, what a browser's DecompressionStream("deflate") reads).
FILM_ENCODING = "uint8-minmax-up-deflate-base64"
# Commands of the control file (Studio -> computation). "fork" is recorded, never applied in place.
# "set" changes one parameter the run *declared* settable (header ``controls``), within its declared
# bounds; any other name or value is refused by the hub, the relay and the run itself.
COMMANDS = ("pause", "resume", "checkpoint", "stop", "fork", "set")
# Commands that need a confirmation (two steps): they end the run or start another one.
CONFIRM_COMMANDS = ("stop", "fork")
# Kinds of declared settable parameters (header ``controls``): a real number in [min, max].
CONTROL_KINDS = ("number",)
_CONTROL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")


def check_controls(controls: Any) -> list[dict] | None:
    """Declared settable parameters, ``[{name, kind: "number", min, max, label?, unit?, doc?}]``:
    unique names, finite bounds with min < max. ValueError with the fix otherwise."""
    if controls is None:
        return None
    out, seen = [], set()
    for c in controls:
        c = dict(c)
        name = c.get("name")
        if not isinstance(name, str) or not _CONTROL_NAME.match(name):
            raise OptionError(f"control {name!r}: a name of letters, digits, '_' or '.' (e.g. 'dt')")
        if name in seen:
            raise OptionError(f"control {name!r} declared twice")
        if c.get("kind", "number") not in CONTROL_KINDS:
            raise OptionError(f"control {name!r}: kind must be one of {CONTROL_KINDS}")
        try:
            lo, hi = float(c["min"]), float(c["max"])
        except (KeyError, TypeError, ValueError):
            raise OptionError(f"control {name!r}: declare its bounds, min and max (numbers)") from None
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise OptionError(f"control {name!r}: bounds must be finite with min < max, got "
                             f"[{lo}, {hi}]")
        seen.add(name)
        out.append({**{k: str(v) for k, v in c.items() if k in ("label", "unit", "doc")},
                    "name": name, "kind": "number", "min": lo, "max": hi})
    return out


def check_set(controls: Any, params: Any) -> tuple[str, float]:
    """Validate the parameters of a ``set`` command against the declared controls of a run:
    ``{"name": declared name, "value": finite number in [min, max]}``. Returns (name, value);
    ValueError with a message for the sender otherwise (nothing is ever clipped or rounded)."""
    if not isinstance(params, dict) or set(params) - {"name", "value"}:
        raise OptionError('set needs params {"name": <declared control>, "value": <number>}')
    decl = {c.get("name"): c for c in (controls or []) if isinstance(c, dict)}
    name = params.get("name")
    if name not in decl:
        raise OptionError(f"{name!r} is not declared settable by this run (settable: "
                         f"{sorted(decl) or 'none'})")
    v = params.get("value")
    try:                    # a JSON integer of 400 digits overflows float(): refused like inf
        finite = not isinstance(v, bool) and isinstance(v, (int, float)) and np.isfinite(float(v))
    except OverflowError:
        finite = False
    if not finite:
        raise OptionError(f"set {name}: the value must be a finite number, got {str(v)[:40]}")
    lo, hi = float(decl[name]["min"]), float(decl[name]["max"])
    if not lo <= float(v) <= hi:
        raise OptionError(f"set {name}={v!r}: outside the declared bounds [{lo:g}, {hi:g}] "
                         "(never clipped)")
    return str(name), float(v)


def append_control(path: Any, line: dict) -> None:
    """Append one command line to a run's control file, creating it with mode 0600: the control
    file is the only way into a run, so only the run's own user may write it (the run refuses a
    control file owned by someone else or writable by group/others, :func:`control_file_problem`)."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def control_file_problem(st: os.stat_result) -> str | None:
    """Why the commands of a control file (its ``os.fstat``) must not be applied, or None: it
    must belong to the user of the run and be writable by nobody else. Otherwise another account
    of the cluster could pause, stop or re-time the run."""
    uid = getattr(os, "getuid", lambda: None)()
    if uid is not None and st.st_uid != uid:
        return f"owned by uid {st.st_uid}, not by this run's user (uid {uid})"
    if st.st_mode & 0o022:
        return f"writable by group or others (mode {st.st_mode & 0o777:o}; chmod go-w it)"
    return None


def record(kind: str, run_id: str, **payload: Any) -> dict:
    """One telemetry-plane record. Unknown kinds are refused (no silent protocol drift)."""
    if kind not in KINDS:
        raise OptionError(f"unknown telemetry record kind {kind!r}; protocol v{PROTOCOL_VERSION} "
                         f"knows {KINDS}")
    return {"v": PROTOCOL_VERSION, "kind": kind, "run_id": run_id, "wall": time.time(), **payload}


def encode_frame(run_id: str, name: str, data: np.ndarray, *, max_px: int, t: float | None = None,
                 iteration: int | None = None, x: np.ndarray | None = None,
                 y: np.ndarray | None = None, scale: str | None = None,
                 resample: str | None = None) -> dict:
    """Lossy 2D frame: strided down to at most ``max_px`` per axis, quantized to 8 bits between
    its min and max (both sent), base64-encoded. Everything lossy is stated in the frame.
    ``scale`` (optional, ``smoke`` | ``signed`` | ``dye``) is the declared display scale;
    ``resample`` (optional text) declares a resampling done by the caller before encoding."""
    if scale is not None and scale not in SCALES:
        raise OptionError(f"frame {name!r}: scale must be one of {SCALES}, got {scale!r}")
    # audit inputs: max_px=0 was a ZeroDivisionError, 4.5 accepted; complex data lost
    # their imaginary part (np.asarray(..., float)); every protocol refusal is now typed
    max_px = check_count(max_px, "max_px", f"frame {name!r}")
    if np.iscomplexobj(data):
        raise OptionError(f"frame {name!r}: complex values (a picture is of real values)",
                          fix="send np.real(...), np.abs(...) or np.angle(...) explicitly")
    a = np.asarray(data, dtype=float)
    if a.ndim != 2:
        raise OptionError(f"frame {name!r}: a 2D array is required, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        n_bad = int(np.size(a) - np.count_nonzero(np.isfinite(a)))
        raise OptionError(f"frame {name!r}: {n_bad} non-finite values (not sent as a picture)")
    sx = max(1, int(np.ceil(a.shape[0] / max_px)))
    sy = max(1, int(np.ceil(a.shape[1] / max_px)))
    b = a[::sx, ::sy]
    lo, hi = float(b.min()), float(b.max())
    q = np.zeros(b.shape, np.uint8) if hi == lo else np.round(255 * (b - lo) / (hi - lo)).astype(np.uint8)
    frame = {"v": PROTOCOL_VERSION, "kind": "frame", "run_id": run_id, "name": name,
             "wall": time.time(), "t": t, "iteration": iteration,
             "shape": list(b.shape), "full_shape": list(a.shape), "stride": [sx, sy],
             "min": lo, "max": hi, "encoding": "uint8-minmax-base64",
             "data": base64.b64encode(np.ascontiguousarray(q).tobytes()).decode("ascii")}
    if x is not None and y is not None:
        frame["extent"] = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    if scale is not None:
        frame["scale"] = scale
    if resample is not None:
        frame["resample"] = resample
    return frame


def decode_frame(frame: dict) -> np.ndarray:
    """Inverse of :func:`encode_frame` (up to the declared 8-bit quantization)."""
    if frame.get("encoding") != "uint8-minmax-base64":
        raise OptionError(f"unknown frame encoding {frame.get('encoding')!r}")
    q = np.frombuffer(base64.b64decode(frame["data"]), np.uint8).reshape(frame["shape"])
    lo, hi = frame["min"], frame["max"]
    return lo + (hi - lo) * q.astype(float) / 255.0


# ---------------------------------------------------------------------------- 3D frames
def _host(a: Any) -> np.ndarray:
    """Device (CuPy) or host array -> small host float array (the caller strides first)."""
    if hasattr(a, "get") and not isinstance(a, np.ndarray):
        a = a.get()
    return np.asarray(a, dtype=float)


def _quant8(b: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = float(b.min()), float(b.max())
    q = np.zeros(b.shape, np.uint8) if hi == lo else np.round(255 * (b - lo) / (hi - lo)).astype(np.uint8)
    return q, lo, hi


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def encode_frame3d(run_id: str, name: str, data: Any, *, max_px: int, extent: Any, scale: str,
                   axes: tuple = ("x", "y", "z"), coords: Any = None, resample: str = "stride",
                   t: float | None = None, iteration: int | None = None) -> dict:
    """Lossy 3D frame. ``data`` (NumPy or CuPy, shape (n0, n1, n2)) is strided down to at most
    ``max_px`` samples per axis **on its own device** (only the small array is copied to the
    host), then quantized to 8 bits between its min and max. ``resample="given"`` declares that
    the caller already resampled the field (e.g. exact spectral evaluation on a uniform grid);
    the stride still applies if the array is larger than ``max_px``. ``coords``: optional
    positions of the samples along each axis (non-uniform nodes, e.g. Chebyshev), strided like
    the data; without them the nodes are declared uniform with the endpoint excluded
    (``lo + (hi - lo) j / n``, the periodic convention) and their positions are sent anyway, so
    a frame always says where its samples are. ``scale``: declared display scale
    (``smoke`` | ``signed`` | ``dye``)."""
    if scale not in SCALES:
        raise OptionError(f"frame3d {name!r}: scale must be one of {SCALES}, got {scale!r}")
    if resample not in ("stride", "given"):
        raise OptionError(f"frame3d {name!r}: resample must be 'stride' or 'given'")
    shape = tuple(int(n) for n in getattr(data, "shape", ()))
    if len(shape) != 3:
        raise OptionError(f"frame3d {name!r}: a 3D array is required, got shape {shape}")
    ext = [[float(lo), float(hi)] for lo, hi in extent]
    if len(ext) != 3:
        raise OptionError(f"frame3d {name!r}: extent needs 3 [lo, hi] pairs")
    st = [max(1, int(np.ceil(n / max_px))) for n in shape]
    b = _host(data[::st[0], ::st[1], ::st[2]])     # stride on the device, copy the small array
    if not np.all(np.isfinite(b)):
        n_bad = int(b.size - np.count_nonzero(np.isfinite(b)))
        raise OptionError(f"frame3d {name!r}: {n_bad} non-finite values (not sent as a picture)")
    q, lo, hi = _quant8(b)
    fr = {"v": PROTOCOL_VERSION, "kind": "frame3d", "run_id": run_id, "name": name,
          "wall": time.time(), "t": t, "iteration": iteration, "shape": list(b.shape),
          "full_shape": list(shape), "resample": resample, "stride": st, "axes": list(axes),
          "extent": ext, "scale": scale, "min": lo, "max": hi, "encoding": "uint8-minmax-base64",
          "lossy": (f"{'given resampling, then ' if resample == 'given' else ''}stride "
                    f"{'x'.join(map(str, st))} ({'x'.join(map(str, shape))} -> "
                    f"{'x'.join(map(str, b.shape))}), 8-bit quantization between min and max"),
          "data": _b64(q)}
    if coords is not None:
        cs = [np.asarray(_host(c)).reshape(-1) for c in coords]
        if len(cs) != 3 or any(len(c) != n for c, n in zip(cs, shape, strict=True)):
            raise OptionError(f"frame3d {name!r}: coords must give {shape} positions per axis")
        fr["coords"] = [c[::s].tolist() for c, s in zip(cs, st, strict=True)]
        fr["nodes"] = "given"
    else:                                          # declared default: periodic-style nodes
        fr["coords"] = [(lo + (hi - lo) * np.arange(0, n, s) / n).tolist()
                        for (lo, hi), n, s in zip(ext, shape, st, strict=True)]
        fr["nodes"] = "uniform, lo + (hi - lo) j / n (endpoint excluded)"
    return fr


def decode_frame3d(frame: dict) -> np.ndarray:
    """Inverse of :func:`encode_frame3d` (up to the 8-bit quantization, error <= (max-min)/510)."""
    if frame.get("encoding") != "uint8-minmax-base64":
        raise OptionError(f"unknown frame3d encoding {frame.get('encoding')!r}")
    q = np.frombuffer(base64.b64decode(frame["data"]), np.uint8).reshape(frame["shape"])
    return frame["min"] + (frame["max"] - frame["min"]) * q.astype(float) / 255.0


def encode_points(run_id: str, name: str, pts: Any, *, max_points: int, extent: Any, mode: str,
                  values: Any = None, t: float | None = None, iteration: int | None = None) -> dict:
    """Lossy particles / polylines. ``pts``: (n, d) particles or (n_lines, n_pts, d) polylines,
    d = 2 or 3, NaN = missing (not yet emitted, left the domain). If there are more than
    ``max_points`` points, whole lines are dropped by a uniform stride first, then points along
    each line (both strides declared). Positions are quantized to 16 bits inside ``extent``
    (step (hi - lo)/65534; 65535 = missing); a point outside the extent is clipped **and
    counted** (``clipped``). ``values`` (same leading shape, optional): 8-bit per point."""
    if mode not in POINT_MODES:
        raise OptionError(f"points3d {name!r}: mode must be one of {POINT_MODES}, got {mode!r}")
    max_points = check_count(max_points, "max_points", f"points3d {name!r}")
    a = _host(pts)
    if a.ndim == 2:
        a = a[:, None, :]
    if a.ndim != 3 or a.shape[2] not in (2, 3):
        raise OptionError(f"points3d {name!r}: (n, d) or (n_lines, n_pts, d) with d = 2 or 3, "
                         f"got shape {a.shape}")
    d = a.shape[2]
    ext = np.array([[float(lo), float(hi)] for lo, hi in extent])
    if ext.shape != (d, 2) or np.any(ext[:, 1] <= ext[:, 0]):
        raise OptionError(f"points3d {name!r}: extent needs {d} increasing [lo, hi] pairs")
    n_lines, n_pts = a.shape[:2]
    full = [n_lines, n_pts]
    sl = max(1, int(np.ceil(n_lines * n_pts / max_points))) if n_pts == 1 else \
        max(1, int(np.ceil(n_lines / max(1, max_points // n_pts))))
    a = a[::sl]
    sp = max(1, int(np.ceil(a.shape[0] * a.shape[1] / max_points)))
    if sp > 1:                                     # keep the last point of each line (its head)
        idx = np.unique(np.r_[np.arange(0, a.shape[1], sp), a.shape[1] - 1])
        a = a[:, idx]
    v = None
    if values is not None:
        v = _host(values)
        v = (v[:, None] if v.ndim == 1 else v)[::sl]
        if sp > 1:
            v = v[:, idx]
        if v.shape != a.shape[:2]:
            raise OptionError(f"points3d {name!r}: values shape {v.shape} != {a.shape[:2]}")
    missing = ~np.all(np.isfinite(a), axis=2)
    u = (a - ext[:, 0]) / (ext[:, 1] - ext[:, 0])
    u = np.where(np.isfinite(u), u, 0.0)
    clipped = int(np.count_nonzero(((u < 0) | (u > 1)).any(axis=2) & ~missing))
    q = np.round(np.clip(u, 0, 1) * 65534).astype(np.uint16)
    q[missing] = 65535
    fr = {"v": PROTOCOL_VERSION, "kind": "points3d", "run_id": run_id, "name": name,
          "wall": time.time(), "t": t, "iteration": iteration, "mode": mode, "dim": d,
          "shape": [int(a.shape[0]), int(a.shape[1])], "full_shape": full, "stride": [sl, sp],
          "extent": ext.tolist(), "encoding": "uint16-extent-base64", "clipped": clipped,
          "missing": int(missing.sum()),
          "lossy": (f"lines strided by {sl}, points by {sp} (last kept), positions quantized to "
                    "16 bits in the extent"), "data": _b64(q.astype("<u2"))}
    if v is not None:
        good = np.isfinite(v) & ~missing
        if good.any():
            vq, lo, hi = _quant8(np.where(good, v, v[good].min()))
            fr.update(values=_b64(vq), vmin=lo, vmax=hi)
    return fr


def decode_points(frame: dict) -> np.ndarray:
    """(n_lines, n_pts, d) positions, NaN where missing (inverse of :func:`encode_points`)."""
    if frame.get("encoding") != "uint16-extent-base64":
        raise OptionError(f"unknown points3d encoding {frame.get('encoding')!r}")
    n, m = frame["shape"]
    d = frame["dim"]
    q = np.frombuffer(base64.b64decode(frame["data"]), "<u2").reshape(n, m, d)
    ext = np.asarray(frame["extent"])
    out = ext[:, 0] + (ext[:, 1] - ext[:, 0]) * q / 65534.0
    out[np.any(q == 65535, axis=2)] = np.nan
    return out


def encode_vec2d(run_id: str, name: str, u: Any, v: Any, *, max_px: int, extent: Any,
                 t: float | None = None, iteration: int | None = None) -> dict:
    """Lossy 2D vector field (for LIC and streamlines in the Studio): both components strided
    to at most ``max_px`` per axis and quantized to 8 bits on one symmetric scale
    ``amp = max(|u|, |v|)``: q = round(127.5 (1 + c / amp)), c = amp (q - 127.5) / 127.5."""
    a, b = getattr(u, "shape", ()), getattr(v, "shape", ())
    if len(a) != 2 or tuple(a) != tuple(b):
        raise OptionError(f"vec2d {name!r}: two 2D arrays of one shape required, got {a} and {b}")
    s = [max(1, int(np.ceil(n / max_px))) for n in a]
    uu, vv = _host(u[::s[0], ::s[1]]), _host(v[::s[0], ::s[1]])
    if not (np.all(np.isfinite(uu)) and np.all(np.isfinite(vv))):
        raise OptionError(f"vec2d {name!r}: non-finite values (not sent)")
    amp = float(max(np.abs(uu).max(), np.abs(vv).max()))
    sc = amp if amp > 0 else 1.0
    q = np.round(127.5 * (1 + np.stack([uu, vv]) / sc)).astype(np.uint8)
    ext = [[float(lo), float(hi)] for lo, hi in extent]
    return {"v": PROTOCOL_VERSION, "kind": "vec2d", "run_id": run_id, "name": name,
            "wall": time.time(), "t": t, "iteration": iteration, "shape": list(uu.shape),
            "full_shape": list(map(int, a)), "stride": s, "extent": ext, "amp": amp,
            "encoding": "uint8-signed-amp-base64",
            "lossy": f"stride {s[0]}x{s[1]}, 8-bit quantization on the symmetric scale +-amp",
            "data": _b64(q)}


def decode_vec2d(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of a :func:`encode_vec2d` frame (error <= amp / 255)."""
    if frame.get("encoding") != "uint8-signed-amp-base64":
        raise OptionError(f"unknown vec2d encoding {frame.get('encoding')!r}")
    q = np.frombuffer(base64.b64decode(frame["data"]), np.uint8).reshape(2, *frame["shape"])
    c = frame["amp"] * (q.astype(float) - 127.5) / 127.5
    return c[0], c[1]


# ---------------------------------------------------------------------------- curves (1D)
def encode_curve(run_id: str, name: str, x: Any, y: Any, *, max_points: int = 4096,
                 xlabel: str = "", ylabel: str = "", logx: bool = False, logy: bool = False,
                 t: float | None = None, iteration: int | None = None) -> dict:
    """A 1D curve (a spectrum, a profile): x and y sent as JSON floats, strided to at most
    ``max_points`` points (the stride is declared; the last point is always kept). Non-finite
    values are refused (the curve is not sent, the caller logs why): a picture never hides a NaN."""
    max_points = check_count(max_points, "max_points", f"curve {name!r}", minimum=2)
    xx, yy = _host(x).reshape(-1), _host(y).reshape(-1)
    if xx.shape != yy.shape or xx.size == 0:
        raise OptionError(f"curve {name!r}: x and y need one non-empty shape, got {xx.shape} and "
                         f"{yy.shape}")
    bad = int(np.count_nonzero(~np.isfinite(xx)) + np.count_nonzero(~np.isfinite(yy)))
    if bad:
        raise OptionError(f"curve {name!r}: {bad} non-finite values (not sent)")
    st = max(1, int(np.ceil(xx.size / max_points)))
    idx = np.unique(np.r_[np.arange(0, xx.size, st), xx.size - 1])
    return {"v": PROTOCOL_VERSION, "kind": "curve", "run_id": run_id, "name": name,
            "wall": time.time(), "t": t, "iteration": iteration, "n": int(xx.size),
            "stride": st, "x": xx[idx].tolist(), "y": yy[idx].tolist(), "xlabel": str(xlabel),
            "ylabel": str(ylabel), "logx": bool(logx), "logy": bool(logy),
            "lossy": "none" if st == 1 else f"stride {st} ({xx.size} -> {idx.size} points)"}


def decode_curve(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of a :func:`encode_curve` frame."""
    return np.asarray(frame["x"], dtype=float), np.asarray(frame["y"], dtype=float)


# ---------------------------------------------------------------------------- the film
def _up_filter(q: np.ndarray, row: int) -> np.ndarray:
    """PNG "Up" filter on rows of length ``row``: r[0] = q[0], r[k] = q[k] - q[k-1] (mod 256).
    Smooth fields give many small differences, which deflate compresses well."""
    a = q.reshape(-1, row)
    d = a.copy()
    d[1:] -= a[:-1]                                  # uint8 arithmetic wraps mod 256 (intended)
    return d.reshape(-1)


def _up_unfilter(d: np.ndarray, row: int) -> np.ndarray:
    """Inverse of :func:`_up_filter` (a cumulative sum mod 256 down the rows)."""
    return np.cumsum(d.reshape(-1, row), axis=0, dtype=np.uint8).reshape(-1)


def film_record(frame: dict, *, i: int, level: int, compress_level: int = 6) -> dict:
    """The film line of a frame (one JSON object per line of ``<run_id>.film.ndjson``).

    ``i`` is the index of the frame in the film of its name, ``level`` the thinning level of the
    film when it was kept (every ``4**level``-th frame of that name). 8-bit frames (``frame``,
    ``frame3d``) are re-encoded **losslessly** with :data:`FILM_ENCODING` (same bytes as the
    frame, about 3 to 10 times smaller on smooth fields); the other kinds keep their encoding."""
    rec = {**frame, "i": check_count(i, "i", "film_record()", minimum=0),
           "level": check_count(level, "level", "film_record()", minimum=0)}
    if frame.get("kind") in ("frame", "frame3d") and frame.get("encoding") == "uint8-minmax-base64":
        q = np.frombuffer(base64.b64decode(frame["data"]), np.uint8)
        row = int(frame["shape"][-1])
        rec["data"] = _b64(np.frombuffer(zlib.compress(_up_filter(q, row).tobytes(),
                                                       compress_level), np.uint8))
        rec["encoding"] = FILM_ENCODING
    return rec


def film_frame(rec: dict) -> dict:
    """The frame of a film line, back in its spool encoding (inverse of :func:`film_record`)."""
    fr = {k: v for k, v in rec.items() if k not in ("i", "level")}
    if rec.get("encoding") == FILM_ENCODING:
        d = np.frombuffer(zlib.decompress(base64.b64decode(rec["data"])), np.uint8)
        fr["data"] = _b64(_up_unfilter(d, int(rec["shape"][-1])))
        fr["encoding"] = "uint8-minmax-base64"
    return fr
