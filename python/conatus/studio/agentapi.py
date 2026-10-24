"""The agent channel of the Studio: what a script or an AI agent reads about a run and may send.

One pure module (stdlib + NumPy, no tornado) shared by the three places that speak it:

* the hub's JSON API ``/api/v1/...`` (:mod:`conatus.studio.hub`, tokens with a scope);
* the spool transport of the client (:mod:`conatus.studio.client`, a session on the cluster reads
  the shared spool and writes the control file itself: no server at all);
* the relay, which re-checks every command against the header of the run file.

What it defines, all versioned by :data:`API_VERSION` (additive changes only within a version):

* :func:`run_state` and :func:`alerts`: the state of a run and what deserves attention, derived
  from its records with explicit thresholds (:data:`SILENT_S`), never guessed;
* :func:`run_view`: the stable JSON of a run (status, step, time, dt, last scalars, log tail,
  events, latest frames, declared views and settable parameters, alerts, allowed commands);
* :func:`check_command`: the one rule for a command (declared commands only, ``set`` within the
  declared bounds, stop and fork confirmed in two steps), raising :class:`CommandRefused` with an
  HTTP-like code and a sentence for the sender;
* :class:`Confirmations`: the two-step confirmation (a short-lived token bound to the sender, the
  run, the command and its parameters);
* :func:`frame_png`: the latest frame of a view as a PNG in the brand scales (an agent that reads
  images can look at the flow).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
import struct
import time
import zlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from conatus.telemetry.protocol import COMMANDS, CONFIRM_COMMANDS, check_set

API_VERSION = 1
SILENT_S = 60.0                      # a live run silent for longer gets the alert "silent"
LIVE_S = 30.0                        # a record in the last LIVE_S seconds: "running"
CONFIRM_TTL_S = 120.0                # a confirmation token is valid this long, once
MAX_PARAMS_BYTES = 4096              # fork/set parameters: small JSON objects only
ENDED = ("finished", "failed", "stopped")
STATES = ("running", "paused", "stale", *ENDED)


class CommandRefused(Exception):
    """A command that is not sent. ``code``: 400 malformed, 403 not allowed (scope, not declared),
    404 unknown run, 409 needs confirmation (``confirm_token`` attached) or wrong state,
    503 relay offline."""

    def __init__(self, code: int, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code, self.extra = int(code), extra

    def body(self) -> dict:
        """The JSON answer: ``{"ok": false, "error": message, "code": code, ...}``."""
        return {"ok": False, "error": str(self), "code": self.code, **self.extra}


# ------------------------------------------------------------------------------ state, alerts
def _get(rec: Mapping | None, key: str, default: Any = None) -> Any:
    return rec.get(key, default) if isinstance(rec, Mapping) else default


def run_state(st: Any, now: float | None = None) -> str:
    """``running | paused | stale | finished | failed | stopped`` from the records of a run
    (a :class:`conatus.telemetry.RunState`): final status (stopped if a stop was received),
    a failure event, the paused flag, else running if a record came in the last
    :data:`LIVE_S` s, else stale."""
    now = time.time() if now is None else now
    status = st.status or {}
    events = list(st.events)
    if status.get("final"):
        if status.get("error") or status.get("failed"):
            return "failed"
        if any(e.get("name") == "command_received" and e.get("cmd") == "stop" for e in events):
            return "stopped"
        return "finished"
    if any(e.get("name") in ("failed", "error") for e in events):
        return "failed"
    if status.get("paused"):
        return "paused"
    return "running" if now - float(st.last_wall or 0) < LIVE_S else "stale"


def _non_finite(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in ("nan", "inf", "-inf", "+inf")


def alerts(st: Any, now: float | None = None, relay_connected: bool | None = None) -> list[dict]:
    """What deserves attention, ``[{"code", "level": info|warning|error, "message"}]``, most
    severe first. Codes (stable): ``non_finite``, ``telemetry_error``, ``failed``, ``silent``,
    ``relay_down``, ``scalar_error``, ``log_errors``, ``dropped``, ``film_full``,
    ``set_refused``."""
    now = time.time() if now is None else now
    out: list[dict] = []
    state = run_state(st, now)
    last = {}
    for rec in list(st.scalars)[-50:]:
        last.update(rec)
    bad = sorted(k for k, v in last.items() if _non_finite(v))
    if bad:
        out.append(dict(code="non_finite", level="error",
                        message=f"non-finite values in {', '.join(bad)} (last record)"))
    status = st.status or {}
    if status.get("error"):
        out.append(dict(code="telemetry_error", level="error", message=str(status["error"])))
    if state == "failed":
        out.append(dict(code="failed", level="error", message="the run reported a failure"))
    age = now - float(st.last_wall or 0)
    if state in ("running", "stale") and age > SILENT_S:
        out.append(dict(code="silent", level="warning",
                        message=f"no record for {age:.0f} s (stuck, ended without closing, or "
                                "the relay is down)"))
    if relay_connected is False and state not in ENDED:
        out.append(dict(code="relay_down", level="warning",
                        message="the relay of this run is not connected: data may be late and "
                                "commands cannot be sent"))
    errs = sorted(k for k, v in last.items() if isinstance(v, str) and v.startswith("error:"))
    if errs:
        out.append(dict(code="scalar_error", level="warning",
                        message=f"declared scalars failing: {', '.join(errs)} ({last[errs[0]]})"))
    logged = [r for r in st.logs if r.get("level") == "error"]
    if logged:
        out.append(dict(code="log_errors", level="warning",
                        message=f"{len(logged)} error(s) in the log; last: {logged[-1].get('message')}"))
    dr, df = int(status.get("dropped_records") or 0), int(status.get("dropped_frames") or 0)
    if dr or df:
        out.append(dict(code="dropped", level="info",
                        message=f"telemetry dropped {dr} record(s) and {df} frame(s) (declared; the "
                                "computation is unaffected)"))
    events = list(st.events)
    if any(e.get("name") == "film_full" for e in events):
        out.append(dict(code="film_full", level="info",
                        message="the film reached its budget: later frames are not kept for replay"))
    for e in reversed(events):
        if e.get("name") in ("param_set", "set_refused"):
            if e.get("name") == "set_refused":
                out.append(dict(code="set_refused", level="info",
                                message=f"last set refused by the run: {e.get('reason')}"))
            break
    rank = {"error": 0, "warning": 1, "info": 2}
    return sorted(out, key=lambda a: rank[a["level"]])


def allowed_commands(st: Any, now: float | None = None) -> list[str]:
    """Commands that make sense now: none once the run has ended; ``set`` only if the run
    declared settable parameters; ``resume`` only when paused, ``pause`` otherwise."""
    state = run_state(st, now)
    if state in ENDED:
        return []
    out = ["resume" if state == "paused" else "pause", "checkpoint", "stop", "fork"]
    if (st.header or {}).get("controls"):
        out.append("set")
    return out


def last_values(st: Any, n: int = 200) -> dict:
    """Latest value of every scalar (declared scalars are throttled: looked up in the last
    ``n`` records), without the record fields."""
    out: dict = {}
    for rec in list(st.scalars)[-n:]:
        out.update({k: v for k, v in rec.items() if k not in ("v", "kind", "run_id", "wall", "relay")})
    return out


def _frame_brief(fr: Mapping) -> dict:
    return {k: fr.get(k) for k in ("kind", "t", "iteration", "wall", "shape", "full_shape", "scale",
                                   "min", "max", "lossy") if fr.get(k) is not None}


def _rec_brief(rec: Mapping) -> dict:
    return {k: v for k, v in rec.items() if k not in ("v", "kind", "run_id", "relay")}


def run_view(key: str, st: Any, *, now: float | None = None, relay: str | None = None,
             relay_connected: bool | None = None, log: int = 20, events: int = 20,
             brief: bool = False, pilots: Iterable[dict] = ()) -> dict:
    """The stable JSON of one run for agents and the dashboard (API v1; fields are only ever
    added). ``brief`` drops the log, the events and the declarations (the runs list)."""
    now = time.time() if now is None else now
    h = st.header or {}
    meta = h.get("meta") or {}
    vals = last_values(st)
    state = run_state(st, now)
    out = {
        "key": key, "run_id": key.partition(":")[2] if relay else key, "relay": relay,
        "state": state, "model": meta.get("model"), "title": meta.get("title"),
        "t": vals.get("t"), "iteration": vals.get("iteration"), "dt": vals.get("dt"),
        "wall_per_step": vals.get("wall_per_step"),
        "last_wall": st.last_wall or None, "age_s": round(now - st.last_wall, 3) if st.last_wall else None,
        "alerts": alerts(st, now, relay_connected),
        "commands": allowed_commands(st, now),
        "settable": list(h.get("controls") or []),
        "pilots": list(pilots),
    }
    if brief:
        out["scalars"] = {k: v for k, v in vals.items()
                          if k not in ("t", "iteration", "dt", "wall_per_step", "gpu_mem_bytes")}
        return out
    out.update({
        "scalars": vals,
        "host": h.get("host"), "slurm_job": h.get("slurm_job"), "pid": h.get("pid"),
        "N": meta.get("N"), "scheme": meta.get("scheme"), "backend": meta.get("backend"),
        "parameters": meta.get("parameters"),
        "status": _rec_brief(st.status) if st.status else None,
        "views": [v.get("name") for v in h.get("views") or []] or sorted(st.frames),
        "stats": [v.get("name") for v in h.get("stats") or []],
        "frames": {n: _frame_brief(f) for n, f in st.frames.items()},
        "log": [_rec_brief(r) for r in list(st.logs)[-log:]] if log else [],
        "events": [_rec_brief(r) for r in list(st.events)[-events:]] if events else [],
        "malformed": st.malformed,
    })
    return out


# ------------------------------------------------------------------------------ commands
def check_command(st: Any, cmd: Any, params: Any, *, now: float | None = None) -> None:
    """Refuse (:class:`CommandRefused`) anything but a declared command that makes sense for this
    run now: a known command, small JSON parameters, ``set`` only on a parameter the run declared
    settable and within its bounds, nothing once the run has ended. Confirmation is separate
    (:class:`Confirmations`)."""
    if cmd not in COMMANDS:
        raise CommandRefused(400, f"unknown command {cmd!r}; declared commands: {list(COMMANDS)}")
    if params is not None and (not isinstance(params, dict)
                               or len(json.dumps(params)) > MAX_PARAMS_BYTES):
        raise CommandRefused(400, f"params must be a JSON object of at most {MAX_PARAMS_BYTES} bytes")
    if st is None:
        raise CommandRefused(404, "unknown run")
    state = run_state(st, now)
    if state in ENDED:
        raise CommandRefused(409, f"the run has {state}: no command can reach it")
    if cmd == "set":
        try:
            check_set((st.header or {}).get("controls"), params)
        except ValueError as err:
            raise CommandRefused(403 if "not declared" in str(err) else 400, str(err)) from None
    elif cmd != "fork" and params:
        raise CommandRefused(400, f"{cmd} takes no parameters")


def _canon(params: Any) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


class Confirmations:
    """Two-step confirmation of stop and fork. The first request gets a token bound to (sender,
    run, command, parameters), valid :data:`CONFIRM_TTL_S` seconds from its issue and once; the
    second request repeats the command with ``confirm: <token>``. The token carries its expiry
    (``<unix expiry>.<mac>``, the mac covering it), so the advertised lifetime is the real one
    (a time-bucketed token would stay valid up to twice as long). ``secret`` keys the tokens
    (random per hub process; the spool transport passes a fixed one: there the token proves
    intent, not identity, which Unix permissions already settle)."""

    def __init__(self, secret: bytes | None = None, ttl_s: float = CONFIRM_TTL_S,
                 single_use: bool = True) -> None:
        self.secret = secret if secret is not None else secrets.token_bytes(32)
        self.ttl_s, self.single_use = float(ttl_s), bool(single_use)
        self._used: dict[str, float] = {}

    def _mac(self, who: str, key: str, cmd: str, params: Any, expiry: int) -> str:
        msg = f"{who}\n{key}\n{cmd}\n{_canon(params)}\n{expiry}".encode()
        return hmac.new(self.secret, msg, hashlib.sha256).hexdigest()[:24]

    def issue(self, who: str, key: str, cmd: str, params: Any, now: float | None = None) -> str:
        """A token for this exact command, valid ``ttl_s`` seconds from ``now`` (whole seconds,
        rounded up: never shorter than announced)."""
        now = time.time() if now is None else now
        expiry = math.ceil(now + self.ttl_s)
        return f"{expiry}.{self._mac(who, key, cmd, params, expiry)}"

    def check(self, token: Any, who: str, key: str, cmd: str, params: Any,
              now: float | None = None) -> bool:
        """True iff ``token`` was issued for this exact command, has not expired and was not
        used yet (the mac is compared in constant time)."""
        now = time.time() if now is None else now
        if not isinstance(token, str) or len(token) > 64:
            return False
        exp_s, _, mac = token.partition(".")
        if not exp_s.isdigit():
            return False
        expiry = int(exp_s)
        if not hmac.compare_digest(mac, self._mac(who, key, cmd, params, expiry)):
            return False
        if not now <= expiry <= now + self.ttl_s + 1:      # expired, or a forged far expiry
            return False
        if self.single_use:
            self._used = {t: e for t, e in self._used.items() if e >= now}
            if token in self._used:
                return False
            self._used[token] = expiry
        return True


def needs_confirmation(cmd: str) -> bool:
    """Stop and fork are confirmed in two steps (protocol ``CONFIRM_COMMANDS``)."""
    return cmd in CONFIRM_COMMANDS


def resolve(keys: Iterable[str], ref: str) -> str:
    """The run key named by ``ref``: an exact key (``relay:run_id`` on the hub), else a run id
    that matches exactly one key. KeyError with the candidates otherwise."""
    keys = list(keys)
    if ref in keys:
        return ref
    hits = [k for k in keys if k.partition(":")[2] == ref]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise KeyError(f"run {ref!r} is ambiguous: {sorted(hits)}")
    raise KeyError(f"no run {ref!r}; known: {sorted(keys)[:20]}")


# ------------------------------------------------------------------------------ frames as PNG
_STOPS = {   # brand scales (brand charter), as studio/app/src/lib/colormap.ts
    "smoke": [(0, "#0B0B0B"), (0.30, "#3A3A38"), (0.62, "#8A8A86"), (1, "#EDEBE6")],
    "signed": [(0, "#D4574A"), (0.30, "#6E2019"), (0.50, "#0B0B0B"), (0.70, "#3F5245"), (1, "#9DBBA6")],
    "dye": [(0, "#0B0B0B"), (1, "#C6F03C")],
}


def _lut(scale: str) -> np.ndarray:
    stops = _STOPS.get(scale, _STOPS["smoke"])
    p = np.linspace(0, 1, 256)
    xs = [s for s, _ in stops]
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in stops], float)
    return np.stack([np.interp(p, xs, rgb[:, k]) for k in range(3)], axis=1).round().astype(np.uint8)


def _png(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def frame_png(fr: Mapping, *, scale: str | None = None, upscale: int = 0) -> bytes:
    """PNG of an 8-bit frame (2D, or the middle slice of axis 1 of a 3D frame) in its declared
    brand scale (``signed``: zero is black, symmetric about 0, like the app). First array axis
    across, second up (the app's orientation). ``upscale``: nearest-neighbour factor (0: so that
    the image is at least 256 px wide). ValueError for other kinds or encodings."""
    if fr.get("kind") not in ("frame", "frame3d") or fr.get("encoding") != "uint8-minmax-base64":
        raise ValueError(f"no picture for a {fr.get('kind')!r} frame (encoding {fr.get('encoding')!r})")
    shape = [int(n) for n in fr["shape"]]
    q = np.frombuffer(base64.b64decode(fr["data"]), np.uint8).reshape(shape)
    if q.ndim == 3:
        q = q[:, shape[1] // 2, :]
    lo, hi = float(fr["min"]), float(fr["max"])
    sc = scale or fr.get("scale") or "smoke"
    v = lo + (hi - lo) * q.astype(float) / 255.0
    if sc == "signed":
        a = max(abs(lo), abs(hi)) or 1.0
        p = 0.5 + 0.5 * v / a
    else:
        p = (v - lo) / ((hi - lo) or 1.0)
    idx = np.clip(np.round(p * 255), 0, 255).astype(np.uint8)
    img = _lut(sc)[idx.T[::-1]]                     # x across, y up
    k = int(upscale) or max(1, -(-256 // img.shape[1]))
    if k > 1:
        img = np.repeat(np.repeat(img, k, axis=0), k, axis=1)
    return _png(np.ascontiguousarray(img))
