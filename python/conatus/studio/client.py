"""Client of the agent channel: read the state of a run and send it declared commands.

Two transports, one interface and the same JSON (API v1, :mod:`conatus.studio.agentapi`):

* :class:`HubClient` talks to a hub over HTTPS (``/api/v1``) with a token (a user's own token or
  an agent token of scope ``read`` or ``command``). Plain HTTP only to this machine (an ssh tunnel
  to a hub running in a job): a token never travels in clear.
* :class:`SpoolClient` reads the shared telemetry spool directly and appends the command to the
  run's control file, like the relay does: what a session on the cluster uses (no server, no
  port; Unix permissions on the spool are the authentication). Its commands go through the same
  checks and are written to ``<spool>/studio_audit.jsonl``.

Stdlib only (urllib): an agent needs nothing beyond ``conatus``. The command line is
``python -m conatus studio ...`` (:mod:`conatus.studio.cli`).
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from conatus.telemetry import RunState
from conatus.telemetry.protocol import RUN_ID_RE, SPOOL_AUDIT, append_control, run_file_id

from . import agentapi
from ..errors import OptionError

# The documented client (docs/site/guide/studio_agents.md); the rest is importable (M5).
__all__ = ["connect", "HubClient", "SpoolClient", "StudioError"]

LOOPBACK = ("127.0.0.1", "localhost", "::1")
SPOOL_CONFIRM_SECRET = b"conatus-spool-confirmation"   # proves intent, not identity (see Confirmations)
TAIL_BYTES = 8 << 20                 # spool transport: bytes of a run file read from its end


class StudioError(Exception):
    """The hub could not be reached or answered something that is not the API (``code`` 0: no
    answer at all)."""

    def __init__(self, message: str, code: int = 0) -> None:
        super().__init__(message)
        self.code = int(code)


# ============================================================================== hub transport
class HubClient:
    """The hub's ``/api/v1`` over HTTP(S). Every method returns the JSON answer (a dict with
    ``ok``; refusals are answers too, with ``error`` and ``code``), except :meth:`frame` with
    ``png=True`` (bytes)."""

    transport = "hub"

    def __init__(self, url: str, token: str, *, timeout: float = 30.0) -> None:
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("https", "http"):
            raise OptionError(f"hub URL must be https:// (or http:// to this machine), got {url!r}",
                              fix="connect('https://hub.example', token) (http:// only to this machine)")
        if u.scheme == "http" and u.hostname not in LOOPBACK:
            raise OptionError(f"refusing plain http:// to {u.hostname!r}: the token would travel in "
                              "clear; use https://, or an ssh tunnel to localhost",
                              fix="use https:// (or an SSH tunnel to localhost)")
        if not token:
            raise OptionError("empty token", fix='pass the token printed by the hub (or CONATUS_HUB_TOKEN)')
        self.url, self.token, self.timeout = url.rstrip("/"), token, float(timeout)

    def _req(self, method: str, path: str, body: dict | None = None, *, raw: bool = False,
             timeout: float | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/json",
            **({"Content-Type": "application/json"} if data is not None else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                payload = r.read()
                status = r.status
        except urllib.error.HTTPError as err:
            payload, status = err.read(), err.code
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            raise StudioError(f"hub {self.url} unreachable: {getattr(err, 'reason', err)}") from None
        if raw and status == 200:
            return payload
        try:
            out = json.loads(payload or b"{}")
        except ValueError:
            raise StudioError(f"hub answered {status} with something that is not JSON: "
                              f"{payload[:120]!r}", status) from None
        if not isinstance(out, dict):
            raise StudioError(f"hub answered {status} with {type(out).__name__}", status)
        out.setdefault("code", status)
        out.setdefault("ok", 200 <= status < 300)
        return out

    @staticmethod
    def _q(ref: str) -> str:
        return urllib.parse.quote(ref, safe="")

    def ping(self) -> dict:
        """The hub's clock; adds ``rtt_ms`` (round trip measured here)."""
        t0 = time.monotonic()
        out = self._req("GET", "/api/v1/ping")
        out["rtt_ms"] = round(1000 * (time.monotonic() - t0), 1)
        return out

    def whoami(self) -> dict:
        """The caller as the hub sees it (name, kind, scope)."""
        return self._req("GET", "/api/v1/whoami")

    def runs(self) -> dict:
        """Brief view of every reachable run."""
        return self._req("GET", "/api/v1/runs")

    def status(self, ref: str, *, log: int = 20, events: int = 20) -> dict:
        """One run in full (``run``)."""
        return self._req("GET", f"/api/v1/runs/{self._q(ref)}?log={int(log)}&events={int(events)}")

    def scalars(self, ref: str, *, since: float = 0.0, names: list[str] | None = None,
                max: int = 2000) -> dict:
        """Scalar records newer than ``since`` (wall time), strided to at most ``max`` rows."""
        q = urllib.parse.urlencode({"since": since, "names": ",".join(names or []), "max": max})
        return self._req("GET", f"/api/v1/runs/{self._q(ref)}/scalars?{q}")

    def log(self, ref: str, n: int = 100) -> dict:
        """The last ``n`` log records and events."""
        return self._req("GET", f"/api/v1/runs/{self._q(ref)}/log?n={int(n)}")

    def frame(self, ref: str, name: str | None = None, *, png: bool = False) -> Any:
        """The latest frame of a view (default: the first declared view with a frame); PNG bytes
        with ``png=True``."""
        q = urllib.parse.urlencode({"name": name or "", "format": "png" if png else "json"})
        return self._req("GET", f"/api/v1/runs/{self._q(ref)}/frame?{q}", raw=png)

    def wait(self, ref: str, *, since: float, timeout: float = 25.0) -> dict:
        """Long poll: the run as soon as a record newer than ``since`` arrives (``changed``)."""
        q = urllib.parse.urlencode({"since": since, "timeout": timeout})
        return self._req("GET", f"/api/v1/runs/{self._q(ref)}/wait?{q}", timeout=timeout + 15)

    def send(self, ref: str, cmd: str, params: dict | None = None,
             confirm: str | None = None) -> dict:
        """One declared command. Stop and fork: the first answer is ``code`` 409 with a
        ``confirm_token``; send again with ``confirm=<token>``."""
        body: dict[str, Any] = {"cmd": cmd}
        if params is not None:
            body["params"] = params
        if confirm is not None:
            body["confirm"] = confirm
        return self._req("POST", f"/api/v1/runs/{self._q(ref)}/commands", body)

    def board(self) -> dict:
        """The dashboard: runs, pilots, relays, recent commands."""
        return self._req("GET", "/api/v1/board")


# ============================================================================== spool transport
def read_run(spool: Path, run_id: str, *, tail_bytes: int = TAIL_BYTES, max_scalars: int = 20000,
             max_logs: int = 2000) -> RunState | None:
    """The :class:`RunState` of one run from the spool: its header (first line), the records of
    the last ``tail_bytes`` of its file (a long run is never read whole) and its latest frames."""
    path = spool / f"{run_id}.jsonl"
    if not RUN_ID_RE.match(run_id) or not path.is_file():
        return None
    st = RunState(run_id, max_scalars, max_logs)
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            first = fh.readline(1 << 20)
            start = max(len(first), size - tail_bytes)
            fh.seek(start)
            chunk = fh.read(size - start)
    except OSError:
        return None
    lines = [first] if first.endswith(b"\n") else []
    if start > len(first):
        chunk = chunk[chunk.find(b"\n") + 1:]             # drop the partial first line
    chunk = chunk[:chunk.rfind(b"\n") + 1]
    lines += chunk.splitlines()
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            st.malformed += 1
            continue
        if isinstance(rec, dict):
            st.ingest(rec)
    prefix = f"{run_id}.frame."
    for f in spool.glob(f"{run_id}.frame.*.json"):
        try:
            fr = json.loads(f.read_text())
        except (OSError, ValueError):
            continue                                     # being replaced: absent this time
        if isinstance(fr, dict):
            st.frames[str(fr.get("name") or f.name[len(prefix):-5])] = fr
    return st


class SpoolClient:
    """The same API on the shared spool, without a server. ``who`` names the sender in the
    control line (``from``) and in the audit (default ``$USER@host``)."""

    transport = "spool"

    def __init__(self, spool: str | Path, *, who: str | None = None,
                 tail_bytes: int = TAIL_BYTES) -> None:
        self.spool = Path(spool)
        who = who or f"{os.environ.get('USER', 'agent')}@{socket.gethostname().split('.')[0]}"
        self.who = re.sub(r"[^\w.@:+-]", "_", str(who))[:64]     # a name, not free text, in the audit
        self.tail_bytes = int(tail_bytes)
        self.confirmations = agentapi.Confirmations(SPOOL_CONFIRM_SECRET, single_use=False)

    def _ids(self) -> list[str]:
        if not self.spool.is_dir():
            return []
        return sorted(r for r in (run_file_id(p.name) for p in self.spool.glob("*.jsonl")) if r)

    def _run(self, ref: str) -> tuple[str, RunState] | dict:
        try:
            rid = agentapi.resolve(self._ids(), ref)
        except KeyError as err:
            return {"api": agentapi.API_VERSION, "ok": False, "code": 404, "error": str(err.args[0])}
        st = read_run(self.spool, rid, tail_bytes=self.tail_bytes)
        if st is None:
            return {"api": agentapi.API_VERSION, "ok": False, "code": 404, "error": f"no run {rid!r}"}
        return rid, st

    @staticmethod
    def _ok(**body: Any) -> dict:
        return {"api": agentapi.API_VERSION, "ok": True, "code": 200, **body}

    def ping(self) -> dict:
        """The spool is reachable (a directory) and how many runs it holds."""
        ok = self.spool.is_dir()
        return {"api": agentapi.API_VERSION, "ok": ok, "code": 200 if ok else 404,
                "wall": time.time(), "mode": "spool", "spool": str(self.spool),
                "runs": len(self._ids()), **({} if ok else {"error": f"no spool {self.spool}"})}

    def whoami(self) -> dict:
        """The sender name written in control lines and the audit."""
        return self._ok(name=self.who, label=self.who, kind="spool", scope="command", runs=None,
                        session=False)

    def runs(self) -> dict:
        """Brief view of every run of the spool, live ones first."""
        now = time.time()
        order = {s: i for i, s in enumerate(agentapi.STATES)}
        out = []
        for rid in self._ids():
            st = read_run(self.spool, rid, tail_bytes=min(self.tail_bytes, 1 << 20))
            if st is not None:
                out.append(agentapi.run_view(rid, st, now=now, brief=True))
        out.sort(key=lambda r: (order.get(r["state"], 9), -(r["last_wall"] or 0)))
        return self._ok(now=now, runs=out, spool=str(self.spool))

    def status(self, ref: str, *, log: int = 20, events: int = 20) -> dict:
        """One run in full."""
        got = self._run(ref)
        if isinstance(got, dict):
            return got
        rid, st = got
        return self._ok(now=time.time(), run=agentapi.run_view(rid, st, log=log, events=events))

    def scalars(self, ref: str, *, since: float = 0.0, names: list[str] | None = None,
                max: int = 2000) -> dict:
        """Scalar records newer than ``since`` (from the tail of the run file)."""
        got = self._run(ref)
        if isinstance(got, dict):
            return got
        rid, st = got
        if isinstance(max, bool) or not isinstance(max, int) or max < 1:     # 0: ZeroDivisionError
            return {"api": agentapi.API_VERSION, "ok": False, "code": 400,
                    "error": f"max must be an integer >= 1, got {max!r}"}
        rows = [r for r in st.scalars if float(r.get("wall") or 0) > since]
        stride = _max(1, -(-len(rows) // max))
        sel = rows[::stride]
        if rows and sel[-1] is not rows[-1]:
            sel.append(rows[-1])
        keep = {"wall", "t", "iteration", *(names or [])}
        out = [{k: v for k, v in r.items() if (not names and k not in ("v", "kind", "run_id"))
                or k in keep} for r in sel]
        return self._ok(key=rid, count=len(rows), stride=stride, rows=out)

    def log(self, ref: str, n: int = 100) -> dict:
        """The last ``n`` log records and events."""
        got = self._run(ref)
        if isinstance(got, dict):
            return got
        rid, st = got
        return self._ok(key=rid, log=[agentapi._rec_brief(r) for r in list(st.logs)[-n:]],
                        events=[agentapi._rec_brief(r) for r in list(st.events)[-n:]])

    def frame(self, ref: str, name: str | None = None, *, png: bool = False) -> Any:
        """The latest frame of a view (PNG bytes with ``png=True``)."""
        got = self._run(ref)
        if isinstance(got, dict):
            return got
        rid, st = got
        name = name or next((v.get("name") for v in (st.header or {}).get("views") or []
                             if v.get("name") in st.frames), None) or next(iter(sorted(st.frames)), None)
        fr = st.frames.get(name) if name else None
        if fr is None:
            return {"api": agentapi.API_VERSION, "ok": False, "code": 404,
                    "error": f"no frame {name!r} for {rid!r} (frames: {sorted(st.frames)})"}
        if png:
            try:
                return agentapi.frame_png(fr)
            except ValueError as err:
                return {"api": agentapi.API_VERSION, "ok": False, "code": 415, "error": str(err)}
        return self._ok(key=rid, name=name, frame=fr)

    def wait(self, ref: str, *, since: float, timeout: float = 25.0, poll_s: float = 0.5) -> dict:
        """The run as soon as its file has a record newer than ``since`` (polled every
        ``poll_s`` s, at most ``timeout`` s)."""
        end = time.monotonic() + float(timeout)
        while True:
            out = self.status(ref)
            if not out.get("ok"):
                return out
            changed = float(out["run"].get("last_wall") or 0) > since
            if changed or time.monotonic() >= end:
                return {**out, "changed": changed}
            time.sleep(min(poll_s, max(0.0, end - time.monotonic())))

    def send(self, ref: str, cmd: str, params: dict | None = None,
             confirm: str | None = None) -> dict:
        """Check the command (same rules as the hub), confirm stop/fork in two steps, append it to
        ``<run>.control.jsonl`` and audit it in ``studio_audit.jsonl``."""
        got = self._run(ref)
        if isinstance(got, dict):
            return got
        rid, st = got
        entry = dict(event="command", user=self.who, transport="spool", run_id=rid, cmd=cmd,
                     params=params)
        try:
            agentapi.check_command(st, cmd, params)
            if agentapi.needs_confirmation(cmd) and not self.confirmations.check(
                    confirm, self.who, rid, cmd, params):
                tok = self.confirmations.issue(self.who, rid, cmd, params)
                raise agentapi.CommandRefused(
                    409, f"{cmd} requires a confirmation: send the same command again with "
                         f"confirm {tok!r} within {agentapi.CONFIRM_TTL_S:.0f} s",
                    needs_confirmation=True, confirm_token=tok)
        except agentapi.CommandRefused as err:
            self._audit(**entry, result=f"refused: {str(err)[:160]}")
            return {"api": agentapi.API_VERSION, "key": rid, **err.body()}
        line = {"cmd": cmd, "seq": int(time.time() * 1e6), "wall": time.time(),
                "from": f"agent:{self.who}", "id": os.urandom(8).hex()}
        if params is not None:
            line["params"] = params
        try:
            append_control(self.spool / f"{rid}.control.jsonl", line)        # 0600
        except OSError as err:
            self._audit(**entry, result=f"failed: {err}")
            return {"api": agentapi.API_VERSION, "ok": False, "code": 500, "key": rid,
                    "error": f"control file not writable: {err}"}
        self._audit(**entry, id=line["id"], result="written to control file")
        return self._ok(key=rid, id=line["id"], line=line,
                        note="written to the run's control file; the run applies it at its next "
                             "check (see its events: command_received, param_set, ...)")

    def _audit(self, **entry: Any) -> None:
        from .auth import AuditLog
        AuditLog(self.spool / SPOOL_AUDIT, echo=False).write(**entry)

    def board(self) -> dict:
        """Runs and the recent commands of the spool audit (no server: no relays, no pilots
        beyond the audit)."""
        out = self.runs()
        recent: list[dict] = []
        try:
            with (self.spool / SPOOL_AUDIT).open("rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - (256 << 10)))
                tail = fh.read().splitlines()[1:]
            for line in tail[-50:]:
                try:
                    recent.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        return {**out, "pilots": [], "relays": {}, "recent": recent[::-1], "who": self.who}


def _max(a: int, b: int) -> int:
    return a if a > b else b


# ============================================================================== configuration
CONFIG = Path("~/.config/conatus/studio.json")
TOKEN_FILE = Path("~/.config/conatus/studio_token")


def connect(*, hub: str | None = None, spool: str | Path | None = None,
            token_file: str | Path | None = None, who: str | None = None,
            timeout: float = 30.0) -> HubClient | SpoolClient:
    """A client from explicit arguments, else ``$CN_STUDIO_URL`` / ``$CN_STUDIO_TOKEN``, else
    ``~/.config/conatus/studio.json`` (``{"url": ..., "token_file": ...}``), else the spool
    (``$CN_TELEMETRY`` or ``scratch/telemetry``). The token is read from a file or the
    environment, never from the command line."""
    from conatus._experimental import experimental
    experimental("studio")
    if spool is not None:
        return SpoolClient(spool, who=who)
    cfg: dict = {}
    try:
        cfg = json.loads(CONFIG.expanduser().read_text())
    except (OSError, ValueError):
        pass
    url = hub or os.environ.get("CN_STUDIO_URL") or cfg.get("url")
    if not url:
        from conatus.telemetry import default_spool
        return SpoolClient(default_spool(), who=who)
    tok = os.environ.get("CN_STUDIO_TOKEN", "").strip()
    tf = token_file or (None if tok else cfg.get("token_file")) or (None if tok else TOKEN_FILE)
    if tf is not None:
        p = Path(tf).expanduser()
        try:
            tok = p.read_text().strip()
            loose = p.stat().st_mode & 0o077
        except OSError as err:                          # not authorised (exit 3), not "hub down"
            raise StudioError(f"no token: {p} unreadable ({err}); pass --token-file or set "
                              "CN_STUDIO_TOKEN", 401) from None
        if loose:
            print(f"warning: {p} is readable by others; chmod 600 it", file=sys.stderr)
    if not tok:
        raise StudioError("no token: empty token file or CN_STUDIO_TOKEN", 401)
    return HubClient(url, tok, timeout=timeout)
