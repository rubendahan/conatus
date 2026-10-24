"""Conatus Studio hub: hosted outside the cluster, fed by outbound relays, HTTPS + auth.

    python -m conatus.studio.hub adduser ruben            # password prompt (scrypt-hashed)
    python -m conatus.studio.hub addrelay mycluster --token-out ~/.config/conatus/relay_token
    python -m conatus.studio.hub addagent claude --scope command --token-out ~/.config/conatus/studio_token
    python -m conatus.studio.hub export                   # hashes only -> CN_HUB_AUTH_JSON secret
    python -m conatus.studio.hub serve --state /data      # 127.0.0.1, behind a TLS proxy (Caddy)
    python -m conatus.studio.hub serve --host 0.0.0.0 --behind-proxy   # container (Fly.io, Render)
    python -m conatus.studio.hub serve --local --spool scratch/telemetry   # hub + relay, 127.0.0.1;
                                  # one-session login token in <state>/local_user_token (0600)

Endpoints
    GET  /healthz                 liveness (no auth, no data)
    GET  /login, POST /login      password (or user API token) -> session cookie, and a session
                                  key handed to the page (``/login-done.js`` stores it)
    POST /logout
    GET  /, /run/<key>, /compare  the Studio app (``studio/app``, built into ``--app``; session
                                  required, else redirect to /login); its files (/assets, /fonts,
                                  /icons, manifest, service worker) are public: code, no data
    GET  /classic                 the former page (``studio/web/index.html``; session required)
    GET  /ws                      browser WebSocket: snapshot then increments (session required;
                                  ``?lite=1``: light stream for phones)
    GET  /api/film/<key>          film summary, ``?name=`` index; ``/api/film/<key>/frames?name=
                                  &from=&to=&step=`` frames (memory, bounded: ``--film-mb``)
    GET  /api/runs, /api/whoami   JSON (session cookie or ``Authorization: Bearer <user token>``)
    POST /api/control/<run>       {"cmd", "params", "confirm"}; stop and fork require confirm=true
    GET  /api/compare             ?runs=a,b&key=energy&x=t  several runs on one curve (S2)
    POST /api/jobs/<op>           S2 jobs, op in catalog | prepare | submit | list | log, body
                                  {"relay", ...}; checked here (template, name, bounded params,
                                  confirm + digest for submit), again by the relay, which alone runs
                                  slurm/submit.sh (and only if started with --allow-submit)
    GET  /relay                   relay WebSocket (``Authorization: Bearer <relay token>``)

Agent channel, API v1 (stable JSON, :mod:`conatus.studio.agentapi`; session cookie, a user's API
token, or an agent token with a scope, ``Authorization: Bearer``; tokens never in a URL)
    GET  /api/v1/ping             liveness + clock of the hub (latency)
    GET  /api/v1/whoami           the caller: name, kind (user | agent), scope, runs reachable
    GET  /api/v1/runs             every run reachable, brief (state, t, iteration, dt, alerts...)
    GET  /api/v1/runs/<run>       one run in full (+ log and events tails: ``?log=&events=``)
    GET  /api/v1/runs/<run>/scalars   ``?since=<wall>&names=a,b&max=`` scalar records
    GET  /api/v1/runs/<run>/log   ``?n=`` log and events
    GET  /api/v1/runs/<run>/frame ``?name=&format=json|png`` latest frame of a view
    GET  /api/v1/runs/<run>/wait  ``?since=<wall>&timeout=<s>`` long poll: answers the run as
                                  soon as a record newer than ``since`` arrives (<= 55 s)
    POST /api/v1/runs/<run>/commands  {"cmd", "params", "confirm"}: declared commands only; a
                                  ``read`` token gets 403; stop and fork answer 409 with a
                                  ``confirm_token`` first, then succeed when repeated with
                                  ``"confirm": <token>`` (sessions may also send ``true``, the
                                  app's second tap); ``set`` only on a declared parameter, in
                                  its bounds
    GET  /api/v1/board            the dashboard: runs, the people and agents piloting them,
                                  relays, recent commands (read only)
    GET  /api/v1/audit            ``?n=`` the last audit entries (users only)
    ``<run>`` is a key ``relay:run_id`` or a run id that names one run.

Security model (see studio/hub/DEPLOY.md): TLS is terminated by the host or by ``--certfile``; outside
``--local`` every non-HTTPS request is refused; session cookies are HttpOnly, SameSite=Strict, bound to
the ``Host`` of the login, and named ``__Host-cn_session`` (Secure) over HTTPS, ``cn_session_<port>`` with
``--local``; the cookie alone opens only the app's pages: data, commands and ``/ws`` also need the
session key, which the app keeps in the storage of its own origin and sends in ``X-CN-Session-Key``
(another service on the same host name receives the cookie, not the key);
state-changing requests must be JSON from the same origin; failed logins are throttled per client;
commands are filtered against the protocol list (hub *and* relay), stop/fork need an explicit
confirmation, and every login and command is written to the audit log (who, when, what, result).
Only the telemetry plane crosses the hub; it never sees scientific data.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import secrets
import sys
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path

import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.web
import tornado.websocket

from conatus._private import private_dir
from conatus.telemetry import COMMANDS, RunState, default_spool
from conatus.telemetry.film import FilmIndex, thin_to_budget

from . import agentapi
from .auth import AuditLog, AuthDB, LoginThrottle, Principal, Sessions
from .compare import compare_runs
from .launch import JOB_OPS, MAX_PARAMS_BYTES, LaunchError, validate_request
from .page import render_page
from . import webapp

WEB_DEFAULT = webapp.classic_page()             # the repository's, else the shipped copy
CONFIRM = ("stop", "fork")          # commands that need an explicit confirmation
COOKIE = "cn_session"               # base name; see webapp.session_cookie_name
ACK_TIMEOUT_S = 10.0
JOB_ACK_TIMEOUT_S = {"submit": 150.0, "prepare": 45.0}   # submit.sh runs on the relay; others 15 s
FILM_MAX_FRAMES = 720               # film frames kept per run and name in memory (then thinned)
MAX_BODY_BYTES = 1 << 20            # HTTP request bodies (commands and jobs are a few kB)
MAX_WAITS = 4                       # concurrent long polls per caller (more: 429)
AUDIT_FIELD_BYTES = 512             # a command's cmd/params larger than this are not copied whole


class Hub:
    """Shared state: relays, runs (bounded), browsers, pending commands."""

    def __init__(self, auth: AuthDB, *, audit: AuditLog, local: bool, web: Path = WEB_DEFAULT,
                 max_scalars: int = 20000, max_logs: int = 2000, max_runs: int = 200,
                 session_ttl_s: float = 12 * 3600, app: Path | None = None,
                 film_mb: float = 512.0, film_max_frames: int = FILM_MAX_FRAMES) -> None:
        self.auth, self.audit, self.local, self.web = auth, audit, bool(local), Path(web)
        self.app = webapp.app_dir(app)                    # built Studio app (static files)
        # films in memory, bounded: <= film_max_frames per run and name, <= film_mb in all;
        # beyond, thinned (every second frame dropped, counted in each film's summary)
        self.films: dict[str, FilmIndex] = {}             # key "<relay>:<run_id>"
        self.film_cursors: dict[tuple[str, str], int] = {}
        self.film_max_bytes = int(film_mb * 2**20)
        self.film_max_frames = int(film_max_frames)
        self.max_scalars, self.max_logs, self.max_runs = int(max_scalars), int(max_logs), int(max_runs)
        self.sessions = Sessions(session_ttl_s)
        self.throttle = LoginThrottle()
        self.relays: dict[str, RelaySocket] = {}
        self.relay_info: dict[str, dict] = {}
        self.runs: dict[str, RunState] = {}               # key "<relay>:<run_id>"
        self.cursors: dict[tuple[str, str], int] = {}     # (relay, run_id) -> bytes ingested
        self.slurm: dict[str, dict] = {}
        self.clients: set = set()
        self.pending: dict[str, asyncio.Future] = {}
        # agent channel: two-step confirmations, who pilots what, recent audit, long-poll waiters
        self.confirmations = agentapi.Confirmations()
        self.pilots: dict[str, dict] = {}                 # principal label -> activity
        self.recent: deque = deque(maxlen=500)            # last audit entries (dashboard)
        self._waiters: set[asyncio.Future] = set()
        self.waiting: dict[str, int] = {}                 # open long polls per caller label
        self._ticker = None
        # behind a TLS-terminating proxy (the server runs with xheaders): which client address to
        # trust (Guard.client_ip). _serve sets it from the real server options.
        self.proxied = not self.local
        # a header holding the client address that the proxy itself writes (overwriting any copy
        # sent by the client), e.g. Fly-Client-IP on Fly.io; None: the last X-Forwarded-For entry
        self.client_ip_header: str | None = None

    # ------------------------------------------------------------------ audit, activity
    def log_audit(self, **entry) -> dict:
        """Write one audit entry (file + stdout, :class:`AuditLog`) and keep it for the board."""
        e = self.audit.write(**entry)
        self.recent.append(e)
        return e

    def seen(self, who: Principal, ip: str, action: str, key: str | None = None) -> None:
        """Remember what a user or an agent did last (the board's \"pilots\")."""
        p = self.pilots.setdefault(who.label, {"name": who.name, "kind": who.kind,
                                               "scope": who.scope, "requests": 0, "commands": 0})
        p.update(last_seen=time.time(), ip=ip, last_action=action)
        p["requests"] += 1
        if key is not None:
            p["last_run"] = key
        if action.startswith("command"):
            p["commands"] += 1

    def pilots_of(self, key: str, n: int = 3) -> list[dict]:
        """The last ``n`` distinct senders of a command to the run ``key`` (from the audit)."""
        out, seen = [], set()
        for e in reversed(self.recent):
            if e.get("event") == "command" and f"{e.get('relay')}:{e.get('run_id')}" == key:
                who = e.get("user")
                if who not in seen:
                    seen.add(who)
                    out.append({"who": who, "cmd": e.get("cmd"), "result": e.get("result"),
                                "wall": e.get("wall")})
                    if len(out) >= n:
                        break
        return out

    def run_view(self, key: str, *, brief: bool = False, log: int = 20, events: int = 20) -> dict:
        """The API v1 view of one run (:func:`agentapi.run_view`) with its relay's link state."""
        relay = key.partition(":")[0]
        return agentapi.run_view(key, self.runs[key], relay=relay,
                                 relay_connected=relay in self.relays, brief=brief, log=log,
                                 events=events, pilots=self.pilots_of(key))

    async def wait_change(self, timeout: float) -> None:
        """Return when the hub ingests anything new, or after ``timeout`` s (long poll)."""
        fut = asyncio.get_running_loop().create_future()
        self._waiters.add(fut)
        try:
            await asyncio.wait_for(fut, timeout)
        except TimeoutError:
            pass
        finally:
            self._waiters.discard(fut)

    def _wake(self) -> None:
        for f in list(self._waiters):
            if not f.done():
                f.set_result(None)

    def ensure_ticker(self) -> None:
        """Every 15 s, a ``tick`` to every browser: phones detect a dead socket (no tick for
        40 s: reconnect) and see the hub's clock."""
        if self._ticker is None:
            self._ticker = tornado.ioloop.PeriodicCallback(
                lambda: self.broadcast({"type": "tick", "wall": time.time()}), 15000)
            self._ticker.start()

    # ------------------------------------------------------------------ views for browsers
    @staticmethod
    def key(relay: str, run_id: str) -> str:
        """The run key ``<relay>:<run_id>`` used by the hub and the browsers."""
        return f"{relay}:{run_id}"

    def relays_view(self, addresses: bool = True) -> dict:
        """Known relays with their info and whether each is connected now (``addresses=False``:
        without their IP, what an agent token sees)."""
        return {name: {**{k: v for k, v in info.items() if addresses or k != "ip"},
                       "connected": name in self.relays} for name, info in self.relay_info.items()}

    def slurm_view(self) -> dict:
        """Merged squeue view of all relays (job ids prefixed by relay, errors joined)."""
        jobs, errors, when = [], [], None
        for relay, s in sorted(self.slurm.items()):
            jobs += [{**j, "id": f"{relay}:{j.get('id')}"} for j in s.get("jobs", [])]
            if s.get("error"):
                errors.append(f"{relay}: {s['error']}")
            when = max(when or 0, s.get("when") or 0) or None
        return {"when": when, "jobs": jobs, "error": "; ".join(errors) or None}

    def snapshot(self, user: str) -> dict:
        """Full state sent to a browser on connection: relays, Slurm jobs and every run."""
        names = ", ".join(f"{n} ({'up' if v['connected'] else 'down'})" for n, v in self.relays_view().items())
        return {"type": "snapshot", "mode": "hub", "local": self.local, "user": user,
                "relays": self.relays_view(),
                "spool": f"hub · relays: {names or 'none yet'}", "slurm": self.slurm_view(),
                "runs": {k: st.snapshot() for k, st in self.runs.items()}}

    def broadcast(self, msg: dict) -> None:
        """Send one JSON message to every browser; drops the ones found closed. Wakes the long
        polls of the agent API."""
        if msg.get("type") not in ("tick",):
            self._wake()
        texts: dict[bool, str] = {}
        for c in list(self.clients):
            lite = getattr(c, "lite", False)            # phones: light stream (webapp.lite)
            if lite not in texts:
                texts[lite] = json.dumps(webapp.lite(msg) if lite else msg, separators=(",", ":"))
            try:
                c.write_message(texts[lite])
            except tornado.websocket.WebSocketClosedError:
                self.clients.discard(c)

    # ------------------------------------------------------------------ relay side
    def relay_up(self, name: str, sock: RelaySocket, hello: dict) -> dict:
        """Register a relay on hello (closing any older link); returns the welcome with its cursors."""
        old = self.relays.get(name)
        if old is not None and old is not sock:
            old.close(4000, "replaced by a newer connection")
        self.relays[name] = sock
        self.relay_info[name] = {"host": hello.get("host"), "since": time.time(),
                                 "ip": sock.client_ip}
        self.broadcast({"type": "relays", "relays": self.relays_view()})
        cursors = {rid: c for (r, rid), c in self.cursors.items() if r == name}
        film = {rid: c for (r, rid), c in self.film_cursors.items() if r == name}
        return {"type": "welcome", "relay": name, "cursors": cursors, "film_cursors": film}

    def relay_down(self, name: str, sock: RelaySocket) -> None:
        """Unregister a relay socket if it is still the current one, and tell the browsers."""
        if self.relays.get(name) is sock:
            del self.relays[name]
            self.broadcast({"type": "relays", "relays": self.relays_view()})

    def _state(self, relay: str, run_id: str) -> RunState:
        k = self.key(relay, run_id)
        if k not in self.runs and len(self.runs) >= self.max_runs:   # evict the stalest run
            old = min(self.runs, key=lambda x: self.runs[x].last_wall)
            del self.runs[old]
            self.films.pop(old, None)
        return self.runs.setdefault(k, RunState(k, self.max_scalars, self.max_logs))

    def ingest_lines(self, relay: str, msg: dict) -> dict | None:
        """Ingest complete lines beyond the cursor; returns a resync request on a gap."""
        run_id = str(msg["run_id"])
        start, end, data = int(msg["start"]), int(msg["end"]), str(msg["data"])
        ck = (relay, run_id)
        if msg.get("reset"):                            # the file was truncated at the source
            self.cursors[ck] = 0
            self.runs.pop(self.key(relay, run_id), None)
        cur = self.cursors.get(ck, 0)
        if start > cur:                                 # a gap: never guess, ask again
            return {"type": "resync", "run_id": run_id, "cursor": cur}
        if end <= cur:                                  # already have all of it
            return None
        raw = data.encode("utf-8")
        if start < cur:                                 # partial overlap: drop what we have
            raw = raw[cur - start:]
        st = self._state(relay, run_id)
        items = []
        for line in raw.decode("utf-8", "replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                st.malformed += 1
                continue
            if not isinstance(rec, dict):
                st.malformed += 1
                continue
            rec = {**rec, "run_id": st.run_id, "relay": relay}
            st.ingest(rec)
            items.append(rec)
        self.cursors[ck] = end
        if items:
            self.broadcast({"type": "items", "items": items})
        return None

    def ingest_film(self, relay: str, msg: dict) -> dict | None:
        """Film lines pushed by a relay (same cursor rules as :meth:`ingest_lines`); indexes them
        in memory (bounded, thinned past the limits) and tells the browsers the film grew."""
        run_id = str(msg["run_id"])
        start, end, data = int(msg["start"]), int(msg["end"]), str(msg["data"])
        ck, key = (relay, run_id), self.key(relay, run_id)
        if msg.get("reset"):                            # the film file was replaced at the source
            self.film_cursors[ck] = 0
            self.films.pop(key, None)
        cur = self.film_cursors.get(ck, 0)
        if start > cur:
            return {"type": "resync", "run_id": run_id, "cursor": cur, "film": True}
        if end <= cur:
            return None
        raw = data.encode("utf-8")
        if start < cur:
            raw = raw[cur - start:]
        if key not in self.runs:                        # a film always belongs to a known run
            self._state(relay, run_id)
        film = self.films.get(key)
        if film is None:
            film = self.films[key] = FilmIndex(key, max_frames=self.film_max_frames)
        new = film.ingest(raw, cur)
        self.film_cursors[ck] = end
        thin_to_budget(self.films, self.film_max_bytes)
        for e in new:
            self.broadcast({"type": "film", "run_id": key, **e})
        return None

    def ingest_frame(self, relay: str, frame: dict) -> None:
        """File a frame pushed by a relay and forward it to the browsers (malformed ones ignored)."""
        if not isinstance(frame, dict) or "name" not in frame or "run_id" not in frame:
            return
        st = self._state(relay, str(frame["run_id"]))
        rec = {**frame, "run_id": st.run_id, "relay": relay}
        st.ingest(rec)
        self.broadcast({"type": "items", "items": [rec]})

    # ------------------------------------------------------------------ commands
    async def command(self, user: str | Principal, ip: str, key: str, body: dict, *,
                      api: int = 0) -> tuple[int, dict]:
        """Validate, forward to the relay, wait for its ack; audit every outcome.

        ``api=0``: the browser route ``/api/control`` (signed-in user; ``confirm: true`` is the
        app's second tap). ``api=1``: the agent channel; a bearer caller (agent or a user's own
        token) confirms stop/fork in two steps with the ``confirm_token`` of the first answer."""
        who = user if isinstance(user, Principal) else Principal(user, session=True)
        cmd, params, confirm = body.get("cmd"), body.get("params"), body.get("confirm")
        relay, _, run_id = key.partition(":")
        entry = dict(event="command", user=who.label, ip=ip, relay=relay, run_id=run_id,
                     cmd=_bounded(cmd), params=_bounded(params))
        if api:
            entry["api"] = api
        self.seen(who, ip, f"command {cmd}", key)

        def refuse(code: int, msg: str, why: str, **extra) -> tuple[int, dict]:
            self.log_audit(**entry, result=f"refused: {why}")
            return code, {"ok": False, "error": msg, "code": code, **extra}

        if cmd not in COMMANDS:
            return refuse(400, f"unknown command {cmd!r}; known: {list(COMMANDS)}", "unknown command")
        if key not in self.runs:
            return refuse(404 if api else 400, f"unknown run {key!r}", "unknown run")
        if not who.can_command(key):
            return refuse(403, f"{who.label} has a {who.scope!r} token"
                          + ("" if who.can_see(key) else " not valid for this run")
                          + ": commands are refused", "scope")
        try:
            agentapi.check_command(self.runs[key], cmd, params)
        except agentapi.CommandRefused as err:
            return refuse(err.code, str(err), str(err)[:120])
        if agentapi.needs_confirmation(cmd):
            confirmed = (confirm is True and (who.session or not api)) or (
                isinstance(confirm, str) and self.confirmations.check(confirm, who.label, key, cmd,
                                                                      params))
            if not confirmed:
                tok = self.confirmations.issue(who.label, key, cmd, params)
                why = "confirmation required" if confirm in (None, False, True) else "bad confirmation"
                return refuse(409, f"{cmd} requires a confirmation: send the same command again "
                                   f"with \"confirm\": \"{tok}\" within "
                                   f"{agentapi.CONFIRM_TTL_S:.0f} s"
                                   + ("" if api else " (or confirm: true from the app)"),
                              why, needs_confirmation=True, confirm_token=tok)
        sock = self.relays.get(relay)
        if sock is None:
            return refuse(503, f"relay {relay!r} is not connected; command not sent", "relay offline")
        cid = uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        self.pending[cid] = fut
        t0 = time.monotonic()
        try:
            await sock.write_message(json.dumps({"type": "command", "id": cid, "run_id": run_id,
                                                 "cmd": cmd, "params": params, "user": who.label}))
            ack = await asyncio.wait_for(fut, ACK_TIMEOUT_S)
        except (TimeoutError, tornado.websocket.WebSocketClosedError):
            self.log_audit(**entry, id=cid, result="sent, no acknowledgement from the relay")
            return 504, {"ok": False, "code": 504,
                         "error": "sent, but the relay did not acknowledge it (state unknown)"}
        finally:
            self.pending.pop(cid, None)
        if not ack.get("ok"):
            self.log_audit(**entry, id=cid, result=f"refused by relay: {ack.get('error')}")
            return 400, {"ok": False, "code": 400, "error": f"relay refused: {ack.get('error')}"}
        self.log_audit(**entry, id=cid, result="written to control file")
        self.broadcast({"type": "control_sent", "run_id": key, "line": ack.get("line", {}),
                        "user": who.label})
        return 200, {"ok": True, "id": cid, "line": ack.get("line"),
                     "relay_ms": round(1000 * (time.monotonic() - t0), 1),
                     "note": "written to the run's control file; the run applies it at its next "
                             "check (see its events: command_received, param_set, ...)"}

    # ------------------------------------------------------------------ jobs (S2)
    async def job(self, user: str, ip: str, op: str, body: dict) -> tuple[int, dict]:
        """Validate a job operation, forward it to one relay, wait for its answer; audit
        prepare / submit (and every refusal of them)."""
        relay = body.get("relay")
        if relay is None and len(self.relays) == 1:
            relay = next(iter(self.relays))               # the only connected relay
        args = {k: v for k, v in body.items() if k != "relay"}
        entry = dict(event="job", user=user, ip=ip, relay=_bounded(relay), op=_bounded(op),
                     template=_bounded(args.get("template")), name=_bounded(args.get("name")),
                     params=_bounded(args.get("params")))
        audited = op in ("prepare", "submit")

        def refuse(code: int, msg: str) -> tuple[int, dict]:
            if audited:
                self.log_audit(**entry, result=f"refused: {msg}")
            return code, {"error": msg}

        if op not in JOB_OPS:
            return refuse(400, f"unknown job operation {op!r}; known: {list(JOB_OPS)}")
        if len(json.dumps(args)) > 4 * MAX_PARAMS_BYTES:
            return refuse(400, "job request too large")
        if op in ("prepare", "submit"):
            try:                                          # structural checks (no file here)
                validate_request(args.get("template"), args.get("name"), args.get("params"))
            except LaunchError as err:
                return refuse(err.code, str(err))
        if op == "submit":
            if args.get("confirm") is not True:
                return refuse(409, "submission requires an explicit confirmation (confirm: true)")
            if not isinstance(args.get("digest"), str) or len(args["digest"]) != 64:
                return refuse(409, "submission requires the digest of the prepared job")
        sock = self.relays.get(relay) if isinstance(relay, str) else None
        if sock is None:
            return refuse(503, f"relay {relay!r} is not connected (connected: {sorted(self.relays)})")
        cid = uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        self.pending[cid] = fut
        try:
            await sock.write_message(json.dumps({"type": "job", "id": cid, "op": op, "args": args,
                                                 "user": user}))
            ack = await asyncio.wait_for(fut, JOB_ACK_TIMEOUT_S.get(op, 15.0))
        except (TimeoutError, tornado.websocket.WebSocketClosedError):
            if audited:
                self.log_audit(**entry, id=cid, result="sent, no answer from the relay")
            return 504, {"error": "sent, but the relay did not answer (state unknown; not resent)"}
        finally:
            self.pending.pop(cid, None)
        if not ack.get("ok"):
            if audited:
                self.log_audit(**entry, id=cid, result=f"refused by relay: {ack.get('error')}")
            return int(ack.get("code") or 400), {"error": f"relay: {ack.get('error')}"}
        result = ack.get("result") or {}
        if audited:
            self.log_audit(**entry, id=cid, result="submitted" if op == "submit" else "ok",
                             run_id=result.get("run_id"), jobid=result.get("jobid"),
                             digest=result.get("digest"))
        if op == "submit":
            self.broadcast({"type": "job_submitted", "user": user, "job": {
                "relay": relay, **{k: result.get(k) for k in ("run_id", "jobid", "template", "name",
                                                               "parent")}}})
        return 200, {**result, "relay": relay}


def _bounded(value, limit: int = AUDIT_FIELD_BYTES):
    """A caller-supplied value as written to the audit: itself if its JSON is small, else a
    note with its size (earlier versions: a 100 MB body used to be copied into the audit line)."""
    try:
        n = len(json.dumps(value))
    except (TypeError, ValueError):
        return f"<not JSON: {type(value).__name__}>"
    return value if n <= limit else f"<{n} bytes, not copied>"


# ====================================================================== HTTP layer
# (the hub's Content-Security-Policy is the app's: no inline script; /classic relaxes it)
class Guard:
    """Mixin: HTTPS only (outside --local), security headers, who is calling."""

    hub: Hub

    def initialize(self, hub: Hub) -> None:
        """Keep the shared :class:`Hub` (tornado handler hook)."""
        self.hub = hub

    def set_default_headers(self) -> None:
        """Security headers on every response (nosniff, same-origin referrer, no framing, no-store, CSP)."""
        self.set_header("X-Content-Type-Options", "nosniff")
        self.set_header("Referrer-Policy", "same-origin")  # "no-referrer" made browsers send Origin: null
        self.set_header("X-Frame-Options", "DENY")
        self.set_header("Cache-Control", "no-store")
        self.set_header("Content-Security-Policy", webapp.CSP_APP)

    def prepare(self) -> None:
        """With --local, answer only a loopback ``Host`` (403 otherwise: a page of another site
        whose DNS name is rebound to 127.0.0.1 reaches the local hub with ITS name in ``Host``;
        security the rule of the local Studio server, ``webapp.loopback_host``).
        Outside --local, refuse plain HTTP (403) and set HSTS."""
        if self.hub.local and not webapp.loopback_host(self.request.host or ""):
            raise tornado.web.HTTPError(403, "Host is not this machine's loopback (DNS rebinding)")
        if not self.hub.local:
            if self.request.protocol != "https":
                raise tornado.web.HTTPError(403, "HTTPS required")
            self.set_header("Strict-Transport-Security", "max-age=31536000")

    @property
    def client_ip(self) -> str:
        """The client address used by the throttles, the audit and the board.

        Behind the proxy (``xheaders``), tornado prefers ``X-Real-Ip`` to ``X-Forwarded-For``, and
        a proxy that does not set it (Caddy, studio/hub/DEPLOY.md) passes the client's own header
        through: any client could pick its address, a fresh one per password guess, defeating
        the login throttle and forging the audit. Trusted instead: the
        header named by ``Hub.client_ip_header`` (``--client-ip-header``, a header the proxy
        overwrites: Fly.io writes the client in ``Fly-Client-IP``, while the last
        ``X-Forwarded-For`` entry there is the app's own edge address), else the last
        ``X-Forwarded-For`` entry (Caddy writes the client there and drops what an untrusted client
        sent); a value that is not an IP address falls back to the socket peer. This assumes the
        hub is reachable only through the proxy (DEPLOY.md: bound to 127.0.0.1 behind Caddy): a
        direct client could write either header, as it could already write X-Forwarded-Proto."""
        if not self.hub.proxied:
            return self.request.remote_ip
        name = self.hub.client_ip_header
        raw = self.request.headers.get(name, "") if name else self.request.headers.get("X-Forwarded-For", "")
        ip = (raw.strip() if name else raw.split(",")[-1].strip())
        if ip and tornado.netutil.is_valid_ip(ip):
            return ip
        ctx = getattr(self.request.connection, "context", None)
        return getattr(ctx, "_orig_remote_ip", None) or self.request.remote_ip

    @property
    def cookie_name(self) -> str:
        """The session cookie of this hub as the browser addresses it (0.1.2):
        ``__Host-cn_session`` over HTTPS; ``cn_session_<port>`` with ``--local``, so two Studios
        on one loopback name do not share or overwrite one cookie."""
        return webapp.session_cookie_name(self.request.host or "", secure=not self.hub.local)

    def session_key(self) -> str | None:
        """The session key the request presents (``X-CN-Session-Key`` header)."""
        return self.request.headers.get(webapp.SESSION_KEY_HEADER)

    def session_user(self, *, page: bool = False) -> str | None:
        """The user of the session cookie, if the session was opened at this ``Host`` and the
        request presents its session key; ``page=True``: the cookie is enough
        (the app's pages, no data)."""
        return self.hub.sessions.user(self.get_cookie(self.cookie_name), origin=self.request.host,
                                      key=self.session_key(), page=page)

    def page_user(self) -> str | None:
        """Who may load a page of the app: a session cookie (no key needed), else a user token."""
        return self.session_user(page=True) or self.get_current_user()

    def get_current_user(self) -> str | None:
        """The user of the session cookie, else of a ``Bearer`` API token, else None."""
        user = self.session_user()
        if user:
            return user
        authz = self.request.headers.get("Authorization", "")
        if authz.startswith("Bearer "):
            return self.hub.auth.user_for_token(authz[7:].strip())
        return None

    def same_origin(self) -> bool:
        """State-changing requests must come from this site's own pages.

        Browsers say it with ``Sec-Fetch-Site`` (``same-origin``, or ``none`` for a request the user
        typed); when that header is absent, an ``Origin`` header, if present, must be this host. A
        form posted by our own page may carry ``Origin: null`` (refusing it would refuse a login in real browsers),
        so ``Origin`` alone is not trusted when ``Sec-Fetch-Site`` says otherwise."""
        return webapp.same_origin(self.request.headers, self.request.host)


class Base(Guard, tornado.web.RequestHandler):
    """HTTP handler base: :class:`Guard` plus errors rendered as JSON."""
    def write_error(self, status_code: int, **kw) -> None:
        """Finish with ``{"error": reason}``, the HTTPError message when there is one."""
        reason = self._reason
        if "exc_info" in kw and isinstance(kw["exc_info"][1], tornado.web.HTTPError):
            reason = kw["exc_info"][1].log_message or reason
        self.finish({"error": reason})


class Health(tornado.web.RequestHandler):
    """``GET /healthz``: unauthenticated liveness probe."""
    def get(self) -> None:
        """Answer ``ok``."""
        self.write("ok")


# Sign-in page, in the art direction of the site (brand charter lab black, silver, fluorescein; IM Fell
# for the title, IBM Plex Sans for the text). Self-contained: inline style only (no script), fonts
# from the app bundle at fixed URLs (studio/app/build.sh copies them; system fonts if absent).
LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0B0B0B"><title>Conatus Studio · sign in</title>
<link rel="icon" href="/icons/mark-silver.svg" type="image/svg+xml">
<link rel="manifest" href="/manifest.webmanifest">
<style>
@font-face{font-family:"IM Fell English";font-style:normal;font-weight:400;font-display:swap;src:url(/fonts/im-fell-english-latin-400-normal.woff2) format("woff2")}
@font-face{font-family:"IBM Plex Sans";font-style:normal;font-weight:400;font-display:swap;src:url(/fonts/ibm-plex-sans-latin-400-normal.woff2) format("woff2")}
@font-face{font-family:"IBM Plex Sans";font-style:normal;font-weight:500;font-display:swap;src:url(/fonts/ibm-plex-sans-latin-500-normal.woff2) format("woff2")}
:root{--bg:#0B0B0B;--text:#EDEBE6;--muted:#A3A29D;--rule:#2A2A28;--surface:#161615;--signal:#C6F03C;--bad:#D4574A;color-scheme:dark}
*{box-sizing:border-box}html,body{margin:0;min-height:100%}
body{background:var(--bg);color:var(--text);font:16px/1.55 "IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
padding:max(16px,env(safe-area-inset-top)) 16px 16px;display:grid;place-items:center;min-height:100dvh}
form{width:100%;max-width:22rem}
.mark{display:flex;align-items:center;gap:.6rem;margin:0 0 2.2rem}
.mark img{width:28px;height:28px}
h1{font:400 2.1rem/1.1 "IM Fell English",Georgia,serif;margin:0}
label{display:block;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:1.1rem 0 .35rem}
input{width:100%;font:inherit;color:var(--text);background:transparent;border:0;border-bottom:1px solid var(--rule);
padding:.5rem 0;border-radius:0;outline:none}
input:focus{border-bottom-color:var(--signal)}
.or{margin:1.4rem 0 0;font-size:.8rem;color:var(--muted)}
button{margin-top:2rem;width:100%;min-height:44px;font:500 .95rem "IBM Plex Sans",Arial,sans-serif;color:var(--bg);
background:var(--text);border:0;border-radius:2px;cursor:pointer}
button:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
.err{color:var(--bad);font-size:.9rem;margin-top:1rem}
.note{color:var(--muted);font-size:.8rem;margin-top:2rem;border-top:1px solid var(--rule);padding-top:1rem}
</style></head>
<body><form method="post" action="/login">
<div class="mark"><img src="/icons/mark-silver.svg" alt=""><h1>Studio</h1></div>
<label for="u">User</label><input id="u" name="user" autocomplete="username" autocapitalize="none" spellcheck="false">
<label for="p">Password</label><input id="p" name="password" type="password" autocomplete="current-password">
<p class="or">or an access token</p>
<label for="t">Token</label><input id="t" name="token" type="password" autocomplete="off">
<button type="submit">Sign in</button>%ERR%
<p class="note">Commands sent from the Studio (pause, resume, checkpoint, stop, fork) are logged with your name.</p>
</form></body></html>"""


# Answer to a successful sign-in (review F11): the session key goes to the page in a meta tag,
# and /login-done.js (no inline script: CSP) keeps it in the storage of this origin, then opens the
# app. The key is never in a URL, a cookie or a log.
LOGIN_DONE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="cn-session-key" content="%KEY%"><title>Conatus Studio</title>
<script src="/login-done.js"></script></head>
<body style="background:#0B0B0B;color:#EDEBE6;font:16px system-ui;padding:16px">
<noscript>The Studio needs JavaScript.</noscript></body></html>"""
LOGIN_DONE_JS = """// Conatus Studio: keep the session key of this sign-in in this origin's storage, then open the app.
(function () {
  var m = document.querySelector('meta[name="cn-session-key"]'), k = m ? m.content : "";
  try { localStorage.setItem("cn_session_key", k); } catch (e) {
    try { sessionStorage.setItem("cn_session_key", k); } catch (e2) { /* the app will ask to sign in */ }
  }
  location.replace("/");
})();
"""


class LoginDone(Base):
    """``GET /login-done.js``: the script of the sign-in answer (public: code, no data)."""
    def get(self) -> None:
        """Serve the script."""
        self.set_header("Content-Type", "text/javascript; charset=utf-8")
        self.write(LOGIN_DONE_JS)


class Login(Base):
    """``/login``: sign-in page and password/token login (throttled, audited)."""
    def get(self) -> None:
        """Serve the sign-in page."""
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.write(LOGIN_PAGE.replace("%ERR%", ""))

    def post(self) -> None:
        """Check a token or user/password (same origin, throttled, audited); set the session cookie."""
        ip = self.client_ip
        if not self.same_origin():
            raise tornado.web.HTTPError(403, "cross-origin login refused")
        user_in = self.get_body_argument("user", "").strip()[:64]
        password = self.get_body_argument("password", "")
        token = self.get_body_argument("token", "").strip()
        # a valid 256-bit access token is accepted even when this address is throttled (on a
        # compute node every client is 127.0.0.1); passwords are throttled before hashing
        user = self.hub.auth.user_for_token(token) if token else None
        if user is None and self.hub.throttle.blocked(ip):
            self.hub.audit.write(event="login", ip=ip, result="throttled")
            self.set_status(429)
            self.set_header("Content-Type", "text/html; charset=utf-8")
            self.write(LOGIN_PAGE.replace("%ERR%", '<div class="err">Too many failures; wait 15 min.</div>'))
            return
        if user is None and not token and user_in and password:
            user = user_in if self.hub.auth.check_password(user_in, password) else None
        if user is None:
            self.hub.throttle.failure(ip)
            self.hub.audit.write(event="login", ip=ip, user=user_in or None, result="failed")
            self.set_status(401)
            self.set_header("Content-Type", "text/html; charset=utf-8")
            self.write(LOGIN_PAGE.replace("%ERR%", '<div class="err">Wrong credentials.</div>'))
            return
        if not token:                                   # a token login does not wipe the
            self.hub.throttle.success(ip)               # password failures of a shared address
        self.hub.audit.write(event="login", ip=ip, user=user, result="ok")
        cookie, key = self.hub.sessions.create(user, origin=self.request.host)
        self.set_cookie(self.cookie_name, cookie,
                        httponly=True, secure=not self.hub.local, samesite="Strict", path="/")
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.write(LOGIN_DONE_PAGE.replace("%KEY%", key))        # URL-safe characters only


class Logout(Base):
    """``POST /logout``: end the browser session."""
    def post(self) -> None:
        """Drop the session (cookie and key required) and clear the cookie (same origin only,
        audited); redirect to /login."""
        if not self.same_origin():
            raise tornado.web.HTTPError(403, "cross-origin request refused")
        self.hub.audit.write(event="logout", ip=self.client_ip, user=self.current_user)
        if self.session_user():                         # the cookie alone cannot end a session
            self.hub.sessions.drop(self.get_cookie(self.cookie_name))
        self.clear_cookie(self.cookie_name, path="/", secure=not self.hub.local, samesite="Strict")
        self.redirect("/login")


class Classic(Base):
    """``GET /classic``: the former single-file page (inline scripts: its own CSP)."""
    def get(self) -> None:
        """Serve the classic page to a signed-in user; redirect to /login otherwise."""
        if not self.page_user():
            self.redirect("/login")
            return
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.set_header("Content-Security-Policy", webapp.CSP_CLASSIC)
        self.write(render_page(self.hub.web))    # view3d.js inlined (studio.page)


class App(Base):
    """The Studio app: pages (``/``, ``/run/<key>``, ``/compare``) for signed-in users only;
    its static files (code, fonts, icons, manifest, service worker: no data) for anyone, so the
    sign-in page can use the fonts and the app can be installed."""
    def get(self, path: str = "") -> None:
        """Index page (session required) or one file of the bundle (404 if absent)."""
        dist = self.hub.app
        if webapp.is_app_route("/" + path):
            if not self.page_user():
                self.redirect("/login")
                return
            got = webapp.static_file(dist, "index.html")
            self.set_header("Content-Type", "text/html; charset=utf-8")
            self.write(got[0] if got else webapp.missing_app_page(dist))
            return
        got = webapp.static_file(dist, path)
        if got is None:
            raise tornado.web.HTTPError(404, f"no file {path!r} in the Studio app")
        body, ctype, immutable = got
        self.set_header("Content-Type", ctype)
        self.set_header("Cache-Control", "public, max-age=31536000, immutable" if immutable
                        else "no-cache")
        self.write(body)


class Api(Base):
    """Base of the JSON API handlers: authentication required."""
    def prepare(self) -> None:
        """Guard checks, then 401 unless a user is signed in."""
        super().prepare()
        if not self.current_user:
            raise tornado.web.HTTPError(401, "authentication required")


class Runs(Api):
    """``GET /api/runs``: summaries of the runs the hub holds."""
    def get(self) -> None:
        """Write the summary of every run, keyed by run key."""
        self.write({k: st.summary() for k, st in self.hub.runs.items()})


class WhoAmI(Api):
    """``GET /api/whoami``: the calling user, the mode and the relays."""
    def get(self) -> None:
        """Write the calling user, hub/local mode and the relays view."""
        self.write({"user": self.current_user, "mode": "local" if self.hub.local else "hub",
                    "relays": self.hub.relays_view()})


class Control(Api):
    """``POST /api/control/<run key>``: a run command, forwarded to its relay."""
    async def post(self, key: str) -> None:
        """Check origin and JSON body, run :meth:`Hub.command`; write its status and reply."""
        if not self.same_origin():
            raise tornado.web.HTTPError(403, "cross-origin request refused")
        ctype = self.request.headers.get("Content-Type", "")
        if not ctype.startswith("application/json"):
            raise tornado.web.HTTPError(415, "Content-Type must be application/json")
        try:
            body = json.loads(self.request.body or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            raise tornado.web.HTTPError(400, "body must be a JSON object") from None
        code, out = await self.hub.command(self.current_user, self.client_ip, key, body)
        self.set_status(code)
        self.write(out)


class Film(Api):
    """``GET /api/film/<key>[?name=]`` and ``/api/film/<key>/frames?name=&from=&to=&step=``."""
    def get(self, key: str, frames: str | None = None) -> None:
        """Film summary, index or frames of one run (``webapp.film_answer``)."""
        args = {k: self.get_query_argument(k) for k in self.request.query_arguments}
        code, out = webapp.film_answer(self.hub.films.get(key), key, args, frames=bool(frames))
        self.set_status(code)
        self.write(out)


class Compare(Api):
    """``GET /api/compare``: one scalar of several runs on a common axis."""
    def get(self) -> None:
        """Compare runs (query: runs, key, x, max_points) via ``compare_runs``; 400 on bad input."""
        ids = [r for r in self.get_query_argument("runs", "").split(",") if r]
        try:
            self.write(compare_runs(self.hub.runs, ids, self.get_query_argument("key", ""),
                                    x=self.get_query_argument("x", "t"),
                                    max_points=int(self.get_query_argument("max_points", "2000"))))
        except ValueError as err:
            raise tornado.web.HTTPError(400, str(err)) from err


class NoRegistry(Api):
    """``GET /api/registry...``: the run registry stays on the cluster; the hub answers an
    empty list with the reason (the app's Details panel asks for it)."""
    def get(self, rest: str = "") -> None:
        """Empty answer, explicit."""
        self.write({"runs": [], "registry": None,
                    "note": "the hub holds no run registry (conatus.runs stays on the cluster; "
                            "the local Studio server shows it)"})


class Jobs(Api):
    """``POST /api/jobs/<op>``: whitelisted job operations, forwarded to a relay."""
    async def post(self, op: str) -> None:
        """Check origin and JSON body, run :meth:`Hub.job`; write its status and reply."""
        if not self.same_origin():
            raise tornado.web.HTTPError(403, "cross-origin request refused")
        if not self.request.headers.get("Content-Type", "").startswith("application/json"):
            raise tornado.web.HTTPError(415, "Content-Type must be application/json")
        try:
            body = json.loads(self.request.body or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            raise tornado.web.HTTPError(400, "body must be a JSON object") from None
        code, out = await self.hub.job(self.current_user, self.client_ip, op, body)
        self.set_status(code)
        self.write(out)


# ====================================================================== agent channel, API v1
class V1(Base):
    """Base of ``/api/v1``: a principal is required (session, user token or agent token); errors
    are ``{"ok": false, "error", "code"}``; failed bearer tokens are throttled per client."""
    who: Principal

    def write_error(self, status_code: int, **kw) -> None:
        """``{"ok": false, "error": reason, "code": status}``."""
        reason = self._reason
        if "exc_info" in kw and isinstance(kw["exc_info"][1], tornado.web.HTTPError):
            reason = kw["exc_info"][1].log_message or reason
        self.finish({"ok": False, "error": reason, "code": status_code, "api": agentapi.API_VERSION})

    def prepare(self) -> None:
        """Guard checks; then the caller (401 without one, 429 after repeated bad tokens)."""
        super().prepare()
        ip = self.client_ip
        user = self.session_user()
        if user:
            self.who = Principal(user, session=True)
            return
        authz = self.request.headers.get("Authorization", "")
        if not authz.startswith("Bearer "):
            raise tornado.web.HTTPError(401, "authentication required: session cookie or "
                                             "'Authorization: Bearer <token>'")
        # the token is checked first: a valid one is never refused because someone else at the
        # same address failed (earlier versions: on a compute node every client is 127.0.0.1, so five bad
        # tokens from any user of the node locked the real agent out for 15 min)
        who = self.hub.auth.principal_for_token(authz[7:].strip())
        if who is None:
            if self.hub.throttle.blocked("api:" + ip):
                raise tornado.web.HTTPError(429, "too many bad tokens from this address; wait 15 min")
            self.hub.throttle.failure("api:" + ip)
            self.hub.log_audit(event="api_auth", ip=ip, result="refused: bad token")
            raise tornado.web.HTTPError(401, "bad token")
        self.who = who

    def key(self, ref: str) -> str:
        """The run key named by ``ref`` among the runs this caller may see (404 otherwise)."""
        visible = [k for k in self.hub.runs if self.who.can_see(k)]
        try:
            return agentapi.resolve(visible, ref)
        except KeyError as err:
            raise tornado.web.HTTPError(404, str(err.args[0])) from None

    def answer(self, body: dict, code: int = 200) -> None:
        """Write a JSON answer tagged with the API version."""
        self.set_status(code)
        self.write({"api": agentapi.API_VERSION, **body})

    def int_arg(self, name: str, default: int, lo: int, hi: int) -> int:
        """An integer query argument clamped to [lo, hi] (400 if not an integer)."""
        try:
            return max(lo, min(hi, int(self.get_query_argument(name, str(default)))))
        except ValueError:
            raise tornado.web.HTTPError(400, f"{name} must be an integer") from None


class V1Ping(V1):
    """``GET /api/v1/ping``: the hub's clock (latency, clock skew) and relays up."""
    def get(self) -> None:
        """Answer the hub's wall clock and the number of relays connected."""
        self.answer({"ok": True, "wall": time.time(), "relays_up": len(self.hub.relays),
                     "mode": "local" if self.hub.local else "hub"})


class V1WhoAmI(V1):
    """``GET /api/v1/whoami``: the caller as the hub sees it."""
    def get(self) -> None:
        """Name, kind, scope, reachable run patterns, session or token."""
        w = self.who
        self.answer({"ok": True, "name": w.name, "label": w.label, "kind": w.kind,
                     "scope": w.scope, "runs": list(w.runs) if w.runs else None,
                     "session": w.session})


class V1Runs(V1):
    """``GET /api/v1/runs``: brief view of every reachable run, live ones first."""
    def get(self) -> None:
        """Runs sorted by state (running, paused, stale, ended) then most recent."""
        self.hub.seen(self.who, self.client_ip, "runs")
        order = {s: i for i, s in enumerate(agentapi.STATES)}
        runs = [self.hub.run_view(k, brief=True) for k in self.hub.runs if self.who.can_see(k)]
        runs.sort(key=lambda r: (order.get(r["state"], 9), -(r["last_wall"] or 0)))
        self.answer({"ok": True, "now": time.time(), "runs": runs,
                     "relays": self.hub.relays_view(addresses=self.who.kind == "user")})


class V1Run(V1):
    """``GET /api/v1/runs/<run>``: one run in full."""
    def get(self, ref: str) -> None:
        """The run view with ``?log=`` and ``?events=`` tails (default 20, at most 500)."""
        key = self.key(ref)
        self.hub.seen(self.who, self.client_ip, "status", key)
        self.answer({"ok": True, "now": time.time(), "run": self.hub.run_view(
            key, log=self.int_arg("log", 20, 0, 500), events=self.int_arg("events", 20, 0, 500))})


class V1RunPart(V1):
    """``GET /api/v1/runs/<run>/(scalars|log|frame|wait)``."""
    async def get(self, ref: str, part: str) -> None:
        """Dispatch on the part asked for."""
        key = self.key(ref)
        st = self.hub.runs[key]
        self.hub.seen(self.who, self.client_ip, part, key)
        if part == "scalars":
            try:
                since = float(self.get_query_argument("since", "0") or 0)
            except ValueError:
                raise tornado.web.HTTPError(400, "since must be a wall time (s)") from None
            names = [n for n in self.get_query_argument("names", "").split(",") if n]
            rows = [r for r in st.scalars if float(r.get("wall") or 0) > since]
            n_max = self.int_arg("max", 2000, 1, 20000)
            stride = max(1, -(-len(rows) // n_max))
            sel = rows[::stride]
            if rows and sel[-1] is not rows[-1]:
                sel.append(rows[-1])
            keep = {"wall", "t", "iteration", *names}
            out = [{k: v for k, v in r.items() if (not names and k not in ("v", "kind", "run_id",
                                                                             "relay")) or k in keep}
                   for r in sel]
            self.answer({"ok": True, "key": key, "count": len(rows), "stride": stride, "rows": out})
        elif part == "log":
            n = self.int_arg("n", 100, 1, 2000)
            self.answer({"ok": True, "key": key,
                         "log": [agentapi._rec_brief(r) for r in list(st.logs)[-n:]],
                         "events": [agentapi._rec_brief(r) for r in list(st.events)[-n:]]})
        elif part == "frame":
            name = self.get_query_argument("name", "") or (
                next((v.get("name") for v in (st.header or {}).get("views") or []
                      if v.get("name") in st.frames), None) or next(iter(sorted(st.frames)), None))
            fr = st.frames.get(name) if name else None
            if fr is None:
                raise tornado.web.HTTPError(404, f"no frame {name!r} for {key!r} (frames: "
                                                 f"{sorted(st.frames)})")
            if self.get_query_argument("format", "json") == "png":
                try:
                    png = agentapi.frame_png(fr)
                except ValueError as err:
                    raise tornado.web.HTTPError(415, str(err)) from None
                self.set_header("Content-Type", "image/png")
                self.write(png)
                return
            self.answer({"ok": True, "key": key, "name": name, "frame": fr})
        else:                                           # wait: long poll
            try:
                since = float(self.get_query_argument("since", "0") or 0)
                timeout = min(55.0, max(0.0, float(self.get_query_argument("timeout", "25"))))
            except ValueError:
                raise tornado.web.HTTPError(400, "since and timeout are numbers (s)") from None
            busy = self.hub.waiting.get(self.who.label, 0)
            if busy >= MAX_WAITS:                       # one caller cannot pile up open polls
                raise tornado.web.HTTPError(429, f"{busy} long polls already open for "
                                                 f"{self.who.label} (at most {MAX_WAITS})")
            self.hub.waiting[self.who.label] = busy + 1
            end = time.monotonic() + timeout
            try:
                while (self.hub.runs.get(key) is st and float(st.last_wall or 0) <= since
                       and time.monotonic() < end and not self.request.connection.stream.closed()):
                    await self.hub.wait_change(end - time.monotonic())
            finally:
                left = self.hub.waiting.get(self.who.label, 1) - 1
                if left > 0:
                    self.hub.waiting[self.who.label] = left
                else:
                    self.hub.waiting.pop(self.who.label, None)
            if key not in self.hub.runs:
                raise tornado.web.HTTPError(404, f"run {key!r} is gone (evicted or reset)")
            self.answer({"ok": True, "now": time.time(),
                         "changed": float(st.last_wall or 0) > since, "run": self.hub.run_view(key)})


class V1Commands(V1):
    """``POST /api/v1/runs/<run>/commands``: one declared command (see :meth:`Hub.command`)."""
    async def post(self, ref: str) -> None:
        """Same-origin JSON only (sessions); bearer callers need no cookie, so no CSRF."""
        if not self.same_origin():
            raise tornado.web.HTTPError(403, "cross-origin request refused")
        if not self.request.headers.get("Content-Type", "").startswith("application/json"):
            raise tornado.web.HTTPError(415, "Content-Type must be application/json")
        try:
            body = json.loads(self.request.body or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            raise tornado.web.HTTPError(400, "body must be a JSON object") from None
        key = self.key(ref)
        code, out = await self.hub.command(self.who, self.client_ip, key, body, api=1)
        self.answer({"key": key, **out}, code)


class V1Board(V1):
    """``GET /api/v1/board``: the dashboard (read only): runs, pilots, relays, recent commands."""
    def get(self) -> None:
        """Agents see the runs they may see and no address of anyone."""
        order = {s: i for i, s in enumerate(agentapi.STATES)}
        runs = [self.hub.run_view(k, brief=True) for k in self.hub.runs if self.who.can_see(k)]
        runs.sort(key=lambda r: (order.get(r["state"], 9), -(r["last_wall"] or 0)))
        user = self.who.kind == "user"
        def pilot(v: dict) -> dict:                     # agents: no address, no unseen run
            if user:
                return v
            return {kk: vv for kk, vv in v.items() if kk != "ip"
                    and not (kk == "last_run" and not self.who.can_see(str(vv)))}
        pilots = [{"label": k, **pilot(v)}
                  for k, v in sorted(self.hub.pilots.items(), key=lambda kv: -kv[1].get("last_seen", 0))]
        recent = [{k: v for k, v in e.items() if user or k != "ip"} for e in self.hub.recent
                  if (e.get("event") == "command" or (user and e.get("event") == "job"))
                  and self.who.can_see(f"{e.get('relay')}:{e.get('run_id')}")][-50:]   # agents: no jobs
        self.answer({"ok": True, "now": time.time(), "runs": runs, "pilots": pilots,
                     "relays": self.hub.relays_view(addresses=user), "recent": recent[::-1],
                     "who": self.who.label})


class V1Audit(V1):
    """``GET /api/v1/audit?n=``: the last audit entries (users only)."""
    def get(self) -> None:
        """403 for agent tokens."""
        if self.who.kind != "user":
            raise tornado.web.HTTPError(403, "the audit log is for users, not agent tokens")
        n = self.int_arg("n", 100, 1, 500)
        self.answer({"ok": True, "entries": list(self.hub.recent)[-n:][::-1]})


class BrowserSocket(Guard, tornado.websocket.WebSocketHandler):
    """``/ws``: live updates to a signed-in browser (listen only). Browsers cannot set headers on
    a WebSocket: the app offers its session key as the subprotocol ``cn-key.<key>`` next to
    ``cn.v1``, and the hub answers ``cn.v1``."""

    def session_key(self) -> str | None:
        """The session key: ``X-CN-Session-Key`` (programs), else the subprotocol list (browsers)."""
        return (self.request.headers.get(webapp.SESSION_KEY_HEADER)
                or webapp.ws_session_key(self.request.headers.get("Sec-WebSocket-Protocol")))

    def select_subprotocol(self, subprotocols: list[str]) -> str | None:
        """Answer ``cn.v1`` when offered; never echo the key."""
        return webapp.ws_select_protocol(subprotocols)
    def prepare(self) -> None:
        """Guard checks, then 401 unless a user is signed in."""
        Guard.prepare(self)
        if not self.current_user:
            raise tornado.web.HTTPError(401, "authentication required")

    def open(self) -> None:
        """Register the browser and send it the snapshot (light one with ``?lite=1``)."""
        self.lite = self.get_query_argument("lite", "0") == "1"
        self.hub.clients.add(self)
        self.hub.ensure_ticker()
        snap = self.hub.snapshot(self.current_user)
        self.write_message(json.dumps(webapp.lite(snap) if self.lite else snap, separators=(",", ":")))

    def on_message(self, message) -> None:
        """Answer ``{"type": "ping", "id"}`` with a pong (the app measures its latency); ignore
        anything else (commands go through the POST API)."""
        try:
            msg = json.loads(message)
        except ValueError:
            return
        if isinstance(msg, dict) and msg.get("type") == "ping":
            self.write_message(json.dumps({"type": "pong", "id": msg.get("id"), "wall": time.time()}))

    def on_close(self) -> None:
        """Unregister the browser."""
        self.hub.clients.discard(self)


class RelaySocket(Guard, tornado.websocket.WebSocketHandler):
    """``/relay``: the websocket of one relay, authenticated by its token."""
    relay: str | None = None

    def check_origin(self, origin: str) -> bool:
        """Accept any origin: relays are not browsers and authenticate by token."""
        return True                                     # not a browser; authenticated by token

    def prepare(self) -> None:
        """Guard checks, then the relay ``Bearer`` token (failures throttled and audited)."""
        Guard.prepare(self)
        ip = self.client_ip
        authz = self.request.headers.get("Authorization", "")
        name = self.hub.auth.relay_for_token(authz[7:].strip()) if authz.startswith("Bearer ") else None
        if name is None:                                # valid token first (see V1.prepare)
            if self.hub.throttle.blocked("relay:" + ip):
                raise tornado.web.HTTPError(429, "too many failures")
            self.hub.throttle.failure("relay:" + ip)
            self.hub.audit.write(event="relay_auth", ip=ip, result="refused")
            raise tornado.web.HTTPError(401, "bad relay token")
        self.relay = name

    def open(self) -> None:
        """Audit the relay connection."""
        self.hub.audit.write(event="relay_connect", relay=self.relay, ip=self.client_ip,
                             result="ok")

    def on_message(self, message) -> None:
        """Dispatch a relay message (hello, lines, frame, slurm, acks); logs malformed ones."""
        try:
            msg = json.loads(message)
            kind = msg.get("type")
            if kind == "hello":
                self.write_message(json.dumps(self.hub.relay_up(self.relay, self, msg)))
            elif self.hub.relays.get(self.relay) is not self:
                return                                  # no hello yet, or superseded connection
            elif kind == "lines":
                reply = self.hub.ingest_lines(self.relay, msg)
                if reply is not None:
                    self.write_message(json.dumps(reply))
            elif kind == "frame":
                self.hub.ingest_frame(self.relay, msg.get("frame"))
            elif kind == "film":
                reply = self.hub.ingest_film(self.relay, msg)
                if reply is not None:
                    self.write_message(json.dumps(reply))
            elif kind == "slurm":
                self.hub.slurm[self.relay] = msg.get("slurm") or {}
                self.hub.broadcast({"type": "slurm", "slurm": self.hub.slurm_view()})
            elif kind in ("command_ack", "job_ack"):
                fut = self.hub.pending.get(str(msg.get("id")))
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except (ValueError, KeyError, TypeError, AttributeError) as err:
            print(f"hub: malformed message from relay {self.relay!r} ignored: {err}", flush=True)

    def on_close(self) -> None:
        """Unregister the relay from the hub."""
        if self.relay:
            self.hub.relay_down(self.relay, self)


def make_app(hub: Hub) -> tornado.web.Application:
    """The tornado application: every route of the hub bound to ``hub``."""
    h = dict(hub=hub)
    return tornado.web.Application([
        (r"/healthz", Health),
        (r"/login", Login, h), (r"/login-done.js", LoginDone, h), (r"/logout", Logout, h),
        (r"/classic", Classic, h),
        (r"/api/runs", Runs, h), (r"/api/whoami", WhoAmI, h),
        (r"/api/film/([^/]+)", Film, h), (r"/api/film/([^/]+)/(frames)", Film, h),
        (r"/api/control/([^/]+)", Control, h),
        (r"/api/compare", Compare, h), (r"/api/jobs/([a-z]+)", Jobs, h),
        (r"/api/registry(/.*)?", NoRegistry, h),
        (r"/api/v1/ping", V1Ping, h), (r"/api/v1/whoami", V1WhoAmI, h),
        (r"/api/v1/runs", V1Runs, h), (r"/api/v1/board", V1Board, h),
        (r"/api/v1/audit", V1Audit, h),
        (r"/api/v1/runs/([^/]+)/commands", V1Commands, h),
        (r"/api/v1/runs/([^/]+)/(scalars|log|frame|wait)", V1RunPart, h),
        (r"/api/v1/runs/([^/]+)", V1Run, h),
        (r"/ws", BrowserSocket, h),
        (r"/relay", RelaySocket, h),
        (r"/(.*)", App, h),                             # app pages and files (last: catch-all)
    ], websocket_max_message_size=8 << 20, websocket_ping_interval=20)


# ====================================================================== CLI
STATE_DEFAULT = "hub_state"


def state_dir(given: str | None, *, local: bool) -> str:
    """The state directory of the hub (0.1.2): ``--state`` when given (``--state
    hub_state`` included: that spelling used to be replaced by a temporary directory under
    ``--local``), else ``$CN_HUB_STATE``, else ``hub_state`` for a hosted hub; ``serve --local``
    without either gets a new private (0700) directory ``conatus_hub_*`` in the system temporary
    directory. That directory is not removed when the hub stops; ``serve`` prints its path."""
    if given:
        return given
    env = os.environ.get("CN_HUB_STATE")
    if env:
        return env
    return tempfile.mkdtemp(prefix="conatus_hub_") if local else STATE_DEFAULT


def _auth_path(a: argparse.Namespace) -> Path:
    return Path(a.auth) if a.auth else Path(a.state) / "auth.json"


def _cmd_adduser(a: argparse.Namespace) -> None:
    db = AuthDB.load(_auth_path(a))
    if a.no_password:
        pw = None
    elif a.password_stdin:
        pw = sys.stdin.readline().rstrip("\n")
    else:
        pw = getpass.getpass(f"password for {a.name}: ")
        if getpass.getpass("again: ") != pw:
            sys.exit("passwords differ")
    db.add_user(a.name, pw)
    msg = f"user {a.name!r} saved in {db.path}"
    if a.token or pw is None:
        msg += f"\naccess token (shown once, keep it secret): {db.add_user_token(a.name)}"
    db.save()
    print(msg)


def _write_secret(path: str | Path, tok: str) -> Path:
    """Write a token into a 0600 file, never printed nor on a command line. Atomic and without
    following links: a fresh file is created exclusively (O_EXCL,
    O_NOFOLLOW, mode 0600 from its creation) next to the target, then renamed over it, so the token
    never lands in an existing file (another account's, or one with looser permissions) nor
    through a symbolic link planted at the path, and a reader never sees a partial token."""
    out = Path(path).expanduser()
    private_dir(out.parent)                             # 0700 when created here
    tmp = out.with_name(f".{out.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(tok + "\n")
        os.replace(tmp, out)                            # replaces a link itself, not its target
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return out


LOCAL_USER_TOKEN_FILE = "local_user_token"              # in --state, when --user-token-out is absent


def _local_user_token(a: argparse.Namespace, port: int, user_tok: str) -> str:
    """Where the one-session user token of ``serve --local`` goes, and the line to print.

    Default: a 0600 file, ``--user-token-out`` or
    ``<state>/local_user_token``; only its path is printed (a job log is readable by the group). ``--print-token`` is the explicit opt-in to print the token itself."""
    url = f"http://127.0.0.1:{port}/login"
    if getattr(a, "print_token", False) and not a.user_token_out:
        return f"open {url} and paste this one-session token (--print-token):\n  {user_tok}"
    out = _write_secret(a.user_token_out or Path(a.state) / LOCAL_USER_TOKEN_FILE, user_tok)
    msg = f"open {url} and paste the one-session token of {out} (0600)"
    if getattr(a, "print_token", False):
        msg += f"\none-session token (--print-token):\n  {user_tok}"
    return msg


def _cmd_addagent(a: argparse.Namespace) -> None:
    """An agent token (scope read | command, optional run patterns), into a 0600 file."""
    db = AuthDB.load(_auth_path(a))
    tok = db.add_agent(a.name, a.scope, a.runs or None)
    db.save()
    if a.token_out:
        out = _write_secret(a.token_out, tok)
        print(f"agent {a.name!r} ({a.scope}) saved in {db.path}; its token is in {out} (0600)")
        return
    print(f"agent {a.name!r} ({a.scope}) saved in {db.path}\nagent token (shown once; keep it in a "
          f"0600 file, e.g. ~/.config/conatus/studio_token):\n{tok}")


def _cmd_agents(a: argparse.Namespace) -> None:
    """List the agent tokens (name, scope, runs, creation date; never a digest)."""
    db = AuthDB.load(_auth_path(a))
    print(json.dumps({n: {k: v for k, v in e.items() if k != "digest"} for n, e in db.agents.items()},
                     indent=1, sort_keys=True))


def _cmd_addrelay(a: argparse.Namespace) -> None:
    db = AuthDB.load(_auth_path(a))
    tok = db.add_relay(a.name)
    db.save()
    if a.token_out:                                     # straight into a 0600 file, never printed
        out = _write_secret(a.token_out, tok)
        print(f"relay {a.name!r} saved in {db.path}; its token is in {out} (0600)")
        return
    print(f"relay {a.name!r} saved in {db.path}\nrelay token (shown once; put it in a 0600 file on "
          f"the cluster, e.g. ~/.config/conatus/relay_token):\n{tok}")


def _cmd_delete(a: argparse.Namespace) -> None:
    """Revoke a user (password and tokens) or a relay (its token)."""
    db = AuthDB.load(_auth_path(a))
    table = {"deluser": db.users, "delrelay": db.relays, "delagent": db.agents}[a.cmd]
    if table.pop(a.name, None) is None:
        sys.exit(f"no {a.cmd[3:]} named {a.name!r} in {db.path}")
    db.save()
    print(f"{a.cmd[3:]} {a.name!r} removed from {db.path}; re-export the secret and restart the hub")


def _cmd_export(a: argparse.Namespace) -> None:
    """The auth file holds only hashes: printable as a hosting secret (CN_HUB_AUTH_JSON)."""
    print(json.dumps(json.loads(AuthDB.load(_auth_path(a)).to_json()), separators=(",", ":")))


def _is_loopback(host: str) -> bool:
    """``host`` binds the loopback interface only (127.0.0.0/8, ::1, localhost)."""
    import ipaddress
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_proxy_binding(a: argparse.Namespace) -> None:
    """Refuse a hub that would trust proxy headers while reachable directly. Without ``--local`` and without TLS the hub trusts
    ``X-Forwarded-Proto`` and a client-IP header (the last ``X-Forwarded-For`` entry, or
    ``--client-ip-header``): a client reaching it directly could forge its address (the login
    throttle, the audit) and the HTTPS check. That is safe only behind the proxy. Bound to a
    non-loopback address, the operator must say so with ``--behind-proxy``
    (``CN_HUB_BEHIND_PROXY=1``), as the Docker image does; bound to 127.0.0.1 (Caddy on a VM)
    the proxy is the only way in."""
    if a.local or a.certfile or _is_loopback(a.host) or a.behind_proxy:
        return
    header = a.client_ip_header or "X-Forwarded-For (last entry)"
    sys.exit(f"refused: `serve --host {a.host}` without --local or TLS trusts the client-IP header "
             f"{header} and X-Forwarded-Proto, which any client that reaches {a.host}:{a.port} "
             "directly can forge (login throttle, audit, the HTTPS check). Only behind a proxy "
             "that overwrites them is this safe. Either bind the hub to 127.0.0.1 behind the "
             "proxy (--host 127.0.0.1), serve TLS directly (--certfile/--keyfile), or declare "
             "that only the proxy can reach this address: --behind-proxy (or "
             "CN_HUB_BEHIND_PROXY=1; the Docker image of studio/hub sets it). See "
             "studio/hub/DEPLOY.md")


def check_local_serve(a: argparse.Namespace) -> None:
    """Refuse a ``serve --local`` that would not do what its line says (Studio public-readiness
    review). ``--local`` always binds 127.0.0.1 and serves plain HTTP: a
    non-loopback ``--host`` was silently ignored, so the user believed the hub was reachable from
    elsewhere (or feared it was); it is now refused with the way to reach it (an ssh tunnel).
    ``--allow-submit`` needs ``slurm/submit.sh`` of a development checkout: without it, the
    switch would only have produced refusals at the first job, so it is refused at start."""
    if not a.local:
        return
    if not _is_loopback(a.host):
        sys.exit(f"refused: `serve --local --host {a.host}`: the local Studio binds 127.0.0.1 only "
                 "(plain HTTP, one-session token). Reach it from another machine through an ssh "
                 "tunnel: ssh -L 8765:localhost:8765 <this machine>")
    if a.allow_submit:
        script = Path(__file__).resolve().parents[3] / "slurm" / "submit.sh"
        if not script.is_file():
            sys.exit(f"refused: --allow-submit needs {script} (a development checkout with its "
                     "Slurm templates); an installed package cannot submit jobs from the Studio")


async def _serve(a: argparse.Namespace) -> None:
    from .relay import Relay  # local mode only

    state = Path(a.state)
    if a.local:
        db = AuthDB.load(_auth_path(a)) if _auth_path(a).exists() else AuthDB({})
        host = "127.0.0.1"
        user_tok = None
        if not db.users:                                # throw-away user for this session only
            db.add_user("local", None)
            user_tok = db.add_user_token("local")
        relay_tok = db.add_relay("local")
        if a.agent_token_out:                           # this session only (memory), into a file
            out = _write_secret(a.agent_token_out, db.add_agent("local-agent", a.agent_scope))
            print(f"agent token ({a.agent_scope}, this hub session only) written to {out}", flush=True)
    else:
        db = AuthDB.load(_auth_path(a))
        host = a.host
        if not db.users or not db.relays:
            sys.exit(f"no user or no relay in {_auth_path(a)} (nor $CN_HUB_AUTH_JSON): run "
                     "`python -m conatus.studio.hub adduser NAME` and `addrelay NAME` first "
                     "(see studio/hub/DEPLOY.md)")
    audit = AuditLog(state / "audit.jsonl")
    hub = Hub(db, audit=audit, local=a.local, web=Path(a.web),
              app=Path(a.app) if a.app else None, film_mb=a.film_mb)
    ssl = None
    if a.certfile:
        import ssl as _ssl
        ssl = _ssl.create_default_context(_ssl.Purpose.CLIENT_AUTH)
        ssl.load_cert_chain(a.certfile, a.keyfile)
    hub.proxied = not a.local and ssl is None           # same condition as xheaders below
    hub.client_ip_header = a.client_ip_header or None
    # behind a TLS-terminating proxy, the scheme comes from X-Forwarded-Proto (xheaders)
    server = tornado.httpserver.HTTPServer(make_app(hub), ssl_options=ssl,
                                           xheaders=not a.local and ssl is None,
                                           max_body_size=MAX_BODY_BYTES)
    server.listen(a.port, address=host)
    scheme = "https" if ssl else "http"
    print(f"Conatus Studio hub on {scheme}://{host}:{a.port} ({'LOCAL mode' if a.local else 'hub mode'};"
          f" audit {audit.path})", flush=True)
    print(f"state directory: {state} (audit, local token; kept after the hub stops)", flush=True)
    relay = None
    if a.local:
        spool = Path(a.spool) if a.spool else default_spool()
        from .launch import Launcher
        relay = Relay(f"ws://127.0.0.1:{a.port}", relay_tok, spool, poll_s=0.5, slurm_s=a.slurm,
                      launcher=Launcher(Path(__file__).resolve().parents[3],
                                        allow_submit=a.allow_submit))
        asyncio.ensure_future(relay.run())
        if user_tok:                                    # a file by default
            print(_local_user_token(a, a.port, user_tok), flush=True)
    await asyncio.Event().wait()


def main(argv: list[str] | None = None) -> None:
    """CLI: ``serve`` the hub, or manage the auth file (adduser, addrelay, deluser, delrelay, export)."""
    from conatus._experimental import experimental
    experimental("hub")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=None,
                    help="directory for auth.json, audit.jsonl and the --local token file (a "
                         "persistent volume when hosted). Default: $CN_HUB_STATE, else "
                         f"{STATE_DEFAULT} for the hub; `serve --local` without either uses a new "
                         "private temporary directory and prints its path")
    ap.add_argument("--auth", default=None, help="auth file (default <state>/auth.json; else $CN_HUB_AUTH_JSON)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the hub")
    # 127.0.0.1 by default: the former 0.0.0.0 was refused
    # by check_proxy_binding unless --behind-proxy or TLS, a default refused by default. The
    # Docker image passes its own `--host 0.0.0.0 --behind-proxy`; TLS served directly needs
    # `--host 0.0.0.0` (or the public address) with --certfile/--keyfile.
    s.add_argument("--host", default="127.0.0.1",
                   help="address to bind (default 127.0.0.1: behind a proxy on the same machine; "
                        "another address needs TLS (--certfile/--keyfile) or --behind-proxy)")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    s.add_argument("--local", action="store_true",
                   help="hub + relay on this machine, 127.0.0.1 only, plain HTTP allowed (tests)")
    s.add_argument("--spool", default=None, help="--local: spool to relay")
    s.add_argument("--slurm", type=float, default=0.0, help="--local: squeue period for the relay (0 = off)")
    s.add_argument("--allow-submit", action="store_true",
                   help="--local: let the in-process relay submit jobs (S2, via slurm/submit.sh)")
    s.add_argument("--web", default=os.environ.get("CN_STUDIO_WEB", str(WEB_DEFAULT)),
                   help="the classic page (served at /classic)")
    s.add_argument("--app", default=None,
                   help="built Studio app (default $CN_STUDIO_APP or scratch/studio_app/dist)")
    s.add_argument("--film-mb", type=float, default=512.0,
                   help="memory budget of all the films held by the hub (MiB; thinned beyond)")
    s.add_argument("--user-token-out", default=None,
                   help="--local: file (0600) for the one-session user token (default "
                        f"<state>/{LOCAL_USER_TOKEN_FILE}); only its path is printed (a job log is "
                        "readable by the group)")
    s.add_argument("--print-token", action="store_true",
                   help="--local: also print the one-session user token itself (explicit opt-in; "
                        "it then lands in any log of this process)")
    s.add_argument("--agent-token-out", default=None,
                   help="--local: also make an agent token for this session, written to this file (0600)")
    s.add_argument("--agent-scope", default="command", choices=("read", "command"),
                   help="--local: scope of that agent token")
    s.add_argument("--certfile", default=None, help="serve TLS directly (else a proxy terminates it)")
    s.add_argument("--keyfile", default=None)
    s.add_argument("--client-ip-header", default=os.environ.get("CN_HUB_CLIENT_IP_HEADER") or None,
                   help="behind a proxy: header holding the client address that the proxy "
                        "overwrites (e.g. Fly-Client-IP on Fly.io); default: the last "
                        "X-Forwarded-For entry (Caddy)")
    s.add_argument("--behind-proxy", action="store_true",
                   default=os.environ.get("CN_HUB_BEHIND_PROXY", "") not in ("", "0"),
                   help="declare that only a TLS-terminating proxy reaches this address (Fly.io, "
                        "Render, a container published on 127.0.0.1): required to bind a "
                        "non-loopback address without TLS, because the hub then trusts the "
                        "proxy's client-IP and X-Forwarded-Proto headers ($CN_HUB_BEHIND_PROXY=1)")
    u = sub.add_parser("adduser", help="add a user (password prompt) or reset their password")
    u.add_argument("name")
    u.add_argument("--token", action="store_true", help="also print a new API/access token")
    u.add_argument("--password-stdin", action="store_true")
    u.add_argument("--no-password", action="store_true", help="token-only user")
    r = sub.add_parser("addrelay", help="add a relay and print its token once")
    r.add_argument("name")
    r.add_argument("--token-out", default=None, help="write the token to this file (0600) instead of printing it")
    g = sub.add_parser("addagent", help="add an agent token (scope read | command), shown once")
    g.add_argument("name")
    g.add_argument("--scope", required=True, choices=("read", "command"),
                   help="read: state only; command: also the declared run commands")
    g.add_argument("--runs", nargs="*", default=None,
                   help="limit to these run keys (fnmatch patterns, e.g. 'mycluster:kh_*')")
    g.add_argument("--token-out", default=None, help="write the token to this file (0600)")
    sub.add_parser("agents", help="list the agent tokens (no digests)")
    for what in ("user", "relay", "agent"):
        d = sub.add_parser(f"del{what}", help=f"revoke a {what}")
        d.add_argument("name")
    sub.add_parser("export", help="print the auth file (hashes only) for $CN_HUB_AUTH_JSON")
    a = ap.parse_args(argv)
    if a.cmd != "serve":
        a.state = state_dir(a.state, local=False)
    if a.cmd == "adduser":
        _cmd_adduser(a)
    elif a.cmd == "addrelay":
        _cmd_addrelay(a)
    elif a.cmd == "addagent":
        _cmd_addagent(a)
    elif a.cmd == "agents":
        _cmd_agents(a)
    elif a.cmd in ("deluser", "delrelay", "delagent"):
        _cmd_delete(a)
    elif a.cmd == "export":
        _cmd_export(a)
    else:
        check_proxy_binding(a)
        check_local_serve(a)
        if not a.local:
            sys.exit("the hosted hub (serve without --local) is not part of this release: "
                     "use `serve --local` (see the Studio guide)")
        a.state = state_dir(a.state, local=a.local)
        try:
            asyncio.run(_serve(a))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
