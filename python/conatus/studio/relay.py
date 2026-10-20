"""Outbound Studio relay: telemetry spool -> hub, hub commands -> ``<run_id>.control.jsonl``.

    CN_RELAY_TOKEN=... python -m conatus.studio.relay --hub wss://studio.example.org [--spool DIR]
    python -m conatus.studio.relay --hub wss://... --token-file ~/.config/conatus/relay_token

Runs inside a job on a compute node (never on a cluster head/login node): it only needs *outbound* HTTPS to the hub, so the
cluster opens no port. It reads the spool the computation already writes (protocol v1) and never
talks to the computation otherwise; a relay that is slow, absent or crashed has no effect on a run
(the computation only ever sees files).

Wire protocol (JSON over one WebSocket, relay-initiated, bearer token in the handshake):

relay -> hub  ``hello``   {protocol, host, spool}
hub -> relay  ``welcome`` {relay, cursors: {run_id: byte offset already ingested},
              film_cursors: {run_id: byte offset of the film already ingested}}
relay -> hub  ``lines``   {run_id, start, end, data, reset?}  complete lines of ``<run_id>.jsonl``,
              bytes [start, end); the hub ingests only what lies beyond its cursor (no duplicates)
relay -> hub  ``frame``   {frame}                              latest frame (idempotent)
relay -> hub  ``film``    {run_id, start, end, data, reset?}  complete lines of ``<run_id>.film.ndjson``
              (the film, Studio rebuild), same cursor rules as ``lines``
relay -> hub  ``slurm``   {slurm}                              squeue snapshot, at most every
              max(slurm_s, 120 s), through slurm/slurm_gate.sh in a checkout
hub -> relay  ``resync``  {run_id, cursor, film?}              gap detected: resend from cursor
              (``film: true``: in the film file)
hub -> relay  ``command`` {id, run_id, cmd, params, user}      appended to the control file
relay -> hub  ``command_ack`` {id, ok, error?, line?}
hub -> relay  ``job``     {id, op, args, user}   S2 job operation (catalog, prepare, submit, list,
              log; :mod:`conatus.studio.launch`), re-validated here with the file checks
relay -> hub  ``job_ack`` {id, ok, result? | error?, code?}

Resumption without loss or duplicates: the hub is the source of truth for the per-file cursors; after
any reconnection the relay restarts each file exactly where the hub says (a restarted hub has no
cursor, so it gets the whole history again, once). Throughput is bounded by a token bucket
(``max_rate`` bytes/s) and reconnection uses exponential backoff with jitter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import tornado.httpclient
import tornado.websocket
from tornado.iostream import StreamClosedError

from conatus.telemetry import COMMANDS, PROTOCOL_VERSION, check_set, default_spool
from conatus.telemetry.protocol import RUN_ID_RE, append_control, run_file_id

from .launch import JOB_OPS, Launcher, LaunchError

REPO_ROOT = Path(__file__).resolve().parents[3]      # python/conatus/studio -> repository
# Slurm etiquette of the project (Execution): >= 120 s between two Slurm client calls,
# never two at once, every call through slurm/slurm_gate.sh (one lock for every session and agent).
# The Studio's squeue panel (relay and local server) used to poll every 30-60 s (local server: down
# to 10 s) outside the gate.
SLURM_MIN_PERIOD_S = 120.0
_SQUEUE_FIELDS = ("id", "name", "state", "elapsed", "left", "partition", "nodes", "reason")


def squeue_command() -> list[str]:
    """The squeue of the user's jobs, prefixed by the repository's ``slurm/slurm_gate.sh`` when this
    package runs from a checkout that has it (an installed package has no gate: plain squeue)."""
    cmd = ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%j|%T|%M|%L|%P|%N|%r"]
    gate = REPO_ROOT / "slurm" / "slurm_gate.sh"
    return [str(gate), *cmd] if gate.is_file() else cmd


def squeue_snapshot() -> dict:
    """``{when, jobs, error}`` from one (gated) squeue; blocking: run it off the event loop. The
    gate may wait for its turn (>= 120 s after the previous Slurm call of any session), hence the
    long timeout; its own "waiting" note on stderr is not an error."""
    cmd = squeue_command()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=20 if cmd[0] == "squeue" else 900)
    except (OSError, subprocess.SubprocessError) as err:     # no Slurm here: shown, not fatal
        return {"when": time.time(), "jobs": [], "error": f"{type(err).__name__}: {err}"}
    jobs = [dict(zip(_SQUEUE_FIELDS, line.split("|"), strict=False))
            for line in out.stdout.splitlines() if line.strip()]
    err = "\n".join(x for x in out.stderr.splitlines() if not x.startswith("[slurm_gate] waiting"))
    return {"when": time.time(), "jobs": jobs, "error": err.strip() or None}
MAX_PARAMS_BYTES = 4096              # fork parameters: small JSON objects only
FILM_BYTES_PER_PUSH = 2 << 20        # film history sent per poll at most (live data goes first)
MAX_LINE_BYTES = 4 << 20             # a single telemetry line longer than this is skipped and logged
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
# Dead-link detection. tornado silently clamps a ping timeout longer than the interval.
PING_INTERVAL_S, PING_TIMEOUT_S = 20.0, 20.0


class TokenBucket:
    """Byte-rate limiter: ``delay(n)`` returns how long to wait before sending n bytes."""

    def __init__(self, rate: float, burst: float, clock=time.monotonic) -> None:
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate, self.burst, self.clock = float(rate), float(burst), clock
        self.tokens, self.t = float(burst), clock()

    def delay(self, n: int) -> float:
        """Refill, spend ``n`` bytes and return the wait (s) needed to repay any debt."""
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        self.tokens -= n                    # may go negative: the debt is the wait
        return 0.0 if self.tokens >= 0 else -self.tokens / self.rate


class Relay:
    """One relay: one spool, one hub. All knobs explicit (see :meth:`explain`)."""

    def __init__(self, hub_url: str, token: str, spool: str | Path, *, poll_s: float = 1.0,
                 max_rate: float = 256e3, max_batch: int = 128 * 1024, backoff_s: float = 1.0,
                 max_backoff_s: float = 60.0, slurm_s: float = SLURM_MIN_PERIOD_S, connect_timeout_s: float = 20.0,
                 log=None, launcher: Launcher | None = None) -> None:
        u = urlparse(hub_url)
        if u.scheme not in ("ws", "wss"):
            raise ValueError(f"hub URL must be ws:// or wss://, got {hub_url!r}")
        if u.scheme == "ws" and u.hostname not in LOOPBACK:
            # the relay token would travel in clear: only allowed for the --local test mode
            raise ValueError(f"refusing plain ws:// to non-loopback host {u.hostname!r}; use wss://")
        if not token:
            raise ValueError("empty relay token")
        self.url = hub_url.rstrip("/") + ("" if u.path.endswith("/relay") else "/relay")
        self.token = token
        self.spool = Path(spool)
        self.poll_s, self.max_batch = float(poll_s), int(max_batch)
        self.bucket = TokenBucket(max_rate, max(max_rate, max_batch))
        self.max_rate = float(max_rate)
        self.backoff_s, self.max_backoff_s = float(backoff_s), float(max_backoff_s)
        self.slurm_s, self.connect_timeout_s = float(slurm_s), float(connect_timeout_s)
        self.cursors: dict[str, int] = {}
        self.film_cursors: dict[str, int] = {}
        self._frame_mtime: dict[str, float] = {}
        self._seen_cmd: deque = deque(maxlen=4096)
        self._seq = int(time.time() * 1000)
        self._stop = asyncio.Event() if _has_loop() else None
        self._conn = None
        self.connected = False
        self.name: str | None = None
        # S2 job launcher (runs slurm/submit.sh here); None = every job operation is refused
        self.launcher = launcher
        self._last_slurm: dict = {}
        self.stats = dict(connects=0, failures=0, sent_bytes=0, sent_lines=0, frames=0, film=0, commands=0,
                          refused_commands=0, jobs=0, refused_jobs=0, last_error=None)
        self._log = log or (lambda msg: print(f"[relay {time.strftime('%H:%M:%S')}] {msg}", flush=True))

    def explain(self) -> str:
        """One-line summary of every relay knob and its statistics."""
        return (f"Relay(spool={self.spool} -> {self.url}, poll every {self.poll_s:g} s, <= "
                f"{self.max_rate:g} B/s in messages <= {self.max_batch} B, backoff {self.backoff_s:g}"
                f"..{self.max_backoff_s:g} s, squeue every "
                f"{max(self.slurm_s, SLURM_MIN_PERIOD_S) if self.slurm_s > 0 else 'never (off)'} s; jobs: "
                f"{self.launcher.explain() if self.launcher else 'no launcher (job operations refused)'}"
                f"; stats {self.stats})")

    # ------------------------------------------------------------------ lifecycle
    async def run(self) -> None:
        """Connect, stream, reconnect with backoff until :meth:`stop`."""
        if self._stop is None:
            self._stop = asyncio.Event()
        delay = self.backoff_s
        while not self._stop.is_set():
            try:
                await self._session()
                delay = self.backoff_s                      # a session happened: reset backoff
            except (TimeoutError, OSError, StreamClosedError, tornado.websocket.WebSocketClosedError, tornado.httpclient.HTTPClientError, ValueError) as err:
                self.stats["failures"] += 1
                self.stats["last_error"] = f"{type(err).__name__}: {err}"
                if isinstance(err, tornado.httpclient.HTTPClientError) and err.code in (401, 403):
                    self._log(f"hub refused the relay token ({err.code}); retrying in {delay:.1f} s")
                else:
                    self._log(f"hub unreachable ({self.stats['last_error']}); retry in {delay:.1f} s")
            self.connected = False
            if self._stop.is_set():
                break
            try:                                            # jittered exponential backoff
                await asyncio.wait_for(self._stop.wait(), delay * (0.8 + 0.4 * random.random()))
            except TimeoutError:
                pass
            delay = min(self.max_backoff_s, 2 * delay)

    def stop(self) -> None:
        """Ask :meth:`run` to finish and close the current connection."""
        if self._stop is not None:
            self._stop.set()
        if self._conn is not None:
            self._conn.close()

    async def _session(self) -> None:
        req = tornado.httpclient.HTTPRequest(self.url, headers={"Authorization": f"Bearer {self.token}"},
                                             connect_timeout=self.connect_timeout_s,
                                             request_timeout=self.connect_timeout_s)
        conn = await tornado.websocket.websocket_connect(req, ping_interval=PING_INTERVAL_S,
                                                         ping_timeout=PING_TIMEOUT_S,
                                                         max_message_size=1 << 20)
        self._conn = conn
        try:
            await conn.write_message(json.dumps({"type": "hello", "protocol": PROTOCOL_VERSION,
                                                 "host": socket.gethostname(), "spool": str(self.spool)}))
            msg = await asyncio.wait_for(conn.read_message(), self.connect_timeout_s)
            welcome = json.loads(msg) if msg is not None else {}
            if welcome.get("type") != "welcome":
                raise ValueError(f"expected a welcome from the hub, got {str(msg)[:120]!r}")
            self.name = welcome.get("relay")
            self.cursors = {k: int(v) for k, v in welcome.get("cursors", {}).items()}
            # an older hub sends no film cursors: it ignores film messages, which then cost only
            # bandwidth; a hub that knows films resumes them exactly like the record files
            self.film_cursors = {k: int(v) for k, v in welcome.get("film_cursors", {}).items()}
            self._frame_mtime.clear()                        # frames: resend the latest of each
            self.connected = True
            self.stats["connects"] += 1
            self._log(f"connected to {self.url} as {self.name!r}; resuming {len(self.cursors)} file(s)")
            reader = asyncio.ensure_future(self._read_loop(conn))
            slurm_next, slurm_task = 0.0, None
            try:
                while not self._stop.is_set() and not reader.done():
                    await self._push(conn)
                    if (self.slurm_s > 0 and time.monotonic() >= slurm_next
                            and (slurm_task is None or slurm_task.done())):
                        slurm_next = time.monotonic() + max(self.slurm_s, SLURM_MIN_PERIOD_S)
                        slurm_task = asyncio.ensure_future(self._slurm_push(conn))
                    try:
                        await asyncio.wait_for(asyncio.shield(reader), self.poll_s)
                    except TimeoutError:
                        pass
                if reader.done() and reader.exception() is not None:
                    raise reader.exception()
            finally:
                reader.cancel()
                if slurm_task is not None:
                    slurm_task.cancel()
        finally:
            self._conn = None
            conn.close()

    # ------------------------------------------------------------------ spool -> hub
    async def _send(self, conn, msg: dict) -> None:
        text = json.dumps(msg, separators=(",", ":"))
        wait = self.bucket.delay(len(text))
        if wait > 0:
            await asyncio.sleep(wait)
        await conn.write_message(text)
        self.stats["sent_bytes"] += len(text)

    async def _push(self, conn) -> None:
        """Send every complete new line of every run file, then changed frames, then at most
        FILM_BYTES_PER_PUSH of film history: a film backlog (a hub restart forgets the film
        cursors; a film may hold 256 MiB) is caught up over several polls and never holds back
        the live lines and frames of any run (review)."""
        if not self.spool.is_dir():
            return
        for path in sorted(self.spool.glob("*.jsonl")):
            run_id = run_file_id(path.name)          # not a control file, not the spool audit
            if run_id is None:
                continue
            while True:                                      # this file, until caught up
                msg = self._read_lines(path, run_id)
                if msg is None:
                    break
                await self._send(conn, msg)
                self.cursors[run_id] = msg["end"]
                self.stats["sent_lines"] += msg["data"].count("\n")
        for path in sorted(self.spool.glob("*.frame.*.json")):
            try:
                mtime = path.stat().st_mtime
                if self._frame_mtime.get(path.name) == mtime:
                    continue
                frame = json.loads(path.read_text())
            except (OSError, ValueError):
                continue                                     # being replaced: next poll
            self._frame_mtime[path.name] = mtime
            await self._send(conn, {"type": "frame", "frame": frame})
            self.stats["frames"] += 1
        budget = FILM_BYTES_PER_PUSH
        for path in sorted(self.spool.glob("*.film.ndjson")):        # films
            run_id = path.name[:-len(".film.ndjson")]
            if not RUN_ID_RE.match(run_id):
                continue
            while budget > 0:
                msg = self._read_lines(path, run_id, self.film_cursors, "film")
                if msg is None:
                    break
                await self._send(conn, msg)
                self.film_cursors[run_id] = msg["end"]
                self.stats["film"] += msg["data"].count("\n")
                budget -= msg["end"] - msg["start"]

    def _read_lines(self, path: Path, run_id: str, cursors: dict[str, int] | None = None,
                    kind: str = "lines") -> dict | None:
        """The next message of complete lines of ``path`` from its cursor (None: nothing new)."""
        cursors = self.cursors if cursors is None else cursors
        start = cursors.get(run_id, 0)
        try:
            size = path.stat().st_size
            reset = size < start                             # file truncated / replaced
            if reset:
                start = 0
            if size == start:
                return None
            with path.open("rb") as fh:
                fh.seek(start)
                chunk = fh.read(min(size - start, self.max_batch))
                if b"\n" not in chunk and size - start > len(chunk):
                    chunk += fh.read(MAX_LINE_BYTES)         # one long line: read it whole
        except OSError:
            return None
        cut = chunk.rfind(b"\n") + 1
        if cut == 0:
            if len(chunk) >= MAX_LINE_BYTES:                 # pathological line: skip it, loudly
                self._log(f"{run_id}: line longer than {MAX_LINE_BYTES} B skipped at byte {start}")
                cursors[run_id] = start + len(chunk)
            return None                                      # incomplete line: wait
        chunk = chunk[:cut]
        msg = {"type": kind, "run_id": run_id, "start": start, "end": start + len(chunk),
               "data": chunk.decode("utf-8", "replace")}
        if reset:
            msg["reset"] = True
        # "replace" may change the byte length of invalid UTF-8: the cursors stay in source bytes
        # (``end``); the hub does not re-check end - start against the decoded data, so a partial
        # overlap (start < its cursor) after such a replacement may cut a line, then counted as
        # malformed there
        return msg

    async def _squeue(self) -> dict:
        return await asyncio.get_running_loop().run_in_executor(None, squeue_snapshot)

    async def _slurm_push(self, conn) -> None:
        """One squeue snapshot sent to the hub, in its own task: a gated squeue may wait minutes
        for its turn, and the telemetry lines must keep flowing meanwhile."""
        self._last_slurm = await self._squeue()
        try:
            await self._send(conn, {"type": "slurm", "slurm": self._last_slurm})
        except StreamClosedError:                    # reconnecting: the next connection sends it
            pass

    # ------------------------------------------------------------------ hub -> spool
    async def _read_loop(self, conn) -> None:
        while True:
            raw = await conn.read_message()
            if raw is None:
                raise StreamClosedError()                    # hub closed: reconnect
            try:
                msg = json.loads(raw)
            except ValueError:
                self._log(f"unreadable message from hub ignored: {str(raw)[:80]!r}")
                continue
            kind = msg.get("type")
            if kind == "command":
                ack = self.apply_command(msg)
                await conn.write_message(json.dumps(ack))
            elif kind == "job":                         # own task: commands keep flowing meanwhile
                asyncio.ensure_future(self._job_reply(conn, msg))
            elif kind == "resync":
                cur = self.film_cursors if msg.get("film") else self.cursors
                cur[str(msg.get("run_id"))] = int(msg.get("cursor", 0))
            else:
                self._log(f"unknown message type from hub ignored: {kind!r}")

    def apply_command(self, msg: dict) -> dict:
        """Validate a hub command and append it to ``<run_id>.control.jsonl`` (protocol v1)."""
        cid = str(msg.get("id", ""))
        ack = {"type": "command_ack", "id": cid, "ok": False}
        cmd, run_id, params = msg.get("cmd"), str(msg.get("run_id", "")), msg.get("params")
        if cid and cid in self._seen_cmd:
            return {**ack, "ok": True, "duplicate": True}
        err = None
        if cmd not in COMMANDS:
            err = f"unknown command {cmd!r}; known: {COMMANDS}"
        elif run_file_id(f"{run_id}.jsonl") is None or not (self.spool / f"{run_id}.jsonl").is_file():
            err = f"unknown run {run_id!r}"
        elif params is not None and (not isinstance(params, dict)
                                     or len(json.dumps(params)) > MAX_PARAMS_BYTES):
            err = f"params must be a JSON object of at most {MAX_PARAMS_BYTES} bytes"
        elif cmd == "set":                                   # against the run's own declaration
            try:
                check_set((read_header(self.spool / f"{run_id}.jsonl") or {}).get("controls"), params)
            except ValueError as e:
                err = str(e)
        elif cmd != "fork" and params:                       # same rule as the hub (check_command)
            err = f"{cmd} takes no parameters"
        if err is not None:                                  # the relay re-checks: defence in depth
            self.stats["refused_commands"] += 1
            self._log(f"command refused: {err}")
            return {**ack, "error": err}
        self._seq = max(self._seq + 1, int(time.time() * 1000))
        line = {"cmd": cmd, "seq": self._seq, "wall": time.time(),
                "from": f"hub:{msg.get('user', '?')}", "id": cid}
        if params is not None:
            line["params"] = params
        try:
            append_control(self.spool / f"{run_id}.control.jsonl", line)     # 0600
        except OSError as e:
            return {**ack, "error": f"control file not writable: {e}"}
        self._seen_cmd.append(cid)
        self.stats["commands"] += 1
        self._log(f"command {cmd} -> {run_id} (from {line['from']})")
        return {**ack, "ok": True, "line": line}

    async def _job_reply(self, conn, msg: dict) -> None:
        ack = await self.apply_job(msg)
        try:
            await conn.write_message(json.dumps(ack))
        except (StreamClosedError, tornado.websocket.WebSocketClosedError):
            self._log(f"job {msg.get('op')} done but the hub is gone; its ack is lost (state in audit)")

    async def apply_job(self, msg: dict) -> dict:
        """S2 job operation from the hub, fully re-validated here (template, name, bounded params,
        checkpoint inside the checkpoint directory and existing, confirmation and digest for a
        submission). Runs off the event loop: submit.sh may take a few seconds."""
        cid = str(msg.get("id", ""))
        ack = {"type": "job_ack", "id": cid, "ok": False}
        op, args, user = msg.get("op"), msg.get("args"), f"hub:{msg.get('user', '?')}"
        if self.launcher is None:
            self.stats["refused_jobs"] += 1
            return {**ack, "error": "this relay has no job launcher", "code": 403}
        if op not in JOB_OPS or not isinstance(args, dict):
            self.stats["refused_jobs"] += 1
            return {**ack, "error": f"unknown job operation {op!r}", "code": 400}
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.launcher.handle(op, args, user=user, slurm=self._last_slurm))
        except LaunchError as err:
            self.stats["refused_jobs"] += 1
            self._log(f"job {op} refused: {err}")
            return {**ack, "error": str(err), "code": err.code}
        self.stats["jobs"] += 1
        if op == "submit":
            self._log(f"job submitted by {user}: {result.get('run_id')} (job {result.get('jobid')})")
        return {**ack, "ok": True, "result": result}


def read_header(path: Path, max_bytes: int = 1 << 20) -> dict | None:
    """The header record of a run file (its first line), or None if absent or unreadable."""
    try:
        with path.open("rb") as fh:
            line = fh.readline(max_bytes)
        rec = json.loads(line)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and rec.get("kind") == "header" else None


def _has_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _read_token(a: argparse.Namespace) -> str:
    """The token never goes on the command line (visible in ps): file or environment only."""
    if a.token_file:
        p = Path(a.token_file).expanduser()
        if p.stat().st_mode & 0o077:
            print(f"warning: {p} is readable by others; chmod 600 it", file=sys.stderr)
        return p.read_text().strip()
    tok = os.environ.get("CN_RELAY_TOKEN", "").strip()
    if not tok:
        sys.exit("no relay token: pass --token-file FILE or set CN_RELAY_TOKEN")
    return tok


def main(argv: list[str] | None = None) -> None:
    """CLI: run a relay from a spool to a hub until interrupted."""
    sys.exit("the relay to a hosted hub is not part of this release: use "
             "`python -m conatus.studio.hub serve --local` (see the Studio guide)")
    from conatus._experimental import experimental
    experimental("hub")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hub", required=True, help="hub URL, wss://host (ws:// only to 127.0.0.1)")
    ap.add_argument("--token-file", default=None, help="file holding the relay token (else $CN_RELAY_TOKEN)")
    ap.add_argument("--spool", default=None, help="telemetry spool (default $CN_TELEMETRY or scratch/telemetry)")
    ap.add_argument("--poll", type=float, default=1.0, help="spool poll period (s)")
    ap.add_argument("--max-rate", type=float, default=256e3, help="upload bound (bytes/s)")
    ap.add_argument("--slurm", type=float, default=SLURM_MIN_PERIOD_S,
                    help="squeue period (s, >= 120: the project's Slurm etiquette; 0 = off)")
    ap.add_argument("--max-backoff", type=float, default=60.0, help="longest wait between reconnections (s)")
    ap.add_argument("--allow-submit", action="store_true",
                    help="allow job submission from the hub (S2; only catalogue templates, only via "
                         "slurm/submit.sh, after an explicit confirmation of the exact script)")
    a = ap.parse_args(argv)
    try:
        os.nice(10)                                          # background priority next to the computation
    except OSError:
        pass
    relay = Relay(a.hub, _read_token(a), Path(a.spool) if a.spool else default_spool(), poll_s=a.poll,
                  max_rate=a.max_rate, slurm_s=a.slurm, max_backoff_s=a.max_backoff,
                  launcher=Launcher(REPO_ROOT, allow_submit=a.allow_submit))
    print(relay.explain(), flush=True)
    try:
        asyncio.run(relay.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
