"""``python -m conatus studio ...``: follow and pilot a run from a terminal, a script or an agent.

    python -m conatus studio runs                       # every run, live first (JSON)
    python -m conatus studio status kh_demo             # one run: state, step, t, dt, scalars, alerts
    python -m conatus studio watch kh_demo              # one JSON line per change, until it ends
    python -m conatus studio send kh_demo pause         # declared commands only
    python -m conatus studio send kh_demo set dt=2e-3   # a parameter the run declared settable
    python -m conatus studio send kh_demo stop          # -> exit 5 + confirm_token; then:
    python -m conatus studio send kh_demo stop --confirm <token>
    python -m conatus studio frame kh_demo --png kh.png # the latest picture of the default view
    python -m conatus studio log kh_demo -n 50
    python -m conatus studio board                      # runs, who pilots them, recent commands

Where it talks: ``--hub URL`` (or ``$CN_STUDIO_URL``, or ``~/.config/conatus/studio.json``) with a
token from ``--token-file`` / ``$CN_STUDIO_TOKEN`` / ``~/.config/conatus/studio_token``; otherwise
``--spool DIR`` (default ``$CN_TELEMETRY`` or ``scratch/telemetry``): the shared spool, read and
written directly (a session on the cluster needs no server).

Output: JSON on stdout (API v1, keys sorted; ``watch``: one compact line per change), errors as
JSON too. Exit codes (stable): 0 ok, 1 refused (bad command, wrong state), 2 usage, 3 not
authorised (token, scope), 4 unknown run or frame, 5 confirmation needed (``confirm_token`` in the
answer), 6 hub unreachable or failing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import agentapi
from .client import StudioError, connect

EXIT = {"ok": 0, "refused": 1, "usage": 2, "auth": 3, "missing": 4, "confirm": 5, "down": 6}


def exit_code(ans: Any) -> int:
    """The process exit code of an API answer (see the module docstring)."""
    if not isinstance(ans, dict) or ans.get("ok"):
        return EXIT["ok"]
    code = int(ans.get("code") or 0)
    if ans.get("needs_confirmation"):
        return EXIT["confirm"]
    if code in (401, 403, 429):
        return EXIT["auth"]
    if code == 404:
        return EXIT["missing"]
    if code >= 500 or code == 0:
        return EXIT["down"]
    return EXIT["refused"]


def _value(text: str) -> Any:
    """A command-line value: JSON if it parses (numbers, true, objects), else the string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def command_params(cmd: str, args: list[str], params_json: str | None) -> dict | None:
    """``set dt=1e-3`` -> ``{"name": "dt", "value": 0.001}``; ``fork Ra=2e6`` -> ``{"Ra": 2e6}``;
    ``--params '{...}'`` taken as is. ValueError on a malformed argument."""
    if params_json is not None:
        p = json.loads(params_json)
        if not isinstance(p, dict):
            raise ValueError("--params must be a JSON object")
        return p
    if not args:
        return None
    kv = {}
    for a in args:
        if "=" not in a:
            raise ValueError(f"expected name=value, got {a!r}")
        k, v = a.split("=", 1)
        kv[k.strip()] = _value(v.strip())
    if cmd == "set":
        if len(kv) != 1:
            raise ValueError("set takes one name=value (e.g. set dt=2e-3)")
        (k, v), = kv.items()
        return {"name": k, "value": v}
    return kv


def _print(obj: Any, compact: bool = False) -> None:
    text = (json.dumps(obj, sort_keys=True, separators=(",", ":")) if compact
            else json.dumps(obj, sort_keys=True, indent=1))
    try:
        print(text, flush=True)
    except BrokenPipeError:                              # the reader went away (| head): stop quietly
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0) from None


def main(argv: list[str] | None = None) -> int:
    """Parse, talk to the hub or the spool, print JSON; returns the exit code."""
    p = argparse.ArgumentParser(prog="python -m conatus studio",
                                description="Follow and pilot runs of the Conatus Studio (JSON out).",
                                epilog="exit codes: 0 ok, 1 refused, 2 usage, 3 not authorised, "
                                       "4 unknown run, 5 confirmation needed, 6 hub unreachable")
    p.add_argument("--hub", default=None, help="hub URL (https://..., or http://localhost:PORT via ssh)")
    p.add_argument("--spool", default=None, help="talk to this spool directly (no hub)")
    p.add_argument("--token-file", default=None, help="file holding the token (0600)")
    p.add_argument("--as", dest="who", default=None,
                   help="spool transport: the sender name in control lines and the audit")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (s)")
    p.add_argument("--compact", action="store_true", help="one JSON line per answer")
    sub = p.add_subparsers(dest="op", required=True)
    sub.add_parser("ping", help="reachability and latency of the hub (or the spool)")
    sub.add_parser("whoami", help="who the hub thinks you are (name, scope)")
    sub.add_parser("runs", help="every run, live first, brief")
    sub.add_parser("board", help="runs, who pilots them, recent commands")
    s = sub.add_parser("status", help="one run in full")
    s.add_argument("run")
    s.add_argument("--log", type=int, default=20)
    s.add_argument("--events", type=int, default=20)
    w = sub.add_parser("watch", help="one JSON line per change of a run, until it ends")
    w.add_argument("run")
    w.add_argument("--count", type=int, default=0, help="stop after this many lines (0: until it ends)")
    w.add_argument("--timeout", dest="wait_timeout", type=float, default=25.0,
                   help="long-poll period (s)")
    w.add_argument("--for", dest="duration", type=float, default=0.0,
                   help="stop after this many seconds (0: no limit)")
    c = sub.add_parser("send", help="send a declared command: pause | resume | checkpoint | stop | "
                                    "fork [k=v ...] | set name=value")
    c.add_argument("run")
    c.add_argument("cmd", choices=list(agentapi.COMMANDS))
    c.add_argument("args", nargs="*", help="set: name=value; fork: k=v parameters")
    c.add_argument("--params", default=None, help="parameters as a JSON object")
    c.add_argument("--confirm", default=None, help="the confirm_token of the first answer (stop, fork)")
    f = sub.add_parser("frame", help="latest frame of a view (JSON, or --png FILE)")
    f.add_argument("run")
    f.add_argument("--name", default=None, help="the view (default: the first declared one)")
    f.add_argument("--png", default=None, help="write a PNG in the brand scale to this file")
    lg = sub.add_parser("log", help="log records and events of a run")
    lg.add_argument("run")
    lg.add_argument("-n", type=int, default=100)
    sc = sub.add_parser("scalars", help="scalar records of a run")
    sc.add_argument("run")
    sc.add_argument("--since", type=float, default=0.0, help="wall time (s since 1970)")
    sc.add_argument("--names", default="", help="comma-separated scalar names")
    sc.add_argument("--max", type=int, default=2000)
    argv = list(sys.argv[1:] if argv is None else argv)
    compact = "--compact" in argv                        # accepted anywhere on the line
    a = p.parse_args([x for x in argv if x != "--compact"])
    a.compact = a.compact or compact
    try:
        cl = connect(hub=a.hub, spool=a.spool, token_file=a.token_file, who=a.who, timeout=a.timeout)
        ans = _dispatch(cl, a)
    except StudioError as err:                           # no answer from the hub
        ans = {"api": agentapi.API_VERSION, "ok": False, "error": str(err), "code": err.code}
    except ValueError as err:                            # bad URL, plain http, bad name=value
        _print({"api": agentapi.API_VERSION, "ok": False, "error": str(err), "code": 400}, a.compact)
        return EXIT["usage"]
    if ans is None:                                      # watch printed its own lines
        return EXIT["ok"]
    if isinstance(ans, int):
        return ans
    _print(ans, a.compact)
    return exit_code(ans)


def _dispatch(cl: Any, a: argparse.Namespace) -> Any:
    if a.op == "ping":
        return cl.ping()
    if a.op == "whoami":
        return cl.whoami()
    if a.op == "runs":
        return cl.runs()
    if a.op == "board":
        return cl.board()
    if a.op == "status":
        return cl.status(a.run, log=a.log, events=a.events)
    if a.op == "log":
        return cl.log(a.run, a.n)
    if a.op == "scalars":
        return cl.scalars(a.run, since=a.since, names=[x for x in a.names.split(",") if x], max=a.max)
    if a.op == "send":
        return cl.send(a.run, a.cmd, command_params(a.cmd, a.args, a.params), a.confirm)
    if a.op == "frame":
        if a.png:
            out = cl.frame(a.run, a.name, png=True)
            if isinstance(out, dict):
                return out
            Path(a.png).write_bytes(out)
            return {"api": agentapi.API_VERSION, "ok": True, "png": str(a.png), "bytes": len(out)}
        return cl.frame(a.run, a.name)
    return _watch(cl, a)


def _watch(cl: Any, a: argparse.Namespace) -> int | None:
    """Print the run, then one line each time it changes, until it ends (or --count/--for)."""
    first = cl.status(a.run, log=0, events=5)
    if not first.get("ok"):
        _print(first, True)
        return exit_code(first)
    _print({"api": agentapi.API_VERSION, "ok": True, "run": first["run"]}, True)
    n, since, t_end = 1, float(first["run"].get("last_wall") or 0), time.monotonic() + (a.duration or 1e18)
    while first["run"]["state"] not in agentapi.ENDED:
        if (a.count and n >= a.count) or time.monotonic() >= t_end:
            return None
        ans = cl.wait(a.run, since=since, timeout=min(a.wait_timeout, max(0.5, t_end - time.monotonic())))
        if not ans.get("ok"):
            _print(ans, True)
            return exit_code(ans)
        if ans.get("changed"):
            run = ans["run"]
            since = float(run.get("last_wall") or since)
            _print({"api": agentapi.API_VERSION, "ok": True, "run": {k: v for k, v in run.items()
                                                                      if k not in ("log",)}}, True)
            n += 1
            first = ans
    return None
