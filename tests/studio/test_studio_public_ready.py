"""The local Studio and the agent channel, ready to be public.

What a user of the release relies on, each checked here:

* **the Studio never touches the run**: the Studio server killed (SIGKILL) in the middle of a
  watched run, or its spool directory taken away, leaves the computation identical to the bit to
  an unwatched run, and the run ends normally;
* **nothing leaves the machine**: a watched run opens no network connection (the telemetry is
  files in the spool, nothing else);
* **local by default**: ``serve --local`` refuses a non-loopback ``--host`` (it used to be silently
  ignored) and ``--allow-submit`` without the Slurm templates of a development checkout;
* **every route needs a session or a token** except the health check, the sign-in page and the
  static files of the app (code, fonts: no data), so a route added later without authentication
  fails here; a browser WebSocket from another site is refused even with a valid session cookie.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pytest

import conatus as cn
from conatus.telemetry import SpoolReader

L = 22.0
STEPS = 300


def _ks():
    box = cn.Box(x=cn.periodic(L))
    m = cn.Model(box, name="Kuramoto-Sivashinsky L=22")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=64, scheme="RK443", dt=0.05)
    sim.initial(u=lambda x: np.cos(2 * np.pi * x / L) * (1 + np.sin(2 * np.pi * x / L)))
    return sim


def _plain_state() -> np.ndarray:
    sim = _ks()
    sim.run(steps=STEPS)
    return sim.solver.X.copy()


def _watch(sim, run_id: str, spool: Path) -> None:
    history = []

    def spacetime(s):                       # the film of a 1D run: its space-time diagram
        history.append(s.grid("u").copy())
        return np.array(history[-64:])

    sim.watch(run_id, film={"u(x,t)": cn.view(spacetime, scale="signed")},
              stats={"energy": lambda s: 0.5 * float(np.mean(s.grid("u") ** 2)),
                     "E(k)": cn.spectrum("u", axis="x")},
              every_s=1e-4, scalars_every_s=1e-4, poll=5, spool=spool)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ the Studio never touches the run
def test_studio_server_killed_mid_run_changes_nothing(tmp_path):
    """A real ``serve --local`` process reads the spool of the run; it is killed with SIGKILL at
    step 100 (no clean shutdown, sockets and files left as they are). The run goes on to its end,
    bit for bit the unwatched run, and its final status reaches the spool."""
    pytest.importorskip("tornado")
    spool, state, port = tmp_path / "spool", tmp_path / "state", _free_port()
    spool.mkdir()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "conatus.studio.hub", "--state", str(state),
                             "serve", "--local", "--spool", str(spool), "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        t_end = time.monotonic() + 60
        while True:                                      # the server is up and answering
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                if proc.poll() is not None or time.monotonic() > t_end:
                    raise AssertionError("the local Studio did not start: "
                                         + proc.stdout.read().decode(errors="replace")) from None
                time.sleep(0.2)
        sim = _ks()
        _watch(sim, "ks_crash", spool)
        killed = []

        def kill(s):
            if s.iteration >= 100 and not killed:
                os.kill(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
                killed.append(s.iteration)

        sim.run(steps=STEPS, callback=kill, every=10)
        sim.unwatch()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
    assert killed == [100] and proc.returncode == -signal.SIGKILL
    assert np.array_equal(sim.solver.X, _plain_state())
    r = SpoolReader(spool, max_scalars=1000, max_logs=100)
    r.poll()
    run = r.runs["ks_crash"]
    assert run.status["final"] is True and run.scalars[-1]["iteration"] == STEPS


def test_spool_taken_away_mid_run_changes_nothing(tmp_path):
    """The spool directory disappears at step 100 (moved away while the writer thread may be
    writing in it) and its path becomes a plain file: the run ends normally, bit for bit the
    unwatched run; the failure is recorded by the telemetry, never raised into the run."""
    spool = tmp_path / "spool"
    spool.mkdir()
    sim = _ks()
    _watch(sim, "ks_gone", spool)
    done = []

    def take_away(s):
        if s.iteration >= 100 and not done:
            os.rename(spool, tmp_path / "gone")
            spool.write_text("not a directory")
            done.append(s.iteration)

    sim.run(steps=STEPS, callback=take_away, every=10)
    sim.unwatch()
    assert done == [100]
    assert np.array_equal(sim.solver.X, _plain_state())


def test_watched_run_opens_no_network_connection(tmp_path, monkeypatch):
    """The telemetry is files in the spool: a watched run (film, numbers, curves, controls) never
    connects anywhere. Every connect of the process is recorded and refused during the run."""
    calls = []

    def refuse(self, *a, **k):
        calls.append(a)
        raise OSError("network refused by the test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    sim = _ks()
    _watch(sim, "ks_offline", tmp_path)
    sim.run(steps=STEPS)
    sim.unwatch()
    monkeypatch.undo()
    assert calls == []
    assert np.array_equal(sim.solver.X, _plain_state())


# ------------------------------------------------------------------------------ local by default
def _serve_ns(**kw) -> argparse.Namespace:
    base = dict(local=True, host="127.0.0.1", allow_submit=False, certfile=None, behind_proxy=False,
                port=8765)
    return argparse.Namespace(**{**base, **kw})


def test_local_serve_refuses_a_public_host_and_a_submit_it_cannot_honour(monkeypatch):
    pytest.importorskip("tornado")
    from conatus.studio import hub
    hub.check_local_serve(_serve_ns())                               # the default passes
    hub.check_local_serve(_serve_ns(host="localhost"))
    hub.check_local_serve(_serve_ns(local=False, host="0.0.0.0"))    # not --local: other rules
    for host in ("0.0.0.0", "192.168.1.10", "::"):
        with pytest.raises(SystemExit, match="binds 127.0.0.1 only.*ssh"):
            hub.check_local_serve(_serve_ns(host=host))
    real = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: False if p.name == "submit.sh" else real(p))
    with pytest.raises(SystemExit, match="--allow-submit needs .*submit.sh"):
        hub.check_local_serve(_serve_ns(allow_submit=True))


# ------------------------------------------------------------------ every route needs a credential
PUBLIC_OK = {"/healthz", "/login"}            # answered 200 to anyone (no data)


def test_every_route_needs_a_session_or_a_token(tmp_path):
    """Sweep every route of the hub in --local mode, with a run in it, without credentials and
    with a wrong token: never a 200 except the health check and the sign-in page; never the run's
    name in an answer. A browser WebSocket from another site is refused even when signed in."""
    pytest.importorskip("tornado")
    from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPRequest
    from tornado.httpserver import HTTPServer
    from tornado.testing import bind_unused_port
    from tornado.websocket import websocket_connect

    from conatus.studio.auth import AuditLog, AuthDB
    from conatus.studio.hub import Hub, make_app

    secret = "secretrun_xyz"
    routes = [r.matcher.regex.pattern.rstrip("$") for r in make_app(Hub(
        AuthDB({}), audit=AuditLog(tmp_path / "a0.jsonl", echo=False), local=True)).default_router.rules[0]
        .target.rules]
    concrete = {"/healthz", "/login", "/logout", "/classic", "/api/runs", "/api/whoami",
                f"/api/film/{secret}", f"/api/film/{secret}/frames", f"/api/control/{secret}",
                "/api/compare", "/api/jobs/prepare", "/api/registry", "/api/v1/ping",
                "/api/v1/whoami", "/api/v1/runs", "/api/v1/board", "/api/v1/audit",
                f"/api/v1/runs/{secret}/commands", f"/api/v1/runs/{secret}/scalars",
                f"/api/v1/runs/{secret}/log", f"/api/v1/runs/{secret}/frame",
                f"/api/v1/runs/{secret}/wait", f"/api/v1/runs/{secret}", "/ws", "/relay", "/",
                f"/run/{secret}", "/compare", "/board"}
    for pat in routes:                                   # a new route must be added to the sweep
        assert any(re.fullmatch(pat, c) for c in concrete), f"route {pat!r} is not swept"

    async def go():
        db = AuthDB({})
        db.add_user("ruben", "correct horse battery")
        hub = Hub(db, audit=AuditLog(tmp_path / "audit.jsonl", echo=False), local=True)
        hub._state("local", secret)                      # a run the hub holds
        server = HTTPServer(make_app(hub))
        sock, port = bind_unused_port()
        server.add_sockets([sock])
        c, base = AsyncHTTPClient(), f"http://127.0.0.1:{port}"
        for path in sorted(concrete):
            if path in ("/ws", "/relay"):
                continue
            for auth in ({}, {"Authorization": "Bearer not-a-token"}):
                for method in ("GET", "POST"):
                    body = b"{}" if method == "POST" else None
                    hdr = {**auth, "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin",
                           "Origin": base}
                    r = await c.fetch(HTTPRequest(base + path, method=method, body=body, headers=hdr,
                                                  follow_redirects=False), raise_error=False)
                    text = (r.body or b"").decode(errors="replace")
                    if path in PUBLIC_OK and method == "GET":
                        assert r.code == 200, (path, r.code)
                    else:
                        assert r.code == 302 or 400 <= r.code < 500, (method, path, auth, r.code)
                        if r.code == 302:
                            assert r.headers["Location"] == "/login", (path, r.headers["Location"])
                    assert secret not in text, (method, path, auth)
        for path in ("/ws", "/relay"):                   # WebSockets: no session, no relay token
            with pytest.raises(HTTPClientError) as err:
                await websocket_connect(f"ws://127.0.0.1:{port}{path}")
            assert err.value.code in (401, 403)
        r = await c.fetch(base + "/login", method="POST", follow_redirects=False, raise_error=False,
                          body=urlencode({"user": "ruben", "password": "correct horse battery"}),
                          headers={"Sec-Fetch-Site": "same-origin", "Origin": base})
        assert r.code == 302, r.code
        cookie = r.headers["Set-Cookie"].split(";")[0]
        with pytest.raises(HTTPClientError) as err:      # cross-site WebSocket hijacking
            await websocket_connect(HTTPRequest(f"ws://127.0.0.1:{port}/ws", headers={
                "Cookie": cookie, "Origin": "http://evil.example"}))
        assert err.value.code == 403
        ws = await websocket_connect(HTTPRequest(f"ws://127.0.0.1:{port}/ws", headers={
            "Cookie": cookie, "Origin": base}))
        snap = json.loads(await ws.read_message())       # the same session from its own page works
        assert isinstance(snap, dict)
        ws.close()
        server.stop()
        await server.close_all_connections()

    asyncio.run(go())
