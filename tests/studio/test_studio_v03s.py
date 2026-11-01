"""Studio correctness of the 0.3 pass: what the records say about a run, and the sign-in page.

Oracles: the simulation's own state read after the run (V4), the failing save_checkpoint a test
provides, the clock for a run that went silent, and a plain string search for the token.
The browser half is packaging/acceptance/studio_truth.mjs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

import conatus as cn
from conatus.errors import NumericalError
from conatus.studio import agentapi
from conatus.telemetry import sink
from conatus.telemetry.sink import Telemetry

pytest.importorskip("tornado")
from tornado.httpclient import AsyncHTTPClient  # noqa: E402
from tornado.httpserver import HTTPServer  # noqa: E402
from tornado.testing import bind_unused_port  # noqa: E402

from conatus.studio import hub as hub_module  # noqa: E402
from conatus.studio.auth import AuditLog, AuthDB  # noqa: E402
from conatus.studio.hub import Hub, make_app  # noqa: E402

FORM = {"Content-Type": "application/x-www-form-urlencoded"}


def records(spool: Path, rid: str) -> list[dict]:
    return [json.loads(x) for x in (spool / f"{rid}.jsonl").read_text().splitlines() if x.strip()]


def film(spool: Path, rid: str) -> list[dict]:
    return [json.loads(x) for x in (spool / f"{rid}.film.ndjson").read_text().splitlines() if x.strip()]


def telemetry(rid: str, spool: Path, **kw) -> Telemetry:
    return Telemetry(rid, spool=spool, capacity=4096, scalar_interval=0.0, frame_interval=0.0, max_px=64, **kw)


def control(spool: Path, rid: str, cmd: str, seq: int, **kw) -> None:
    with (spool / f"{rid}.control.jsonl").open("a") as fh:
        fh.write(json.dumps({"cmd": cmd, "seq": seq, "wall": time.time(), "from": "test", **kw}) + "\n")


def heat(T0=1.0, quadratic=False):
    box = cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(-1.0, 1.0))
    m = cn.Model(box, name="heat", kappa=0.1)
    m.scalar("T")
    m.equations("dt(T) - kappa*lap(T) = T*T" if quadratic else "dt(T) - kappa*lap(T) = 0")
    m.at("walls", T=0)
    sim = m.simulate((16, 24), scheme="RK443", dt=1e-3)
    sim.initial(T=lambda x, z: T0 * (1.5 + np.sin(x)) * np.cos(np.pi * z / 2))
    return sim


# ------------------------------------------------------------------ state of a silent run
@pytest.mark.parametrize(("status", "age", "want"), [
    ({"paused": True}, 1.0, "paused"), ({"paused": True}, agentapi.LIVE_S + 1, "stale"),
    ({}, 1.0, "running"), ({}, agentapi.LIVE_S + 1, "stale"),
])
def test_a_silent_run_is_stale_paused_or_not(status, age, want):
    now = 1000.0
    st = types.SimpleNamespace(status=status, events=[], last_wall=now - age)
    assert agentapi.run_state(st, now=now) == want


def test_a_run_the_hub_is_still_reading_is_not_stale():
    """Backlog: the last record read is old, but one was read just now. Then silence means stale."""
    now = 1000.0
    from conatus.telemetry import RunState
    st = RunState("r", 10, 10)
    st.status, st.last_wall, st.last_received = {}, now - 400.0, now - 1.0
    assert agentapi.run_state(st, now=now) == "running"
    assert any(a["code"] == "behind" for a in agentapi.alerts(st, now, True))
    st.last_received = now - agentapi.LIVE_S - 1.0
    assert agentapi.run_state(st, now=now) == "stale"


def test_a_failure_in_the_initial_output_is_failed_not_finished(tmp_path):
    """The initial row of an analysis file is written before the steps; its failure ended the watch
    as finished, because it was outside the part of run() that reports failures."""
    from conatus.studio.client import read_run
    sim = heat()
    sim.watch("io0", poll=1, every_s=0.01, scalars_every_s=0.01, spool=tmp_path)

    def bad_task(s):
        raise ValueError("boom in the initial output")
    sim.output(tmp_path / "a.h5", {"x": bad_task}, every=1)
    with pytest.raises(Exception, match="boom"):
        sim.run(steps=2)
    sim.unwatch()
    st = read_run(tmp_path, "io0")
    assert agentapi.run_state(st) == "failed"
    assert "boom" in st.status["failed"]


def test_a_paused_run_sends_a_heartbeat(tmp_path, monkeypatch):
    assert sink.PAUSE_HEARTBEAT_S < agentapi.LIVE_S
    monkeypatch.setattr(sink, "PAUSE_HEARTBEAT_S", 0.05)
    tel = telemetry("hb", tmp_path, control_interval=0.01)
    control(tmp_path, "hb", "pause", 1)
    solver = types.SimpleNamespace(iteration=1, t=0.1)
    th = threading.Thread(target=tel._handle_controls, args=(solver, None, None, {}))
    th.start()
    time.sleep(0.6)
    control(tmp_path, "hb", "resume", 2)
    th.join(10)
    assert not th.is_alive()
    tel.close()
    paused = [r for r in records(tmp_path, "hb") if r["kind"] == "status" and r.get("paused")]
    assert len(paused) >= 3


# ------------------------------------------------------------------ checkpoint and fork outcomes
class Solver:
    iteration, t = 7, 0.5

    def __init__(self, error=None):
        self.error = error

    def save_checkpoint(self, path):
        if self.error:
            raise self.error
        Path(path).write_bytes(b"x")
        return path


def apply_one(tmp_path, rid, cmd, solver, checkpoint_dir, **kw):
    tel = telemetry(rid, tmp_path, control_interval=0.0)
    control(tmp_path, rid, cmd, 41, **kw)
    tel._handle_controls(solver, checkpoint_dir, None, {})
    tel.close()
    return {r["name"]: r for r in records(tmp_path, rid) if r["kind"] == "event"}


@pytest.mark.parametrize("cmd", ["checkpoint", "fork"])
def test_outcome_written(tmp_path, cmd):
    kw = {"params": {"Ra": 2}} if cmd == "fork" else {}
    ev = apply_one(tmp_path, f"ok_{cmd}", cmd, Solver(), tmp_path, **kw)
    e = ev[f"{cmd}_written"]
    assert e["seq"] == 41 and e["iteration"] == 7 and Path(e["path"]).exists()
    assert ev["command_received"]["seq"] == 41 and f"{cmd}_failed" not in ev


@pytest.mark.parametrize("cmd", ["checkpoint", "fork"])
def test_outcome_failed_with_its_reason(tmp_path, cmd):
    ev = apply_one(tmp_path, f"bad_{cmd}", cmd, Solver(OSError("disk full")), tmp_path)
    e = ev[f"{cmd}_failed"]
    assert e["seq"] == 41 and "disk full" in e["reason"] and "OSError" in e["reason"]
    assert f"{cmd}_written" not in ev


def test_outcome_failed_without_a_directory(tmp_path):
    ev = apply_one(tmp_path, "nodir", "checkpoint", Solver(), None)
    assert "no checkpoint_dir" in ev["checkpoint_failed"]["reason"]


# ------------------------------------------------------------------ V4: the state at the end is stored
def test_the_last_stored_values_are_the_final_state(tmp_path):
    sim = heat()
    sim.watch("v4", film={"T": cn.view("T", scale="smoke")}, stats={"max T": lambda s: float(np.max(s.grid("T")))},
              poll=1, every_s=30.0, scalars_every_s=30.0, controls_every_s=1e-6, spool=tmp_path)
    sim.run(steps=40)
    truth_t, truth_max = float(sim.t), float(np.max(sim.grid("T")))
    sim.unwatch()
    rec = records(tmp_path, "v4")
    stat = [r for r in rec if r["kind"] == "scalars" and "max T" in r]
    assert stat[-1]["t"] == pytest.approx(truth_t) and stat[-1]["max T"] == pytest.approx(truth_max)
    frames = [r for r in film(tmp_path, "v4") if r.get("name") == "T"]
    assert frames[-1]["t"] == pytest.approx(truth_t)
    final = [r for r in rec if r["kind"] == "status" and r.get("final")]
    assert len(final) == 1 and rec.index(final[0]) > rec.index(stat[-1])


def test_no_second_record_for_the_same_iteration(tmp_path):
    sim = heat()
    sim.watch("once", film={"T": cn.view("T", scale="smoke")}, stats={"m": lambda s: float(np.max(s.grid("T")))},
              poll=1, every_s=1e-6, scalars_every_s=1e-6, controls_every_s=0.0, spool=tmp_path)
    sim.run(steps=3)
    sim.unwatch()
    its = [r["iteration"] for r in film(tmp_path, "once") if r.get("name") == "T"]
    assert len(its) == len(set(its))
    scal = [r["iteration"] for r in records(tmp_path, "once") if r["kind"] == "scalars"]
    assert len(scal) == len(set(scal))


def test_a_failed_run_is_not_sampled_at_close(tmp_path):
    calls = []
    sim = heat(T0=20.0, quadratic=True)

    def stat(s):
        calls.append(s.iteration)
        return float(np.max(s.grid("T")))

    sim.watch("blow", stats={"m": stat}, poll=1, every_s=1e-6, scalars_every_s=1e-6,
              controls_every_s=1e-6, spool=tmp_path)
    with pytest.raises(NumericalError):
        sim.run(steps=2000)
    n = len(calls)
    sim.unwatch()
    assert len(calls) == n
    status = [r for r in records(tmp_path, "blow") if r["kind"] == "status"][-1]
    st = types.SimpleNamespace(status=status, events=[], last_wall=time.time())
    assert status.get("final") and agentapi.run_state(st) == "failed"


# ------------------------------------------------------------------ sign-in page
def with_hub(body, *, hosted=False):
    async def go():
        db = AuthDB({})
        db.add_user("u", None)
        hub = Hub(db, audit=AuditLog(None, echo=False), local=not hosted)
        server = HTTPServer(make_app(hub))
        sock, port = bind_unused_port()
        server.add_sockets([sock])
        try:
            await body(AsyncHTTPClient(), f"http://127.0.0.1:{port}")
        finally:
            server.stop()
    asyncio.run(go())


def test_login_says_why():
    secret = "S3CRET-" + "x" * 40

    async def body(c, base):
        r = await c.fetch(base + "/login", method="POST", body="", headers=FORM, raise_error=False)
        assert r.code == 400 and b"Enter the token" in r.body
        r = await c.fetch(base + "/login", method="POST", body=f"token={secret}", headers=FORM, raise_error=False)
        assert r.code == 401 and b"not valid" in r.body and secret.encode() not in r.body
        r = await c.fetch(base + "/login", method="POST", body=f"token={secret}", raise_error=False,
                          headers={**FORM, "Origin": "http://evil.example"})
        assert r.code == 403 and b"another site" in r.body
        r = await c.fetch(base + "/login?reason=expired")
        assert b"session has ended" in r.body
        r = await c.fetch(base + "/login?reason=%3Cscript%3Ealert(1)%3C/script%3E")
        assert b"alert(1)" not in r.body
        for _ in range(6):
            r = await c.fetch(base + "/login", method="POST", body=f"token={secret}", headers=FORM, raise_error=False)
        assert r.code == 429 and b"Too many failed attempts" in r.body and r.headers["Retry-After"] == "900"
    with_hub(body)


def test_an_empty_form_does_not_count_as_a_failed_attempt():
    async def body(c, base):
        for _ in range(8):
            r = await c.fetch(base + "/login", method="POST", body="", headers=FORM, raise_error=False)
            assert r.code == 400
    with_hub(body)


def test_a_token_in_a_url_reaches_no_log_and_no_page(caplog):
    secret = "S3CRET-" + "y" * 40

    async def body(c, base):
        with caplog.at_level(logging.DEBUG):
            r = await c.fetch(f"{base}/login?token={secret}")
            r2 = await c.fetch(f"{base}/nope?token={secret}", raise_error=False)
        assert secret.encode() not in r.body and secret.encode() not in r2.body
        assert r.effective_url.endswith(f"?token={secret}")     # the request did carry it
    with_hub(body)
    assert "/login" in caplog.text                              # the access log was captured
    assert secret not in caplog.text


def test_the_local_page_asks_for_the_token_only():
    async def body(c, base):
        r = await c.fetch(base + "/login")
        assert b'name="token"' in r.body and b'name="password"' not in r.body
    with_hub(body)


def test_the_hosted_page_keeps_the_password_form():
    assert 'name="password"' in hub_module.LOGIN_FIELDS_HUB and 'name="token"' in hub_module.LOGIN_FIELDS_HUB


def test_the_login_page_has_no_script_and_no_external_resource():
    assert "<script" not in hub_module.LOGIN_PAGE
    assert "http://" not in hub_module.LOGIN_PAGE and "https://" not in hub_module.LOGIN_PAGE
    assert 'autocomplete="off"' in hub_module.LOGIN_FIELDS_LOCAL and 'method="post"' in hub_module.LOGIN_PAGE
