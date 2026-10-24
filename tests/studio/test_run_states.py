"""How a watched run ends, as the Studio and the agent API report it (0.2.1).

Before 0.2.1 a run that raised (a blow-up stopped by the finite-state check, an error in a callback)
closed its telemetry at exit with no error and was shown as "finished"; a run stopped from the
Studio was "stopped" for the agent API but "finished" in the web app. The oracles are independent
of the telemetry: the blow-up time of dT/dt = T^2 (finite, 1/T0), the exception the run raises,
and the control line written by hand.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import conatus as cn
from conatus.errors import NumericalError
from conatus.studio import agentapi
from conatus.studio.client import SpoolClient, connect
from conatus.telemetry import SpoolReader, StopRun


def _heat(T0: float = 1.0, quadratic: bool = False):
    """Heat equation between walls; with ``quadratic`` the source T^2, whose solution of the ODE
    part blows up at t = 1/max(T0) (the diffusion rate here is 0.1 (1 + pi^2/4) ~ 0.35, far
    slower)."""
    box = cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(-1.0, 1.0))
    m = cn.Model(box, name="heat", kappa=0.1)
    m.scalar("T")
    m.equations("dt(T) - kappa*lap(T) = T*T" if quadratic else "dt(T) - kappa*lap(T) = 0")
    m.at("walls", T=0)
    sim = m.simulate((16, 24), scheme="RK443", dt=1e-3)
    sim.initial(T=lambda x, z: T0 * (1.5 + np.sin(x)) * np.cos(np.pi * z / 2))
    return sim


def _state(spool, rid: str) -> tuple[str, dict]:
    r = SpoolReader(spool, max_scalars=100, max_logs=100)
    r.poll()
    st = r.runs[rid]
    return agentapi.run_state(st), st


def test_a_diverging_run_is_failed_with_the_exception(tmp_path):
    sim = _heat(T0=20.0, quadratic=True)           # blow-up near t = 1 / 50 = 0.02
    sim.watch("blowup", poll=5, spool=tmp_path)
    with pytest.raises(NumericalError, match="non-finite"):
        sim.run(steps=2000)                         # t = 2: far past the blow-up time
    assert sim.t < 1.0                              # it stopped near the blow-up, not at the end
    sim.unwatch()
    state, st = _state(tmp_path, "blowup")
    assert state == "failed"
    assert st.status["final"] is True and st.status["failed"].startswith("NumericalError: non-finite")
    ev = [e for e in st.events if e.get("name") == "failed"]
    assert ev and ev[-1]["error_type"] == "NumericalError"
    al = agentapi.alerts(st)
    assert al[0]["code"] == "failed" and "NumericalError" in al[0]["message"]
    # the same answer through the spool client (python -m conatus studio status)
    ans = SpoolClient(tmp_path).status("blowup")
    assert ans["ok"] and ans["run"]["state"] == "failed"


def test_an_error_in_a_callback_is_failed_and_still_raised(tmp_path):
    sim = _heat()
    sim.watch("cb", poll=2, spool=tmp_path)

    def boom(s):
        if s.iteration >= 6:
            raise ValueError("boom at 6")
    with pytest.raises(ValueError, match="boom at 6"):
        sim.run(steps=20, callback=boom, every=2)
    sim.unwatch()
    state, st = _state(tmp_path, "cb")
    assert state == "failed" and st.status["failed"] == "ValueError: boom at 6"


def test_a_run_stopped_from_the_studio_is_stopped_not_finished(tmp_path):
    for rid, on_stop in (("st_raise", "raise"), ("st_return", "return")):
        sim = _heat()
        sim.watch(rid, poll=1, spool=tmp_path, controls_every_s=0.0)
        (tmp_path / f"{rid}.control.jsonl").write_text(json.dumps({"cmd": "stop", "seq": 1}) + "\n")
        if on_stop == "raise":
            with pytest.raises(StopRun):
                sim.run(steps=50)
        else:
            sim.run(steps=50, on_stop="return")
            assert sim.stopped
        assert sim.iteration < 50
        sim.unwatch()
        state, st = _state(tmp_path, rid)
        assert state == "stopped", rid
        assert st.status["stopped"].startswith("stopped from the Studio") and "failed" not in st.status


def test_a_run_that_ends_normally_is_finished(tmp_path):
    sim = _heat()
    sim.watch("ok", poll=5, spool=tmp_path)
    sim.run(steps=20)
    sim.unwatch()
    state, st = _state(tmp_path, "ok")
    assert state == "finished" and "failed" not in st.status and "stopped" not in st.status


def test_stopped_flag_alone_marks_the_final_status(tmp_path):
    # the event list is bounded: the final status carries "stopped" itself, so a stop whose
    # command_received event scrolled out of the kept events is still reported as stopped
    class St:
        status = {"final": True, "stopped": "stopped from the Studio"}
        events: list = []
        last_wall = 0.0
    assert agentapi.run_state(St()) == "stopped"
    St.status = {"final": True}
    assert agentapi.run_state(St()) == "finished"


def test_status_without_a_source_says_none_is_configured(tmp_path, monkeypatch):
    for k in ("CN_STUDIO_URL", "CN_STUDIO_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))           # no ~/.config/conatus/studio.json
    monkeypatch.setenv("CN_TELEMETRY", str(tmp_path / "empty"))
    ans = connect().status("kh_demo")
    assert not ans["ok"] and ans["code"] == 404
    assert "no Studio source is configured" in ans["error"] and "--spool" in ans["error"]
    assert "known: []" not in ans["error"]
    (tmp_path / "given").mkdir()
    ans = connect(spool=tmp_path / "given").status("kh_demo")
    assert "holds no run" in ans["error"] and "configured" not in ans["error"]
    from conatus.studio import cli
    assert cli.main(["status", "kh_demo"]) == cli.EXIT["missing"]
