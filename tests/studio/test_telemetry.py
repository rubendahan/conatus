"""Studio telemetry: non-blocking, lossy only where declared, never harmful to the run."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from conatus.telemetry import SpoolReader, StopRun, Telemetry, decode_frame, encode_frame, record

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ivp"))


def _tel(tmp_path, **kw):
    kw.setdefault("capacity", 1000)
    kw.setdefault("scalar_interval", 0.0)
    kw.setdefault("frame_interval", 0.0)
    kw.setdefault("max_px", 64)
    return Telemetry("run_a", spool=tmp_path, **kw)


def test_records_reach_the_spool_and_reader(tmp_path):
    tel = _tel(tmp_path)
    tel.scalars(t=0.5, energy=1.25, bad=float("nan"))
    tel.log("hello")
    tel.frame("T", np.outer(np.arange(10.0), np.ones(4)), t=0.5, iteration=3)
    tel.close()
    lines = [json.loads(x) for x in (tmp_path / "run_a.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in lines][:3] == ["header", "scalars", "log"]
    assert lines[1]["bad"] == "nan"                        # non-finite shown, not dropped
    rd = SpoolReader(tmp_path, max_scalars=10, max_logs=10)
    out = rd.poll()
    st = rd.runs["run_a"]
    assert st.header["protocol"] == 1 and st.scalars[-1]["energy"] == 1.25 and "T" in st.frames
    assert any(r.get("kind") == "frame" for r in out)
    assert rd.poll() == []                                  # incremental: nothing new


def test_full_queue_drops_and_counts_without_blocking(tmp_path):
    tel = _tel(tmp_path, capacity=4)
    t0 = time.perf_counter()
    for i in range(20000):
        tel.scalars(i=i)
    assert time.perf_counter() - t0 < 2.0                  # never waits for the writer
    tel.close()
    assert tel.dropped_records > 0
    assert tel.written_records + tel.dropped_records >= 20000


def test_frame_encoding_is_declared_lossy(tmp_path):
    a = np.sin(np.linspace(0, 3, 300))[:, None] * np.cos(np.linspace(0, 2, 90))[None, :]
    fr = encode_frame("r", "f", a, max_px=64)
    b = decode_frame(fr)
    assert fr["stride"] == [5, 2] and b.shape == tuple(fr["shape"])
    assert np.max(np.abs(b - a[::5, ::2])) <= (fr["max"] - fr["min"]) / 255 + 1e-12
    with pytest.raises(ValueError):
        encode_frame("r", "f", np.array([[np.nan, 1.0]]), max_px=8)
    with pytest.raises(ValueError):
        record("nonsense", "r")


def test_unwritable_spool_never_raises_into_the_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    tel = Telemetry("r", spool=blocker / "sub", capacity=10, scalar_interval=0.0, frame_interval=0.0,
                    max_px=8)
    tel.scalars(a=1.0)
    tel.frame("f", np.ones((3, 3)))
    tel.close()
    assert tel.error is not None and "unavailable" in tel.error


def test_ivp_run_is_bit_identical_with_telemetry(tmp_path):
    from ivp_cases import burgers_1d
    from conatus.timestep import IVPSolver
    plan, u = burgers_1d(N=32)[:2]
    runs = []
    for with_tel in (False, True):
        s = IVPSolver(plan, "RK443", 1e-3)
        s.set_state({u: lambda x: 0.5 * np.sin(x)})
        if with_tel:
            tel = _tel(tmp_path)
            cb = tel.ivp_callback(scalars={"umax": lambda s: float(np.max(np.abs(s.grid("u"))))},
                                  frames={"u2d": lambda s: np.tile(s.grid("u"), (4, 1))})
            s.run(40, callback=cb, every=5)
            tel.close()
        else:
            s.run(40)
        runs.append(np.array(s.X))
    assert np.array_equal(runs[0], runs[1])
    recs = SpoolReader(tmp_path, max_scalars=100, max_logs=10)
    recs.poll()
    sc = list(recs.runs["run_a"].scalars)
    assert len(sc) == 8 and sc[-1]["iteration"] == 40 and "umax" in sc[-1] and sc[-1]["wall_per_step"] > 0


def test_controls_checkpoint_and_stop(tmp_path):
    from ivp_cases import burgers_1d
    from conatus.timestep import IVPSolver
    plan, u = burgers_1d(N=16)[:2]
    s = IVPSolver(plan, "RK222", 1e-3)
    s.set_state({u: lambda x: np.sin(x)})
    tel = _tel(tmp_path, control_interval=0.0)
    ctl = tmp_path / "run_a.control.jsonl"
    ctl.write_text(json.dumps({"cmd": "checkpoint", "seq": 1}) + "\n" + json.dumps({"cmd": "bogus"}) + "\n")
    cb = tel.ivp_callback(checkpoint_dir=tmp_path / "ck")
    (tmp_path / "ck").mkdir()
    s.run(2, callback=cb, every=1)
    assert list((tmp_path / "ck").glob("run_a_it*.npz"))
    with ctl.open("a") as fh:
        fh.write(json.dumps({"cmd": "stop", "seq": 2}) + "\n")
    with pytest.raises(StopRun):
        s.run(5, callback=cb, every=1)
    tel.close()
    logs = [json.loads(x) for x in (tmp_path / "run_a.jsonl").read_text().splitlines()]
    assert any(r["kind"] == "log" and "unknown command" in r["message"] for r in logs)


def test_newton_wrapper_reports_iterations(tmp_path):
    from conatus.ecs.newton import NewtonKrylovHookstep
    nk = NewtonKrylovHookstep(tol=1e-12, max_iterations=20, gmres_rtol=1e-10, gmres_max_iterations=10,
                              delta0=1.0, delta_min=1e-8, delta_max=10.0)
    tel = _tel(tmp_path)
    A = np.diag([2.0, 3.0])
    res = tel.wrap_newton(nk).solve(lambda z: A @ z + z**3 - 1.0,
                                    lambda z, v: A @ v + 3 * z**2 * v, np.zeros(2), scale=1.0)
    tel.close()
    recs = [json.loads(x) for x in (tmp_path / "run_a.jsonl").read_text().splitlines()]
    newton = [r for r in recs if r["kind"] == "newton"]
    assert res.relative_residual < 1e-12 and len(newton) == res.iterations and "gmres_iters" in newton[0]


def test_declared_scalars_are_throttled_but_step_scalars_are_not(tmp_path):
    from ivp_cases import burgers_1d
    from conatus.timestep import IVPSolver
    plan, u = burgers_1d(N=16)[:2]
    s = IVPSolver(plan, "RK222", 1e-3)
    s.set_state({u: lambda x: np.sin(x)})
    calls = []
    tel = _tel(tmp_path, scalar_interval=3600.0)
    cb = tel.ivp_callback(scalars={"costly": lambda s: calls.append(1) or 1.0})
    s.run(5, callback=cb, every=1)
    tel.close()
    sc = [json.loads(x) for x in (tmp_path / "run_a.jsonl").read_text().splitlines()]
    sc = [r for r in sc if r["kind"] == "scalars"]
    assert len(calls) == 1 and len(sc) == 5 and "costly" in sc[0] and "costly" not in sc[1]


def test_on_checkpoint_hook_enriches_the_event_and_never_breaks_the_run(tmp_path):
    """S2: the declared hook's dict rides on checkpoint_written; a failing hook is only logged."""
    from ivp_cases import burgers_1d
    from conatus.timestep import IVPSolver
    plan, u = burgers_1d(N=16)[:2]
    s = IVPSolver(plan, "RK222", 1e-3)
    s.set_state({u: lambda x: np.sin(x)})
    tel = _tel(tmp_path, control_interval=0.0)
    ctl = tmp_path / "run_a.control.jsonl"
    ctl.write_text(json.dumps({"cmd": "checkpoint", "seq": 1}) + "\n")
    (tmp_path / "ck").mkdir()
    hooks = [lambda solver, path: {"recurrence": {"candidates": [{"T": 1.5, "d": float("nan")}]},
                                   "seen": path.name},
             lambda solver, path: 1 / 0]
    s.run(1, callback=tel.ivp_callback(checkpoint_dir=tmp_path / "ck", on_checkpoint=hooks[0]), every=1)
    with ctl.open("a") as fh:
        fh.write(json.dumps({"cmd": "checkpoint", "seq": 2}) + "\n")
    s.run(1, callback=tel.ivp_callback(checkpoint_dir=tmp_path / "ck", on_checkpoint=hooks[1]), every=1)
    tel.close()
    recs = [json.loads(x) for x in (tmp_path / "run_a.jsonl").read_text().splitlines()]
    evs = [r for r in recs if r["kind"] == "event" and r["name"] == "checkpoint_written"]
    assert len(evs) == 2 and evs[0]["seen"] == Path(evs[0]["path"]).name and "t" in evs[0]
    assert evs[0]["recurrence"]["candidates"][0] == {"T": 1.5, "d": "nan"}      # nested, NaN kept
    assert "recurrence" not in evs[1]
    assert any(r["kind"] == "log" and "on_checkpoint failed: ZeroDivisionError" in r["message"] for r in recs)


def test_reserved_field_names_are_renamed_not_raised(tmp_path):
    """event(kind=...) raised a TypeError into the run (Newton seed job, converged TW lost)."""
    tel = _tel(tmp_path)
    tel.event("solution", kind="travelling_wave", residual=1e-15)
    tel.scalars(t=0.0, iteration=0, wall=3.0)
    tel.close()
    recs = [json.loads(x) for x in (tmp_path / "run_a.jsonl").read_text().splitlines()]
    ev = next(r for r in recs if r["kind"] == "event")
    assert ev["user_kind"] == "travelling_wave" and ev["name"] == "solution"
    sc = next(r for r in recs if r["kind"] == "scalars")
    assert sc["user_wall"] == 3.0 and sc["wall"] != 3.0
    assert sum("reserved" in r.get("message", "") for r in recs) == 2
