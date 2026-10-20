"""``sim.watch(...)`` (native layer, Studio rebuild): declared views and statistics only, explained,
and a watched run is bit-identical to an unwatched one (film, curves and Studio commands included)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import conatus as cn
from conatus.errors import InputTypeError, OptionError
from conatus.model.watch import power_spectrum
from conatus.telemetry import SpoolReader, StopRun

L = 22.0


def _ks():
    box = cn.Box(x=cn.periodic(L))
    m = cn.Model(box, name="Kuramoto-Sivashinsky L=22")
    u = m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=64, scheme="RK443", dt=0.05)
    sim.initial(u=lambda x: np.cos(2 * np.pi * x / L) * (1 + np.sin(2 * np.pi * x / L)))
    return sim, u


def _channel():
    box = cn.Box(x=cn.periodic(4.0), z=cn.walls(-1, 1))
    m = cn.Model(box, name="channel", nu=1e-2)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u·grad(u)", "div(u) = 0",
                "dt(T) - nu*lap(T) = -u·grad(T)")
    m.at("bottom", u=0, T=1)
    m.at("top", u=0, T=0)
    sim = m.simulate(N=(16, 12), scheme="RK443", dt=1e-3)
    sim.initial(T=lambda x, z: 0.5 - z / 2 + 0.1 * np.sin(np.pi * x / 2) * np.cos(np.pi * z / 2))
    return sim


def test_watched_run_is_bit_identical(tmp_path):
    plain, _ = _ks()
    plain.run(steps=300)
    sim, _ = _ks()
    history = []

    def spacetime(s):                       # a declared space-time diagram (a function of sim)
        history.append(s.grid("u").copy())
        return np.array(history)

    sim.watch("ks", film={"u(x,t)": cn.view(spacetime, scale="signed")},
                  stats={"energy": lambda s: 0.5 * float(np.mean(s.grid("u") ** 2)),
                         "E(k)": cn.spectrum("u", axis="x"), "u(x)": cn.profile("u", along="x"),
                         "boom": lambda s: 1 / 0},                     # a failing statistic
                  every_s=1e-4, scalars_every_s=1e-4, poll=5, spool=tmp_path)
    sim.run(steps=300)
    sim.unwatch()
    assert np.array_equal(plain.solver.X, sim.solver.X)                # observing changes nothing
    r = SpoolReader(tmp_path, max_scalars=1000, max_logs=100)
    r.poll()
    r.poll_film()
    run = r.runs["ks"]
    assert [v["name"] for v in run.header["views"]] == ["u(x,t)"]
    assert {s["name"]: s["kind"] for s in run.header["stats"]} == {
        "energy": "series", "E(k)": "spectrum", "u(x)": "profile", "boom": "series"}
    assert "power spectrum of u along x" in run.header["watch_explain"]
    assert run.scalars[-1]["iteration"] == 300 and run.scalars[-1]["boom"].startswith("error")
    assert run.frames["E(k)"]["kind"] == "curve" and run.frames["E(k)"]["logy"] is True
    assert r.films["ks"].summary()["names"]["u(x,t)"]["count"] >= 1
    assert run.status["final"] is True


def test_watch_and_user_callback_share_the_run(tmp_path):
    sim, _ = _ks()
    seen = []
    sim.watch("ks2", stats={"m": lambda s: float(np.mean(s.grid("u")))}, poll=4, spool=tmp_path)
    sim.run(steps=24, callback=lambda s: seen.append(s.iteration), every=6)
    sim.unwatch()
    assert seen == [6, 12, 18, 24]
    with pytest.raises(OptionError, match="together"):
        sim.run(steps=1, callback=lambda s: None)            # also on the native layer


def test_stop_from_the_studio_ends_sim_run(tmp_path):
    sim, _ = _ks()
    sim.watch("ks3", poll=1, spool=tmp_path, controls_every_s=0.0)
    (tmp_path / "ks3.control.jsonl").write_text(json.dumps({"cmd": "checkpoint", "seq": 1}) + "\n"
                                                + json.dumps({"cmd": "stop", "seq": 2}) + "\n")
    with pytest.raises(StopRun):
        sim.run(steps=50)
    assert cn.StopRun is StopRun
    sim.unwatch()
    assert list((tmp_path / "checkpoints").glob("ks3_it*.npz"))


def test_declarations_are_checked_with_the_fix(tmp_path):
    sim, _ = _ks()
    with pytest.raises(InputTypeError, match="not a declared quantity"):
        sim.watch("x", stats={"energy": "energy"}, spool=tmp_path)
    with pytest.raises(InputTypeError, match="not an unknown"):
        sim.watch("x", film=["T"], spool=tmp_path)
    with pytest.raises(OptionError, match="scale"):
        cn.view("u", scale="rainbow")
    ch = _channel()
    with pytest.raises(InputTypeError, match="name a component"):
        ch.watch("x", film=["u"], spool=tmp_path)
    with pytest.raises(OptionError, match="not periodic"):
        ch.watch("x", stats={"E": cn.spectrum("T", axis="z")}, spool=tmp_path)
    w = ch.watch("x", film={"T": "T", "w": cn.view("u.z", scale="signed")}, spool=tmp_path)
    with pytest.raises(OptionError, match="already watched"):
        ch.watch("y", spool=tmp_path)
    text = ch.explain(full=True)
    assert "resampled onto uniform points" in text and "Telemetry(" in text
    ch.run(steps=20)
    ch.unwatch()
    head = json.loads((tmp_path / "x.jsonl").read_text().splitlines()[0])
    assert head["views"][0]["default"] is True and head["views"][1]["field"] == "u.z"
    assert w.closed


def test_spectrum_is_parseval_and_profile_is_the_mean(tmp_path):
    rng = np.random.default_rng(3)
    for n in (16, 17):
        a = rng.standard_normal((n, 5))
        k, e = power_spectrum(a, 0, 2 * np.pi)
        e0 = np.mean(np.mean(a, axis=0) ** 2)
        assert np.isclose(e.sum() + e0, np.mean(a ** 2), rtol=1e-12)
        assert np.allclose(k, np.arange(1, n // 2 + 1))
    sim = _channel()
    w = sim.watch("p", stats={"T(z)": cn.profile("T", along="z")}, spool=tmp_path)
    z, prof = w._curve_fn(w.stats["T(z)"])(None)
    assert np.allclose(prof, sim.grid("T").mean(axis=0)) and np.all(np.diff(z) > 0)
    sim.unwatch()


def test_explain_declares_the_sent_resolution_of_each_film(tmp_path):
    """Studio review: the film resolution is declared (sent shape, stride, bytes per frame), the
    default sends grids up to 384 nodes per axis whole, and a smaller max_px strides visibly."""
    sim = _channel()
    w = sim.watch("res", film={"T": cn.view("T")}, spool=tmp_path)
    assert "sent 16x12 of the 16x12 grid nodes (stride 1x1, max_px=384)" in w.explain()
    sim.unwatch()
    sim = _channel()
    w = sim.watch("res2", film={"T": cn.view("T")}, max_px=8, spool=tmp_path)
    assert "sent 8x6 of the 16x12 grid nodes (stride 2x2, max_px=8)" in w.explain()
    sim.unwatch()
    with pytest.raises(OptionError, match="max_px"):
        _channel().watch("bad", film={"T": cn.view("T")}, max_px=1, spool=tmp_path)


def test_watch_refuses_names_and_shapes_it_cannot_send(tmp_path):
    """Studio review: bad names, built-in names, view/stat clashes, ids the relay would drop, and
    a film of the wrong dimension are refused at sim.watch(...), not per frame or silently."""
    sim = _channel()
    cases = [
        (dict(stats={"a/b": cn.profile("T", along="z")}), "cannot name"),
        (dict(stats={"t": lambda s: 1.0}), "always sent"),
        (dict(film={"T": cn.view("T")}, stats={"T": cn.profile("T", along="z")}), "both a view"),
        (dict(film={"T": cn.view("T", mode="isosurface")}), "3D view"),
        (dict(film={"T": cn.view("T", at={"z": 0.0})}), "a film is 2D"),
    ]
    for kw, msg in cases:
        with pytest.raises((InputTypeError, OptionError), match=msg):
            sim.watch("ok_id", spool=tmp_path, **kw)
    with pytest.raises(InputTypeError, match="run id"):
        sim.watch("my run", film={"T": cn.view("T")}, spool=tmp_path)


def test_run_every_is_checked_even_when_watched(tmp_path):
    sim, _ = _ks()
    sim.watch("ks3", stats={"m": lambda s: 0.0}, poll=3, spool=tmp_path)
    for bad in (0, -3, 2.5, True):
        with pytest.raises(OptionError, match="every"):
            sim.run(steps=4, callback=lambda s: None, every=bad)
    assert sim.iteration == 0
    sim.unwatch()


def test_native_run_forwards_record_and_refuses_fractional_steps(tmp_path):
    """``sim.run`` did not forward ``record=``/``cache=`` (the run registry was
    reachable only through ``sim.solver``), the native callback received the expert solver
    instead of the Simulation (the watch functions get the Simulation), and ``steps=2.5`` (also
    ``IVPSolver.run(2.5)`` and ``Ensemble.run``) took 2 steps without a word."""
    from conatus.runs import Run
    sim, _ = _ks()
    seen = []
    assert sim.run(steps=4, callback=lambda s: seen.append(s), every=2) is sim
    assert seen == [sim, sim]
    for bad in (2.5, True, -1):
        with pytest.raises(OptionError, match="steps"):
            sim.run(steps=bad)
        with pytest.raises(OptionError, match="steps="):
            sim.solver.run(bad)
    assert sim.iteration == 4
    run = sim.run(steps=2, record=dict(name="ks_native", root=str(tmp_path)))
    assert isinstance(run, Run) and sim.iteration == 6
    with pytest.raises(OptionError, match="cache=True needs record"):
        sim.run(steps=2, cache=True)


def test_watch_on_sine_cosine_axis_uses_the_physical_nodes(tmp_path):
    """On walls(a, b, basis="sincos") sim.grid returns the N + 1
    nodes of [a, b]; the watch used the 2N nodes of the doubled period (film refused, profile
    with mismatched lengths)."""
    box = cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos"))
    m = cn.Model(box, kappa=1.0)
    m.scalar("T")
    m.equations("dt(T) - kappa*lap(T) = 0")
    m.at("walls", T=0)
    sim = m.simulate(N=(8, 8), scheme="RK222", dt=1e-3)
    sim.initial(T=lambda x, z: np.sin(np.pi * z) * np.cos(np.pi * x))
    w = sim.watch("sincos_nodes", film=["T"], stats={"prof": cn.profile("T", along="z")},
                  spool=tmp_path / "spool")
    try:
        frame = w._frame_fn(w.views["T"])(None)
        assert frame.shape == (8, 9)
        z, prof = w._curve_fn(w.stats["prof"])(None)
        np.testing.assert_array_equal(z, sim.coordinate("z"))
        assert prof.shape == z.shape
    finally:
        w.close()
