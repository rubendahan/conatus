"""``sim.initial(t=...)``: a non-zero initial time (feat2).

Problem: u_t = u_yy / 2 on [0, 5] with the moving walls u(0) = cos(t), u(5) = exp(-5) cos(t - 5),
exact u = exp(-y) cos(t - y) (tests/model/test_native_time.py). Started at t0 = 0.7 from the
exact state, the run over one time unit reaches the exact state at 1.7 to the error of the same
run started at 0 (the boundary data are read on the given clock), while the same data started
at the default t = 0 are off by O(1). Outputs, checkpoints, restarts, resume_or_start and explain()
carry the start time; inconsistent uses are refused."""
import h5py
import numpy as np
import pytest

import conatus as cn
from conatus.errors import OptionError

Y = 5.0


def _model():
    m = cn.Model(cn.Box(y=cn.walls(0, Y)), name="moving walls")
    m.scalar("u")
    m.equations("dt(u) - dy(dy(u))/2 = 0")
    m.at("bottom", u="cos(t)")
    m.at("top", f"u = {float(np.exp(-Y))!r}*cos(t - {Y})")
    return m


def _exact(t):
    return lambda y: np.exp(-y) * np.cos(t - y)


def _err(sim):
    y = np.asarray(sim.coordinate("y"))
    return float(np.abs(sim.grid("u") - np.exp(-y) * np.cos(sim.t - y)).max())


def test_start_time_drives_the_boundary_data_and_the_clock():
    a = _model().simulate(N=32, scheme="RK443", dt=0.01, dealias=1)
    a.initial(u=_exact(0.0))
    a.run(until=1.0)
    b = _model().simulate(N=32, scheme="RK443", dt=0.01, dealias=1)
    b.initial(u=_exact(0.7), t=0.7)
    assert b.t == 0.7 and b.iteration == 0
    b.run(until=1.7)
    assert b.iteration == 100 and abs(b.t - 1.7) < 1e-14
    assert _err(b) < 5 * _err(a) + 1e-12, (_err(a), _err(b))
    c = _model().simulate(N=32, scheme="RK443", dt=0.01, dealias=1)
    c.initial(u=_exact(0.7))                       # same data on the default clock t = 0
    c.run(until=1.0)
    assert _err(c) > 1e-3
    assert "at t = 0.7 (initial(t=...)" in b.explain()
    assert "initial(t=" not in a.explain()
    with pytest.raises(cn.errors.ResolutionError, match="already at t = 1.7"):
        b.run(until=1.0)


def test_outputs_checkpoints_and_restart_keep_the_clock(tmp_path):
    sim = _model().simulate(N=16, scheme="SBDF2", dt=0.01)
    sim.initial(u=_exact(2.5), t=2.5)
    sim.output(tmp_path / "run.h5", ["u"], every=5)
    sim.run(steps=10)
    path = sim.save(tmp_path / "state.h5")
    with h5py.File(tmp_path / "run.h5", "r") as f:
        times = list(f["scales/sim_time"][:])
    assert np.allclose(times, [2.5, 2.55, 2.6]), times
    ref = sim.solver.X.copy()
    sim.run(steps=5)
    fresh = _model().simulate(N=16, scheme="SBDF2", dt=0.01)
    fresh.restart(path)
    assert abs(fresh.t - 2.6) < 1e-14 and fresh.solver._t_initial == 2.5
    assert np.array_equal(fresh.solver.X, ref)
    fresh.run(steps=5)
    assert np.array_equal(fresh.solver.X, sim.solver.X)        # bit for bit on the same clock
    p2 = fresh.save(tmp_path / "again.npz")                     # carried by later checkpoints
    again = _model().simulate(N=16, scheme="SBDF2", dt=0.01)
    again.restart(p2)
    assert again.solver._t_initial == 2.5


def test_resume_or_start_compares_the_start_time(tmp_path):
    d = tmp_path / "chain"

    def job(t, steps=10):
        s = _model().simulate(N=16, scheme="SBDF2", dt=0.01)
        s.resume_or_start(d, u=_exact(t), t=t)
        s.checkpoints(d, every=5, keep=2)
        s.run(steps=steps)
        return s
    a = job(1.25)
    assert abs(a.t - 1.35) < 1e-14 and "at t = 1.25" in a.explain()
    b = job(1.25)                                      # the same script: continued
    assert abs(b.t - 1.45) < 1e-14 and "continued from" in b.explain()
    assert "its chain started at t = 1.25" in b.explain()     # stated on a resume too
    with pytest.raises(OptionError, match="started at t = 1.25"):
        job(0.0)
    with pytest.raises(OptionError, match="started at t = 1.25"):
        _model().simulate(N=16, scheme="SBDF2", dt=0.01).resume_or_start(d, u=_exact(0.0))


def test_checkpoint_without_the_record(tmp_path):
    """A checkpoint written before the start time was recorded: continued without t= (its own
    clock), refused with a non-zero t= (cannot be checked)."""
    import json
    sim = _model().simulate(N=16, scheme="SBDF2", dt=0.01)
    sim.initial(u=_exact(0.0))
    sim.run(steps=5)
    d = tmp_path / "old"
    d.mkdir()
    p = sim.save(d / "checkpoint_000000005.npz")
    with np.load(p) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    del meta["t_initial"]
    np.savez(p, meta=json.dumps(meta), **arrays)
    ok = _model().simulate(N=16, scheme="SBDF2", dt=0.01).resume_or_start(d, u=_exact(0.0))
    assert ok.solver._t_initial is None and ok.iteration == 5
    with pytest.raises(OptionError, match="does not record the start time"):
        _model().simulate(N=16, scheme="SBDF2", dt=0.01).resume_or_start(d, u=_exact(3.0), t=3.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1.0", 1j])
def test_bad_initial_times_are_refused(bad):
    sim = _model().simulate(N=16, scheme="SBDF2", dt=0.01)
    with pytest.raises(OptionError, match="finite real number"):
        sim.initial(u=_exact(0.0), t=bad)


def test_an_unknown_named_t_keeps_its_meaning():
    """A model with an unknown named t has no time t: initial(t=...) is that unknown's data (v0.1)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="unknown t")
    m.scalar("t")
    m.equations("dt(t) - dx(dx(t)) = 0")
    sim = m.simulate(N=16, scheme="SBDF1", dt=1e-3)
    sim.initial(t=lambda x: np.sin(x))
    assert sim.t == 0.0
    x = np.asarray(sim.coordinate("x"))
    assert np.allclose(sim.grid("t"), np.sin(x))


@pytest.mark.parametrize("name", ["initial_time", "implicit_coriolis"])
def test_guide_examples_run_as_written(name):
    """The guide blocks of io.md (start time) and operators.md (implicit Coriolis) are regions of
    docs/site/guide/_check_features.py: each region runs on its own in a fresh namespace (as a
    reader would paste it), then the checked function with its asserts (review of feat2)."""
    import importlib.util
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "docs" / "site" / "guide"
    src = (root / "_check_features.py").read_text()
    tag = {"initial_time": "initial-time", "implicit_coriolis": "coriolis"}[name]
    body = src.split(f"# [{tag}]\n", 1)[1].split(f"# [/{tag}]", 1)[0]
    code = "\n".join(ln[4:] for ln in body.splitlines())
    page = {"initial_time": "io.md", "implicit_coriolis": "operators.md"}[name]
    assert re.search(rf'start-after: "# \[{re.escape(tag)}\]"', (root / page).read_text())
    exec(compile(code, f"{page}[{tag}]", "exec"), {})  # noqa: S102 -- the guide block itself
    spec = importlib.util.spec_from_file_location("_check_features", root / "_check_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CHECKS[name]()
