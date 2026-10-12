"""cn.io (I/O): HDF5 analysis files, exact checkpoints, restart of an IVP from a file.

Problem: viscous Burgers u_t - nu u_xx = -u u_x on a periodic interval (nonlinear, so the
multistep history F(X^{n-i}) matters for an exact restart). Checks: a run cut in two by an HDF5
checkpoint is bit-for-bit the uninterrupted run (SBDF3, CNAB2, RK443); HDF5 and npz restarts agree
bit-for-bit; analysis rows are the grid values at their times and are readable with h5py alone;
an existing file is never replaced or appended to silently; periodic checkpoints keep the k most
recent and the latest one resumes a "next job"; mismatched restarts are refused.
"""
import h5py
import numpy as np
import pytest

import conatus as cn
from conatus.errors import InputTypeError, OptionError


def _sim(scheme="SBDF3", dt=2e-3, N=32):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="burgers", nu=0.05)
    m.scalar("u")
    m.equations("dt(u) - nu*dx(dx(u)) = -u*dx(u)")
    sim = m.simulate(N=N, scheme=scheme, dt=dt)
    sim.initial(u=lambda x: np.sin(x) + 0.3 * np.cos(2 * x))
    return sim


@pytest.mark.parametrize("scheme", ["SBDF3", "CNAB2", "RK443"])
def test_hdf5_checkpoint_restart_is_bit_for_bit(tmp_path, scheme):
    ref = _sim(scheme)
    ref.run(steps=24)
    a = _sim(scheme)
    a.run(steps=11)
    p = a.save(tmp_path / "ck.h5", grids=True)
    assert p.suffix == ".h5" and not list(tmp_path.glob("*.part"))
    b = _sim(scheme)                                   # a fresh process would do the same
    b.restart(p).run(steps=13)
    assert b.iteration == ref.iteration and b.t == ref.t
    assert np.array_equal(b.coeffs("u"), ref.coeffs("u"))
    assert np.array_equal(b.solver.X, ref.solver.X)
    assert "State restarted from" in b.explain()


def test_hdf5_and_npz_restarts_agree(tmp_path):
    a = _sim()
    a.run(steps=7)
    a.save(tmp_path / "ck.h5")
    a.save(tmp_path / "ck.npz")
    b, c = _sim(), _sim()
    b.restart(tmp_path / "ck.h5").run(steps=5)
    c.restart(tmp_path / "ck.npz").run(steps=5)
    assert np.array_equal(b.solver.X, c.solver.X)


def test_checkpoint_file_is_readable_and_explained(tmp_path):
    a = _sim()
    a.run(steps=5)
    p = a.save(tmp_path / "ck.h5", grids=True)
    with h5py.File(p, "r") as f:                       # without Conatus
        assert f.attrs["format"] == "conatus.checkpoint/1"
        assert f.attrs["iteration"] == 5 and f.attrs["scheme"] == "SBDF3"
        assert np.array_equal(f["grids/u"][()], a.grid("u"))
        assert "history" in f and len(f["history"]) == 6        # 3 levels of M X and F
    d = cn.io.read(p)
    assert isinstance(d, cn.io.CheckpointData) and d.iteration == 5
    assert d.provenance["numerics"] and "code" in d.provenance
    txt = d.explain()
    assert "iteration 5" in txt and "3 history level(s)" in txt and "code:" in txt


def test_restart_refusals(tmp_path):
    a = _sim()
    a.run(steps=3)
    p = a.save(tmp_path / "ck.h5")
    with pytest.raises(OptionError, match="mismatch on dt"):
        _sim(dt=1e-3).restart(p)
    with pytest.raises(OptionError, match="mismatch on scheme"):
        _sim("CNAB2").restart(p)
    (tmp_path / "empty").mkdir()
    with pytest.raises(OptionError, match="no checkpoint"):
        _sim().restart(cn.io.latest(tmp_path / "empty"))
    with pytest.raises(OptionError, match="no such directory"):   # was None
        cn.io.latest(tmp_path / "empty_dir_never_created")
    out = _sim().output(tmp_path / "a.h5", ["u"], every=1)
    out.write()
    with pytest.raises(OptionError, match="not a Conatus checkpoint"):
        _sim().restart(tmp_path / "a.h5")
    with pytest.raises(OptionError, match="needs an HDF5 path"):
        a.save(tmp_path / "x.npz", grids=True)


def test_analysis_file_rows_and_scales(tmp_path):
    sim = _sim()
    ke = lambda s: float(0.5 * np.mean(s.grid("u") ** 2))      # noqa: E731
    out = sim.output(tmp_path / "snap.h5", tasks={"u": "u", "KE": ke}, every=5)
    u0 = sim.grid("u").copy()
    sim.run(steps=20)
    d = cn.io.read(tmp_path / "snap.h5")
    # the initial state is a row (output(initial=True) is the default)
    assert list(d.iteration) == [0, 5, 10, 15, 20] and list(d.write_number) == [1, 2, 3, 4, 5]
    assert np.allclose(d.sim_time, np.array([0, 5, 10, 15, 20]) * 2e-3, rtol=0, atol=1e-15)
    assert np.array_equal(d.tasks["u"][0], u0)
    assert np.array_equal(d.tasks["u"][-1], sim.grid("u"))
    assert d.tasks["KE"].shape == (5,) and d.tasks["KE"][-1] == ke(sim)
    assert np.array_equal(d.scales["x"], sim.coordinate("x"))
    assert d.attrs["every"] == 5 and "function" in d.sources["KE"]
    with h5py.File(tmp_path / "snap.h5", "r") as f:     # readable without Conatus
        assert f["tasks/u"].shape == (5, 32) and f.attrs["writes"] == 5
    assert "every 5 steps" in out.explain() and "snap.h5" in sim.explain()
    assert "state at the start (iteration 0" in out.explain()
    assert "5 write(s)" in d.explain()
    # initial=False: the former layout, first row at iteration `every`
    s2 = _sim()
    o2 = s2.output(tmp_path / "late.h5", ["u"], every=5, initial=False)
    s2.run(steps=10)
    assert list(cn.io.read(tmp_path / "late.h5").iteration) == [5, 10]
    assert "first row at iteration 5" in o2.explain()
    with pytest.raises(OptionError, match="initial"):
        _sim().output(tmp_path / "bad.h5", ["u"], every=5, initial="yes")


def test_analysis_file_modes(tmp_path):
    p = tmp_path / "s.h5"
    sim = _sim()
    sim.output(p, ["u"], every=4)
    sim.run(steps=8)
    sim.save(tmp_path / "ck.h5")
    with pytest.raises(OptionError, match="exists"):
        _sim().output(p, ["u"], every=4)
    with pytest.raises(OptionError, match="differ"):
        _sim().restart(tmp_path / "ck.h5").output(p, {"u": "u", "v": "u"}, every=4, mode="a")
    with pytest.raises(OptionError, match="before the last write"):
        _sim().output(p, ["u"], every=4, mode="a")       # t = 0 < last write
    nxt = _sim().restart(tmp_path / "ck.h5")            # the next job appends to the same file
    o = nxt.output(p, ["u"], every=4, mode="a")
    nxt.run(steps=8)
    d = cn.io.read(p)                                   # iteration 8 is not written twice
    assert list(d.iteration) == [0, 4, 8, 12, 16] and list(d.write_number) == [1, 2, 3, 4, 5]
    assert "already in the file" in o.explain()
    fresh = _sim()
    fresh.output(p, ["u"], every=4, mode="w")           # explicit replacement
    e = cn.io.read(p)                                   # datasets appear with the first row
    assert len(e.sim_time) == 0 and e.tasks == {} and "no row" in e.explain()


def test_task_declarations_are_checked(tmp_path):
    sim = _sim()
    with pytest.raises(InputTypeError, match="not an unknown"):
        sim.output(tmp_path / "a.h5", ["T"], every=1)
    with pytest.raises(InputTypeError, match="components are for vectors"):
        sim.output(tmp_path / "a.h5", ["u.x"], every=1)
    with pytest.raises(OptionError, match="integer >= 1"):
        sim.output(tmp_path / "a.h5", ["u"], every=2.0)
    with pytest.raises(OptionError, match="mode="):
        sim.output(tmp_path / "a.h5", ["u"], every=1, mode="x")


def test_periodic_checkpoints_and_next_job(tmp_path):
    ref = _sim()
    ref.run(steps=30)
    a = _sim()
    series = a.checkpoints(tmp_path / "chk", every=6, keep=2)
    a.run(steps=20)                                    # "job 1" stops at 20 (last file: 18)
    files = sorted(p.name for p in (tmp_path / "chk").glob("*.h5"))
    assert files == ["checkpoint_000000012.h5", "checkpoint_000000018.h5"]
    last = cn.io.latest(tmp_path / "chk")
    assert last.name == "checkpoint_000000018.h5" and "2 most recent" in series.explain()
    b = _sim().restart(last)                           # "job 2"
    b.run(steps=12)
    assert np.array_equal(b.solver.X, ref.solver.X)


def test_expert_solver_with_analysis_callback(tmp_path):
    sim = _sim()
    solver = sim.solver
    out = cn.io.AnalysisFile(solver, tmp_path / "e.h5", ["u"], every=3)
    solver.run(9, callback=out, every=out.every)
    d = cn.io.read(tmp_path / "e.h5")
    assert list(d.iteration) == [3, 6, 9] and d.scales == {}
    assert np.array_equal(d.tasks["u"][-1], solver.grid("u"))
    assert "expert solver" in out.explain()


def test_checkpoint_directory_of_another_run_refused_and_keep_across_jobs(tmp_path):
    """Files already in a checkpoint directory were ignored: a
    new run replaced the other run's files of the same names (silently), cn.io.latest could pick
    the other run's later file, and keep= never removed the files of the previous job."""
    d = tmp_path / "chk"
    a = _sim()
    a.checkpoints(d, every=6, keep=2)
    a.run(steps=20)                                     # job 1: files 12, 18
    other = _sim()                                      # a new run (not restarted) in that directory
    other.checkpoints(d, every=6, keep=2)
    with pytest.raises(OptionError, match="another run"):
        other.run(steps=6)
    assert sorted(p.name for p in d.glob("*.h5")) == ["checkpoint_000000012.h5",
                                                      "checkpoint_000000018.h5"]
    early = _sim().restart(d / "checkpoint_000000012.h5")   # restart behind the latest file
    early.checkpoints(d, every=6, keep=2)
    with pytest.raises(OptionError, match="later checkpoints"):
        early.run(steps=6)
    b = _sim().restart(cn.io.latest(d))                 # job 2: continues the run, keep applies
    b.checkpoints(d, every=6, keep=2)
    b.run(steps=12)
    assert sorted(p.name for p in d.glob("*.h5")) == ["checkpoint_000000024.h5",
                                                      "checkpoint_000000030.h5"]
    ref = _sim()
    ref.run(steps=30)
    assert np.array_equal(b.solver.X, ref.solver.X)
    c = _sim()
    c.initial(u=lambda x: np.sin(x))                    # a new initial state after a restart:
    c.restart(cn.io.latest(d))                          # (restart then initial = a new run)
    c.initial(u=lambda x: np.sin(x))
    c.checkpoints(d, every=6, keep=2)
    with pytest.raises(OptionError, match="another run"):
        c.run(steps=6)


def test_analysis_resume_after_a_job_killed_between_checkpoints(tmp_path):
    """Review of the job 1 writes rows every 4 steps and a checkpoint every 6, and
    is killed at step 20 (rows up to 20, last checkpoint 18). mode='a' refuses (time would go
    back) and now says what to write; mode='resume' drops rows 20 (declared) and the exact restart
    writes it again: the file equals the uninterrupted run's file row for row, bit for bit."""
    ref = _sim()
    ref.output(tmp_path / "ref.h5", ["u"], every=4)
    ref.run(steps=32)
    p, d = tmp_path / "s.h5", tmp_path / "chk"
    a = _sim()
    a.output(p, ["u"], every=4)
    a.checkpoints(d, every=6, keep=2)
    a.run(steps=20)
    b = _sim().restart(cn.io.latest(d))                 # iteration 18
    with pytest.raises(OptionError, match="mode='resume'"):
        b.output(p, ["u"], every=4, mode="a")
    out = b.output(p, ["u"], every=4, mode="resume")
    assert out.dropped == 1 and "dropped 1 row(s) written after iteration 18" in out.explain()
    assert cn.io.read(p).iteration.tolist() == [0, 4, 8, 12, 16]
    b.run(steps=14)                                     # iteration 18: no row of the start
    got, want = cn.io.read(p), cn.io.read(tmp_path / "ref.h5")
    assert got.iteration.tolist() == want.iteration.tolist() == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert "not a multiple of every" in out.explain()
    assert np.array_equal(got.tasks["u"], want.tasks["u"])
    assert np.array_equal(got.sim_time, want.sim_time)
