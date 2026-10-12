"""One script for every job of a restart chain: ``resume_or_start``
starts from the initial state when the checkpoint directory holds nothing and continues from
the latest checkpoint otherwise; outputs without ``mode`` then create (fresh start; an existing file is refused) or continue
(restart) their file, declared by ``explain()``. The chain is bit-for-bit the uninterrupted run,
and the analysis file holds every row once."""
import h5py
import numpy as np
import pytest

import conatus as cn
from conatus.errors import OptionError


def _job(d, out, steps):
    """The same script for every job: build, resume or start, write, run."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="burgers", nu=0.05)
    m.scalar("u")
    m.equations("dt(u) - nu*dx(dx(u)) = -u*dx(u)")
    sim = m.simulate(N=32, scheme="SBDF2", dt=2e-3)
    sim.resume_or_start(d, u=lambda x: np.sin(x) + 0.3 * np.cos(2 * x))
    sim.checkpoints(d, every=4, keep=2)
    sim.output(out, ["u"], every=2)
    sim.run(steps=steps)
    return sim


def test_one_script_for_every_job(tmp_path):
    d, out = tmp_path / "chk", tmp_path / "run.h5"
    a = _job(d, out, 10)                    # job 1: fresh start, directory created
    ex = a.explain()
    assert "no checkpoint there (directory created)" in ex and "mode 'w' chosen" in ex
    b = _job(d, out, 6)                     # job 2: restart from iteration 8, rows 10 dropped
    ex = b.explain()
    assert "continued from checkpoint_" in ex and "(iteration 8, t = 0.016)" in ex and "mode 'resume' chosen" in ex
    assert b.iteration == 14
    ref = _job(tmp_path / "ref", tmp_path / "ref.h5", 14)
    assert np.array_equal(b.solver.X, ref.solver.X)       # the chain is the uninterrupted run
    with h5py.File(out, "r") as f, h5py.File(tmp_path / "ref.h5", "r") as g:
        assert list(f["scales/iteration"][:]) == list(g["scales/iteration"][:])
        assert np.array_equal(f["tasks/u"][:], g["tasks/u"][:])


def test_fresh_start_never_replaces_an_existing_output(tmp_path):
    """Review outputs but no checkpoint (a first job killed before its first checkpoint, an
    emptied checkpoint directory): the fresh start refuses the existing file, mode='w' replaces
    it explicitly."""
    d, out = tmp_path / "chk", tmp_path / "run.h5"
    out.write_bytes(b"rows of an earlier run")
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat")
    m.scalar("u")
    m.equations("dt(u) - dx(dx(u)) = 0")
    sim = m.simulate(N=16, scheme="SBDF1", dt=1e-3)
    sim.resume_or_start(d, u=lambda x: np.sin(x))
    with pytest.raises(OptionError, match="the file exists but resume_or_start found no checkpoint"):
        sim.output(out, ["u"], every=2)
    assert out.read_bytes() == b"rows of an earlier run"      # untouched
    sim.output(out, ["u"], every=2, mode="w")                 # explicit: replaced
    assert h5py.is_hdf5(out)


def test_resume_or_start_needs_the_initial_state(tmp_path):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat")
    m.scalar("u")
    m.equations("dt(u) - dx(dx(u)) = 0")
    sim = m.simulate(N=16, scheme="SBDF1", dt=1e-3)
    with pytest.raises(OptionError, match="no initial state"):
        sim.resume_or_start(tmp_path / "d")


def test_latest_refuses_a_missing_directory(tmp_path):
    with pytest.raises(OptionError, match="no such directory"):
        cn.io.latest(tmp_path / "chk_typo")
    (tmp_path / "chk").mkdir()
    assert cn.io.latest(tmp_path / "chk") is None          # an empty directory: no checkpoint yet
