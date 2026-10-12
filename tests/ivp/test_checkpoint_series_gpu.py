"""Periodic HDF5 checkpoints on the GPU backend (``sim.checkpoints``) and restart from
``cn.io.latest``: a run split in two jobs equals the uninterrupted run bit for bit, for a
one-step and a multistep scheme (finding 2; gpu1 campaign). Skipped without a GPU."""
import numpy as np
import pytest

import conatus as cn
from conatus.runtime import cuda

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not cuda.available(), reason="no CUDA device on this node")]


def _sim(scheme):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(-1.0, 1.0)), name="Allen-Cahn")
    m.scalar("u")
    m.equations("dt(u) - 0.05*lap(u) = u - u**3")
    m.at("walls", u=0)
    return m.simulate((32, 24), scheme=scheme, dt=1e-3, backend="cuda")


def _init(sim):
    sim.initial(u=lambda x, z: 0.3 * np.cos(np.pi * z / 2) * (1 + 0.5 * np.sin(2 * x)))


@pytest.mark.parametrize("scheme", ["RK443", "SBDF3"])
def test_gpu_checkpoint_series_restart_is_bit_identical(tmp_path, scheme):
    pytest.importorskip("h5py")
    a = _sim(scheme)
    _init(a)
    a.checkpoints(tmp_path, every=5, keep=2)
    a.run(steps=12)                                 # files at 5 and 10 (keep 2), state at 12
    p = cn.io.latest(tmp_path)
    assert p is not None and p.name == "checkpoint_000000010.h5", p
    b = _sim(scheme)                                # "next job": fresh solver, restart from disk
    b.restart(p)                                    # (before a writes 15 and 20 and drops 10)
    a.run(steps=8)                                  # uninterrupted run: 20 steps
    assert b.solver.iteration == 10
    b.run(steps=10)
    assert b.solver.iteration == a.solver.iteration == 20 and b.solver.t == a.solver.t
    xa, xb = a.solver.X, b.solver.X
    assert type(xa).__module__.startswith("cupy")
    assert bool((xa == xb).all()), float(abs(xa - xb).max())
