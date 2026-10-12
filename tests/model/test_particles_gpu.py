"""Particles, dye and sim.lyapunov on the CUDA backend: GPU = CPU to the rounding of
the two backends (the measured levels are asserted; the run prints them)."""
import numpy as np
import pytest

import conatus as cn
from conatus.runtime import cuda

pytestmark = pytest.mark.gpu
needs_gpu = pytest.mark.skipif(not cuda.available(), reason="no CUDA device")


def _couette(backend):
    m = cn.Model(cn.Box(x=cn.periodic(4.0), y=cn.walls(-1, 1)), name="Couette", nu=0.1)
    m.velocity("u"), m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    m.at("bottom", u=(-1, 0))
    m.at("top", u=(1, 0))
    m.dye("c", velocity="u", kappa=0.01)
    sim = m.simulate(N=(32, 32), scheme="RK443", dt=0.01, backend=backend)
    sim.initial(u=lambda x, y: (y + 0.2 * np.sin(np.pi * x / 2) * (1 - y ** 2), 0 * x),
                c=lambda x, y: np.exp(-((x - 2) ** 2 + y ** 2) / 0.3))
    return sim


@needs_gpu
def test_particles_and_dye_gpu_equal_cpu(tmp_path):
    X0 = np.random.default_rng(2).uniform([0, -0.9], [4, 0.9], (500, 2))
    out = {}
    for be in ("cpu", "cuda"):
        sim = _couette(be)
        pt = sim.particles(X0, path=tmp_path / f"p_{be}.h5", every=50)
        sim.dye("c", tmp_path / f"d_{be}.h5", every=50, field=False)
        sim.run(until=1.0)
        out[be] = (pt.positions(unwrapped=True), cn.io.read(tmp_path / f"d_{be}.h5"))
    dx = np.abs(out["cpu"][0] - out["cuda"][0]).max()
    dm = np.abs(out["cpu"][1].tasks["mass"] - out["cuda"][1].tasks["mass"]).max()
    print(f"GPU vs CPU: particle positions max diff {dx:.2e}, dye mass max diff {dm:.2e}")
    assert dx < 1e-11 and dm < 1e-12 * abs(out["cpu"][1].tasks["mass"][0])


@needs_gpu
def test_native_lyapunov_gpu_equal_cpu():
    res = {}
    for be in ("cpu", "cuda"):
        m = cn.Model(cn.Box(x=cn.periodic(22.0)), name="KS")
        m.scalar("u")
        m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
        sim = m.simulate(N=32, scheme="RK443", dt=0.05, backend=be)
        sim.initial(u=lambda x: 0.3 * np.cos(2 * np.pi * x / 22) + 0.2 * np.sin(4 * np.pi * x / 22))
        res[be] = sim.lyapunov(6, transient=20.0, align=10.0, T=20.0, reortho_every=10,
                               inner="euclidean", n_blocks=2, seed=0, zero_mean="u",
                               covariant=True, clv_window=5.0, clv_backward=5.0,
                               clv_checkpoints=2)
    d = np.abs(res["cpu"].exponents - res["cuda"].exponents).max()
    print(f"GPU vs CPU: exponents max diff {d:.2e}; recompute deviation (GPU) "
          f"{res['cuda'].clv_recompute_deviation:.2e}")
    # chaotic over 50 time units: the two backends separate from their rounding (lambda_1 ~ 0.05,
    # growth e^2.5): a bound far below the statistical error of such a short run
    assert d < 1e-6
    r = res["cuda"]
    assert np.all(r.clv_covariance_residual < 100 * r.clv_covariance_floor)
