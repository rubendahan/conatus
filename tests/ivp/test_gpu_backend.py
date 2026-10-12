"""IVPSolver(backend="cuda") vs the CPU backend (same algebra; cross-backend model §74:
agreement to rounding amplified by the dynamics, not bitwise). Skipped without a GPU."""
import numpy as np
import pytest

from ivp_cases import ks_1d, rayleigh_benard_2d

from conatus.runtime import cuda
from conatus.timestep import IVPSolver

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not cuda.available(), reason="no CUDA device on this node")]


def _t_init(x, z):
    return 1e-2 * np.cos(np.pi * z) * np.cos(np.pi * x)


def _pair(plan, f, scheme, n, **kw):
    out = []
    for backend in ("cpu", "cuda"):
        s = IVPSolver(plan, scheme, 1e-3, backend=backend, **kw)
        s.set_state({f["T"]: _t_init})
        s.run(n)
        out.append(s)
    return out


@pytest.mark.parametrize("scheme", ["RK443", "SBDF2", "RK222"])
@pytest.mark.parametrize("sym", [False, True])
def test_rb2d_gpu_matches_cpu(scheme, sym):
    plan, f = rayleigh_benard_2d(32, 16, Ra=1e5, Lx=2.0, symmetric=sym)
    c, g = _pair(plan, f, scheme, 20)
    for name in ("T", "u"):
        a, b = c.coeffs(name), g.coeffs(name)
        assert np.abs(a - b).max() <= 1e-11 * np.abs(a).max()


def test_sbdf4_rk443_startup_gpu():
    plan, f = rayleigh_benard_2d(32, 16, Ra=1e5, Lx=2.0)
    c, g = _pair(plan, f, "SBDF4", 10, startup="rk443")
    a, b = c.coeffs("T"), g.coeffs("T")
    assert np.abs(a - b).max() <= 1e-11 * np.abs(a).max()


def test_ks_gpu_matches_cpu():
    plan, u = ks_1d(128)[:2]
    out = []
    for backend in ("cpu", "cuda"):
        s = IVPSolver(plan, "RK443", 0.05, backend=backend)
        s.set_state({u: lambda x: np.cos(x / 16) * (1 + np.sin(x / 16))})
        s.run(50)
        out.append(s.coeffs(u.name))
    assert np.abs(out[0] - out[1]).max() <= 1e-10 * np.abs(out[0]).max()


def test_gpu_restart_bitwise(tmp_path):
    plan, f = rayleigh_benard_2d(32, 16, Ra=1e5, Lx=2.0)
    a = IVPSolver(plan, "SBDF3", 1e-3, backend="cuda", startup="rk443")
    a.set_state({f["T"]: _t_init})
    a.run(6)
    p = a.save_checkpoint(tmp_path / "ck")
    a.run(6)
    b = IVPSolver(plan, "SBDF3", 1e-3, backend="cuda", startup="rk443")
    b.load_checkpoint(p)
    b.run(6)
    assert np.array_equal(a.coeffs("T"), b.coeffs("T"))


@pytest.mark.parametrize("grids", [False, True])
def test_gpu_restart_h5_bitwise_and_graph_after_restart(tmp_path, grids):
    """HDF5 checkpoint from the GPU backend (payload copied to the host; grids=True adds the grid
    values) restarts bit for bit, and a CUDA graph captured after load_checkpoint replays the same
    steps as run() (campagne gpu1)."""
    pytest.importorskip("h5py")
    plan, f = rayleigh_benard_2d(32, 16, Ra=1e5, Lx=2.0)
    a = IVPSolver(plan, "RK443", 1e-3, backend="cuda")
    a.set_state({f["T"]: _t_init})
    a.run(6)
    p = a.save_checkpoint(tmp_path / "ck.h5", grids=grids)
    a.run(6)
    b = IVPSolver(plan, "RK443", 1e-3, backend="cuda")
    b.load_checkpoint(p)
    b.run(6)
    assert np.array_equal(a.coeffs("T"), b.coeffs("T"))
    c = IVPSolver(plan, "RK443", 1e-3, backend="cuda")
    c.load_checkpoint(p)
    c.run_graph(6)
    assert c.iteration == a.iteration == 12
    x, y = a.coeffs("T"), c.coeffs("T")
    assert np.abs(x - y).max() <= 1e-13 * np.abs(x).max()


@pytest.mark.parametrize("sym", [False, True])
def test_cuda_graph_replay_matches_plain_steps(sym):
    plan, f = rayleigh_benard_2d(32, 16, Ra=1e5, Lx=2.0, symmetric=sym)
    a = IVPSolver(plan, "RK443", 1e-3, backend="cuda")
    b = IVPSolver(plan, "RK443", 1e-3, backend="cuda")
    for s in (a, b):
        s.set_state({f["T"]: _t_init})
    a.run(12)
    b.run_graph(5)
    b.run_graph(7)
    assert b.iteration == 12
    x, y = a.coeffs("T"), b.coeffs("T")
    assert np.abs(x - y).max() <= 1e-13 * np.abs(x).max()


@pytest.mark.parametrize("scheme,startup", [("SBDF2", "lower_order"), ("SBDF3", "rk443"),
                                             ("SBDF4", "lower_order")])
@pytest.mark.parametrize("sym", [False, True])
def test_cuda_graph_sbdf_matches_plain_steps(scheme, startup, sym):
    """SBDF graph replay: start-up by plain steps, then the captured full-order step on fixed
    history buffers; same numbers as run(), and the history written back lets run() continue."""
    plan, f = rayleigh_benard_2d(32, 16, Ra=1e5, Lx=2.0, symmetric=sym)
    a = IVPSolver(plan, scheme, 1e-3, backend="cuda", startup=startup)
    b = IVPSolver(plan, scheme, 1e-3, backend="cuda", startup=startup)
    for s in (a, b):
        s.set_state({f["T"]: _t_init})
    a.run(16)
    b.run_graph(9)          # includes the start-up steps
    b.run_graph(4)
    b.run(3)                # plain steps after the replay use the written-back history
    assert b.iteration == 16
    x, y = a.coeffs("T"), b.coeffs("T")
    assert np.abs(x - y).max() <= 1e-13 * np.abs(x).max()
    assert len(a._MXh) == len(b._MXh)


def test_cuda_graph_refuses_time_dependent_f():
    plan, f = rayleigh_benard_2d(16, 8, Ra=1e5, Lx=2.0)
    s = IVPSolver(plan, "SBDF2", 1e-3, backend="cuda", time_parameter="t")
    with pytest.raises(ValueError, match="depends on t"):
        s.run_graph(1)
