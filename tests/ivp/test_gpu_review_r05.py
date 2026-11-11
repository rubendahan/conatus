"""Review of v05b: the integrator's CUDA graph with systems longer than the
band kernel's shared-memory limit (n * 8 > 40 KiB), where each captured solve allocates its work
vector on the stream (an allocation node in the graph). Oracle: the plain device steps and the CPU
backend. Skipped without a GPU."""
import numpy as np
import pytest

from ivp_cases import rayleigh_benard_2d

from conatus.runtime import cuda
from conatus.timestep import IVPSolver

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not cuda.available(), reason="no CUDA device on this node")]


def _t_init(x, z):
    return 1e-2 * np.cos(np.pi * z) * np.cos(np.pi * x)


@pytest.mark.parametrize("scheme", ["RK443", "SBDF2"])
def test_graph_replay_with_long_systems(scheme):
    plan, f = rayleigh_benard_2d(8, 1300, Ra=1e3, Lx=2.0)
    out = {}
    for name, backend, graph in (("cpu", "cpu", False), ("plain", "cuda", False), ("graph", "cuda", True)):
        s = IVPSolver(plan, scheme, 1e-5, backend=backend)
        s.set_state({f["T"]: _t_init})
        if graph:
            s.run_graph(4)
            s.run_graph(4)
        else:
            s.run(8)
        assert s.iteration == 8
        out[name] = np.asarray(s.coeffs("T"))
    scale = np.abs(out["cpu"]).max()
    assert np.abs(out["graph"] - out["plain"]).max() <= 1e-13 * scale
    assert np.abs(out["plain"] - out["cpu"]).max() <= 1e-10 * scale
