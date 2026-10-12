"""User pointwise functions (m.function) on the GPU backend: the function receives CuPy arrays on
the device (never a silent host round trip), and the GPU run agrees with the CPU run to rounding
amplified by the dynamics. explain() of the GPU solver is recorded (the fused RHS paths do not
compile user functions: F then goes through the CuPy evaluator, declared there). Skipped
without a GPU ."""
import numpy as np
import pytest

import conatus as cn
from conatus.runtime import cuda

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not cuda.available(), reason="no CUDA device on this node")]


_SEEN: set = set()


def _h(a):                                      # Allen-Cahn-type reaction, array-generic
    _SEEN.add(type(a).__module__.split(".")[0])
    return a ** 3 / (1 + a ** 2)


def _dh(a):
    return a ** 2 * (3 + a ** 2) / (1 + a ** 2) ** 2


def _run(backend, seen):
    # one function object for both runs: a name registered twice with two different callables is
    # refused (pointwise(): "a name means one thing")
    _SEEN.clear()
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="reaction-diffusion")
    m.scalar("u")
    m.function("h", _h, derivative=_dh)
    m.equations("dt(u) - 0.1*lap(u) = u - h(u)")
    m.at("walls", u=0)
    sim = m.simulate((16, 24), scheme="RK443", dt=1e-3, backend=backend)
    sim.initial(u=lambda x, z: np.sin(np.pi * z) * (1 + 0.5 * np.cos(x)))
    sim.run(steps=200)
    seen |= _SEEN
    return np.asarray(sim.grid("u")), sim.solver.explain()


def test_user_function_on_gpu_matches_cpu():
    seen_cpu, seen_gpu = set(), set()
    u_cpu, _ = _run("cpu", seen_cpu)
    u_gpu, why = _run("cuda", seen_gpu)
    print(why)
    assert seen_cpu == {"numpy"}
    assert seen_gpu == {"cupy"}, f"the user function received {seen_gpu} on the GPU backend"
    # 200 RK443 steps of a smooth, dissipative problem: rounding-level agreement (the CPU
    # self-sensitivity of such runs is ~1e-15; same bound as the Chebyshev families test)
    d = np.max(np.abs(u_gpu - u_cpu)) / np.max(np.abs(u_cpu))
    assert d < 1e-12, d
