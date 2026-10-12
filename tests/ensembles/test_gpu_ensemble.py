"""Ensembles and batched tangents on the CUDA backend: agreement with the CPU and with single
GPU runs; the measured level is asserted (bit for bit where observed, else the rounding bound)."""
import numpy as np
import pytest

from conatus.runtime import cuda
from conatus.timestep import IVPSolver

pytestmark = pytest.mark.gpu

from conatus.chaos import lyapunov_spectrum  # noqa: E402
from conatus.ensembles import Ensemble  # noqa: E402
from ens_cases import lorenz_solver, run_single  # noqa: E402
from ivp_cases import ks_1d, rayleigh_benard_2d  # noqa: E402

needs_gpu = pytest.mark.skipif(not cuda.available(), reason="no CUDA device")
rng = np.random.default_rng(3)


@needs_gpu
@pytest.mark.parametrize("case", ["ks", "rb2d"])
def test_gpu_ensemble_members(case):
    import cupy as cp
    plan = ks_1d(N=32, L=22.0)[0] if case == "ks" else rayleigh_benard_2d(16, 16, 2e3)[0]
    dt, amp, nsteps = (0.02, 0.3, 20) if case == "ks" else (1e-3, 0.05, 5)
    cpu = IVPSolver(plan, "RK443", dt)
    gpu = IVPSolver(plan, "RK443", dt, backend="cuda")
    X0 = amp * rng.standard_normal((6, cpu.system.n))
    ref_gpu = np.array([run_single(gpu, x, nsteps) for x in X0])
    ref_cpu = np.array([run_single(cpu, x, nsteps) for x in X0])
    loop = Ensemble(gpu, 6, rhs="loop")
    loop.set_states(cp.asarray(X0))
    loop.run(nsteps)
    assert np.array_equal(loop.states(), ref_gpu)                 # F loop: bit for bit on the GPU
    bat = Ensemble(gpu, 6, rhs="batched")
    bat.set_states(cp.asarray(X0))
    bat.run(nsteps)
    scale = np.abs(ref_cpu).max()
    assert np.abs(bat.states() - ref_gpu).max() < 1e-12 * scale
    assert np.abs(bat.states() - ref_cpu).max() < 1e-12 * scale
    one = Ensemble(gpu, 1, rhs="batched")
    one.set_states(cp.asarray(X0[4:5]))
    one.run(nsteps)
    assert np.abs(one.states()[0] - bat.states()[4]).max() < 1e-13 * scale
    assert "GPU" in bat.explain()


@needs_gpu
def test_gpu_lyapunov_equals_cpu():
    """Same trajectories, same random vectors: GPU and CPU spectra over a short window agree to
    the rounding amplified by the chaotic dynamics over T = 5 (bound 1e-8)."""
    X0 = np.array([[1.0, 1.0, 20.0], [2.0, -1.0, 25.0]])
    kw = dict(n_vectors=3, transient_steps=100, tangent_transient_steps=50, steps=500,
              reortho_every=10, inner="euclidean", n_blocks=5, seed=2, clv_window=10,
              clv_backward_transient=10)
    rc = lyapunov_spectrum(lorenz_solver(0.01), X0, **kw)
    rg = lyapunov_spectrum(lorenz_solver(0.01, backend="cuda"), X0, **kw)
    assert np.abs(rc.exponents - rg.exponents).max() < 1e-8
    assert np.abs(np.abs(rc.clv) - np.abs(rg.clv)).max() < 1e-6
    assert rg.settings["backend"] == "cuda"
