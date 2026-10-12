"""Exact tangent and adjoint of the poloidal–toroidal Couette map on the GPU (general-form
linearized RHS): dot test, Taylor test, CPU agreement, cost of vjp / direct."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from conatus.runtime import cuda
from couette_cases import BOX_GHC, couette_pt
from conatus.adjoint import FlowMap
from conatus.timestep import IVPSolver

pytestmark = pytest.mark.gpu
needs_gpu = pytest.mark.skipif(not cuda.available(), reason="no CUDA device")


def _state(s, f, amp):
    Lx, Lz = BOX_GHC
    s.set_state({f["U"]: lambda X, Y, Z: Y + 0 * X,
                 f["phi"]: lambda X, Y, Z: amp * (1 - Y**2) ** 2 * np.cos(2 * np.pi * X / Lx)
                 * np.cos(2 * np.pi * Z / Lz) * (1 + Y),
                 f["psi"]: lambda X, Y, Z: amp * (1 - Y**2) * np.sin(2 * np.pi * (X / Lx + Z / Lz))})
    return s.X.copy()


@needs_gpu
@pytest.mark.parametrize("grid", [(8, 13, 8), (32, 33, 32)])
def test_gpu_pt_flowmap_dot_taylor_and_cpu(grid):
    import cupy as cp
    plan, f = couette_pt(*grid, 400.0, BOX_GHC)
    g = IVPSolver(plan, "RK443", 0.02, backend="cuda", f_backend="fused", decouple=True)
    X = _state(g, f, 0.05)
    rng = np.random.default_rng(0)
    # V: a smooth admissible direction (difference of two smooth states). A white-noise V put
    # 1e-5 on every coefficient (tau rows and the top modes included) and fm(X + e V) blew up at
    # 32x33x32 with dt = 0.02, on CPU as on GPU: Taylor order NaN (gpu1). Smooth V, CPU:
    # Taylor errors 6.5e-6 / 1.6e-6 (order 2.0) at 32x33x32.
    _state(g, f, 0.08)
    D = g.X - X
    V = D * (1e-2 * np.sqrt(g.system.n) / float(cp.linalg.norm(D)))
    g.X = X.copy()
    W = cp.asarray(rng.standard_normal(g.system.n))
    nsteps = 10
    fm = FlowMap(g, nsteps)
    fm(X), fm.jvp(X, V), fm.vjp(X, W)                  # warm-up (plans, kernels, factors)
    t0 = time.perf_counter()
    P0 = fm(X)
    cp.cuda.Device().synchronize()
    t_dir = time.perf_counter() - t0
    t0 = time.perf_counter()
    JV = fm.jvp(X, V)
    cp.cuda.Device().synchronize()
    t_jvp = time.perf_counter() - t0
    t0 = time.perf_counter()
    JTW = fm.vjp(X, W)
    cp.cuda.Device().synchronize()
    t_vjp = time.perf_counter() - t0
    lhs, rhs = float(W @ JV), float(JTW @ V)
    dot = abs(lhs - rhs) / abs(lhs)
    # random V, W with ~1e5 entries: <W, JV> itself cancels by ~sqrt(n); the rounding of the
    # pairing is measured against ||W|| ||JV|| as well (both reported)
    dot_norm = abs(lhs - rhs) / (float(cp.linalg.norm(W)) * float(cp.linalg.norm(JV)))
    # measured (8x13x8, 10 steps): CPU 6.6e-10 / 7.9e-12, GPU 4.6e-10 / 5.5e-12: the same on both
    # backends, i.e. the conditioning of the stiff 4th-order map on random vectors, not the GPU
    assert dot < 1e-8 and dot_norm < 1e-10
    errs = [float(cp.linalg.norm(fm(X + e * V) - P0 - e * JV)) for e in (1e-3, 5e-4)]
    order = np.log2(errs[0] / errs[1])
    assert 1.9 < order < 2.1
    if grid[0] <= 8:
        c = IVPSolver(plan, "RK443", 0.02, f_backend="fused", decouple=True)
        c.X = cp.asnumpy(X)
        fc = FlowMap(c, nsteps)
        ref = fc.vjp(cp.asnumpy(X), cp.asnumpy(W))
        # GPU = CPU vjp at the conditioning level of this stiff map on a random cotangent: the dot
        # test itself is 6.6e-10 (CPU) / 4.6e-10 (GPU) relative (comment above); measured GPU - CPU
        # 3.3e-10 (L40S) and 2.7e-10 (A100,
        #) against the former 1e-10 bound (gpu1)
        assert np.linalg.norm(cp.asnumpy(JTW) - ref) < 2e-9 * np.linalg.norm(ref)
    out = os.environ.get("CN_OUT")
    if out:
        p = Path(out) / "pt_adjoint_gpu.json"
        rec = json.loads(p.read_text()) if p.exists() else {}
        rec["x".join(map(str, grid))] = dict(n=g.system.n, nsteps=nsteps, direct_s=t_dir,
                                             jvp_over_direct=t_jvp / t_dir,
                                             vjp_over_direct=t_vjp / t_dir, dot=dot, dot_normalized=dot_norm,
                                             taylor_order=order)
        p.write_text(json.dumps(rec, indent=1))
