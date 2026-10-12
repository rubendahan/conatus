"""Fine split of one fused F evaluation on the GPU, full space vs x -> -x subspace (RB 2D
primitives): scatter, factor coefficients, transforms per axis, pointwise kernel, assembly,
gather. Instruments XT/np.fft calls by wrapping them; timings include device syncs."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ivp"))
from ivp_cases import rayleigh_benard_2d  # noqa: E402

from conatus.timestep import IVPSolver  # noqa: E402
from conatus.timestep import fused as FM  # noqa: E402


def main():
    import cupy as cp
    sync = cp.cuda.Device().synchronize
    acc: dict = defaultdict(float)

    def timed(label, fn):
        def w(*a, **k):
            sync()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            sync()
            acc[label] += time.perf_counter() - t0
            return r
        return w

    FM.XT.backward = timed("XT.backward", FM.XT.backward)
    FM.XT.forward = timed("XT.forward", FM.XT.forward)
    cp.fft.irfft = timed("fft.irfft", cp.fft.irfft)
    cp.fft.rfft = timed("fft.rfft", cp.fft.rfft)
    FM.FusedRHS._pointwise = timed("pointwise", FM.FusedRHS._pointwise)
    FM.FusedRHS._diff = timed("diff", FM.FusedRHS._diff)
    FM.FusedRHS._full = timed("full_embed", FM.FusedRHS._full)
    out = Path(os.environ.get("CN_OUT", "."))
    res = []
    for Nx, Nz in [(256, 64), (512, 128), (1024, 256)]:
        for sym in (False, True):
            plan, f = rayleigh_benard_2d(Nx, Nz, Ra=1e6, Lx=4.0, symmetric=sym)
            s = IVPSolver(plan, "SBDF2", 1e-5, backend="cuda", f_backend="fused")
            s.set_state({f["T"]: lambda x, z: 1e-3 * np.cos(np.pi * z) * np.cos(2 * np.pi * x / 4.0)})
            s.system.scatter = timed("scatter", s.system.scatter)
            s.system.gather_rhs = timed("gather_rhs", s.system.gather_rhs)
            s._fused.evaluate = timed("F.evaluate", s._fused.evaluate)
            s.F = timed("F", s.F)
            s.system.solve = timed("solve", s.system.solve)
            s.run(4)
            acc.clear()
            n = 20
            sync()
            t0 = time.perf_counter()
            s.run(n)
            sync()
            row = dict(size=f"{Nx}x{Nz}", sym=sym, step_ms=(time.perf_counter() - t0) / n * 1e3,
                       parts_ms={k: round(v / n * 1e3, 4) for k, v in sorted(acc.items())})
            res.append(row)
            print(json.dumps(row), flush=True)
            s = None
            cp.get_default_memory_pool().free_all_blocks()
    (out / "profile_fused_parts.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
