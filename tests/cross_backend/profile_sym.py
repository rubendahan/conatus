"""Where an RK443 step goes on the GPU for RB 2D (primitive variables, fused RHS), full space
versus the x -> -x symmetry subspace: F, linear solves, scatter/gather, timed with device syncs
(parts slightly overestimate the whole). Explains the measured symmetry gain."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ivp"))
from ivp_cases import rayleigh_benard_2d  # noqa: E402

from conatus.timestep import IVPSolver  # noqa: E402


def profile(Nx, Nz, sym, sync):
    plan, f = rayleigh_benard_2d(Nx, Nz, Ra=1e6, Lx=4.0, symmetric=sym)
    s = IVPSolver(plan, "RK443", 1e-5, backend="cuda", f_backend="fused")
    s.set_state({f["T"]: lambda x, z: 1e-3 + 0 * x})
    acc: dict = defaultdict(float)

    def wrap(obj, name, label):
        fn = getattr(obj, name)

        def w(*a, **k):
            sync()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            sync()
            acc[label] += time.perf_counter() - t0
            return r
        setattr(obj, name, w)

    wrap(s, "F", "F")
    wrap(s._fused, "evaluate", "F.fused")
    wrap(s.system, "solve", "solve")
    wrap(s.system, "scatter", "scatter")
    wrap(s.system, "gather_rhs", "gather_rhs")
    for _ in range(3):
        s.step()
    acc.clear()
    n = 20
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        s.step()
    sync()
    return dict(size=f"{Nx}x{Nz}", sym=sym, unknowns=int(s.system.n),
                step_ms=(time.perf_counter() - t0) / n * 1e3,
                parts_ms={k: v / n * 1e3 for k, v in acc.items()})


def main():
    import cupy as cp
    sync = cp.cuda.Device().synchronize
    out = Path(os.environ.get("CN_OUT", "."))
    sizes = [tuple(map(int, a.split("x"))) for a in sys.argv[1:]] or [(256, 64), (512, 128), (1024, 256)]
    res = []
    for Nx, Nz in sizes:
        for sym in (False, True):
            row = profile(Nx, Nz, sym, sync)
            res.append(row)
            print(json.dumps(row), flush=True)
            cp.get_default_memory_pool().free_all_blocks()
    (out / "profile_sym.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
