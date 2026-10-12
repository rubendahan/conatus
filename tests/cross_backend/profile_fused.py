"""Where an RK443 step goes on the GPU (RB 2D psi-theta, fused RHS): each component of
IVPSolver._rk_step timed with device syncs (F, linear solves, M@X / L@X, gather/scatter),
plus the whole step. Timings include the syncs, so the parts overestimate the whole a little."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ivp"))
from ivp_cases import rayleigh_benard_2d_psi_theta  # noqa: E402

from conatus.timestep import IVPSolver  # noqa: E402


def main():
    import cupy as cp
    sync = cp.cuda.Device().synchronize
    out = Path(os.environ.get("CN_OUT", "."))
    res = []
    sizes = [tuple(map(int, a.split("x"))) for a in sys.argv[1:]] or [(256, 64), (512, 128), (1024, 256)]
    for Nx, Nz in sizes:
        plan, f = rayleigh_benard_2d_psi_theta(Nx, Nz, Ra=1e6, Lx=4.0)
        s = IVPSolver(plan, "RK443", 1e-5, backend="cuda", f_backend="fused")
        s.set_state({f["theta"]: lambda x, z: 1e-3 * x * 0 + 1e-3})
        acc: dict = defaultdict(float)
        ncall: dict = defaultdict(int)

        def wrap(obj, name, label):
            fn = getattr(obj, name)

            def w(*a, **k):
                sync()
                t0 = time.perf_counter()
                r = fn(*a, **k)
                sync()
                acc[label] += time.perf_counter() - t0
                ncall[label] += 1
                return r
            setattr(obj, name, w)

        sysm = s.system
        wrap(s, "F", "F")
        wrap(sysm, "solve", "solve")
        wrap(sysm, "scatter", "scatter")
        wrap(sysm, "gather_rhs", "gather_rhs")
        wrap(s._fused, "evaluate", "F.fused_evaluate")
        for _ in range(3):
            s.step()
        acc.clear()
        ncall.clear()
        n = 20
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            s.step()
        sync()
        tot = (time.perf_counter() - t0) / n
        # Band of the per-mode systems vs the comparator (kl=9, ku=17, border folded into the band).
        bands = [dict(label=b[0], n=b[1], border=b[2], kl=b[3], ku=b[4]) for b in getattr(sysm, "host", sysm).bands()]
        row = dict(size=f"{Nx}x{Nz}", step_ms=tot * 1e3, bands=bands,
                   parts_ms={k: v / n * 1e3 for k, v in acc.items()},
                   calls_per_step={k: v / n for k, v in ncall.items()})
        res.append(row)
        print(json.dumps(row), flush=True)
        s = None
        cp.get_default_memory_pool().free_all_blocks()
    (out / "profile_fused.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
