"""Where does a GPU RB 2D step go? Times F, solves, gather/scatter and SpMV separately
(cupyx.profiler.benchmark: CPU + GPU times). Usage: python ... NxxNz [...]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ivp_cases import rayleigh_benard_2d  # noqa: E402

from cupyx.profiler import benchmark  # noqa: E402

from conatus.timestep import IVPSolver  # noqa: E402


def row(name, r):
    print(f"{name:28s} cpu {np.median(r.cpu_times) * 1e3:8.3f} ms   gpu {np.median(r.gpu_times[0]) * 1e3:8.3f} ms", flush=True)


for arg in sys.argv[1:] or ["256x64", "1024x256"]:
    Nx, Nz = map(int, arg.split("x"))
    plan, f = rayleigh_benard_2d(Nx, Nz, Ra=1e6, Lx=4.0)
    s = IVPSolver(plan, "RK443", 1e-5, backend="cuda")
    s.set_state({f["T"]: lambda x, z: 1e-3 * np.cos(np.pi * z) * np.cos(np.pi * x / 2)})
    s.run(2)
    X = s.X
    print(f"== {Nx}x{Nz}, n={s.system.n}")
    row("step RK443", benchmark(s.step, n_repeat=10))
    row("F", benchmark(s.F, (X, 0.0), n_repeat=20))
    sysm = s.system
    rhs = sysm.M @ X
    row("solve", benchmark(sysm.solve, (1.0, 0.5e-5, rhs), n_repeat=20))
    row("M@X", benchmark(lambda: sysm.M @ X, n_repeat=20))
    row("scatter", benchmark(sysm.scatter, (X,), n_repeat=20))
    data = dict(zip(sysm.variables, sysm.scatter(X)))
    ev = s.evaluator
    for e, eq in zip(s._rhs_exprs, plan.problem.equations):
        if e is not None:
            row(f"eval {str(e)[:20]}", benchmark(ev.evaluate, (e, data, dict(plan.params)), n_repeat=20))
    outs = [None if e is None else ev._embed(ev.evaluate(e, data, dict(plan.params)), eq.lhs.rank, sh)
            for e, sh, eq in zip(s._rhs_exprs, sysm.eq_shapes, plan.problem.equations)]
    row("gather_rhs", benchmark(sysm.gather_rhs, (outs,), n_repeat=20))
    u = data[f["u"]]
    b = ev.bases[1]
    row("to_grid(u)", benchmark(ev.to_grid, (u, 1), n_repeat=20))
    g = ev.to_grid(u, 1)
    row("to_coeffs(u)", benchmark(ev.to_coeffs, (g, 1), n_repeat=20))
