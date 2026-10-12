"""Tutorial 2b -- the same Rayleigh--Benard IVP on a GPU (requires: gpu).

Expert layer: the same tutorial written directly in the IR (bases, tau variables and lifts,
LHS/RHS by hand). The native version one directory up gives bit-identical matrices (tests/model).

Expected values (checked at the end of the script):
  CPU and GPU states agree to rounding: max |T_cpu - T_gpu| <= 1e-11 max |T| after 20 RK443
  steps at Ra = 1e5 (cross-backend model; tests/ivp/test_gpu_backend.py). Later the difference grows
  with the flow (linear growth rates up to ~225 at Ra = 1e5, then nonlinear saturation): two CPU
  orderings of the same sums (evaluator vs fused) differ by ~4e-6 after 200 steps as well;
  the CUDA-graph replay gives the same numbers as plain GPU steps (<= 1e-13 relative).
Run on a GPU node: python t02b_rayleigh_benard_gpu.py   (about a minute, mostly start-up)
"""
# [imports]
import time

import numpy as np

from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, IVPSolver,
                            Parameter, RealFourier, VectorField, discretize, div, dot, dt, grad,
                            integ, interp, lap, lift, unit_vector)
# [/imports]


# [problem]
def rayleigh_benard(Nx, Nz, Ra, Lx=2.0):
    """The IVP of tutorial 2a, wrapped in a function (same equations, same declared numerics)."""
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, Nx, (0, Lx), dealias=1.5)
    bz = Chebyshev(z, Nz, (-0.5, 0.5), dealias=1.5)
    D = Domain(cs, [bx, bz])
    u, T, p = VectorField("u", D), Field("T", D), Field("p", D)
    tp = Field("tp", Domain(cs, []))
    tT1, tT2 = Field("tT1", D.without(z)), Field("tT2", D.without(z))
    tu1, tu2 = VectorField("tu1", D.without(z)), VectorField("tu2", D.without(z))
    Ra_, Pr = Parameter("Ra", Ra), Parameter("Pr", 1.0)
    ez = unit_vector(z)
    pb = IVP([u, T, p, tp, tT1, tT2, tu1, tu2], name="Rayleigh-Benard 2D")
    pb.add_equation((div(u) + dot(ez, lift(tu2, bz, -1)) + tp, 0))
    pb.add_equation((dt(T) - lap(T) + lift(tT1, bz, -1) + lift(tT2, bz, -2), -dot(u, grad(T))))
    pb.add_equation((dt(u) - Pr * lap(u) + grad(p) - Pr * Ra_ * T * ez
                     + lift(tu1, bz, -1) + lift(tu2, bz, -2), -dot(u, grad(u))))
    pb.add_equation((interp(T, z, "left"), 0.5))
    pb.add_equation((interp(T, z, "right"), -0.5))
    pb.add_equation((interp(u, z, "left"), 0))
    pb.add_equation((interp(u, z, "right"), 0))
    pb.add_equation((integ(p), 0))
    return discretize(pb), T


plan, T = rayleigh_benard(128, 64, Ra=1e5)
T_init = lambda X, Z: -Z + 1e-2 * np.cos(np.pi * X) * np.cos(np.pi * Z)  # noqa: E731
# [/problem]

# [backends]
runs, early = {}, {}
for label, kw in [("cpu", dict(backend="cpu")),
                  ("gpu", dict(backend="cuda", f_backend="fused"))]:   # both choices explicit
    s = IVPSolver(plan, "RK443", 2e-4, **kw)
    s.set_state({T: T_init})
    s.run(20)                                        # compare here: rounding not yet amplified
    early[label] = s.grid("T").copy()
    t0 = time.perf_counter()
    s.run(180)                                       # timing (the flow becomes strongly nonlinear)
    runs[label] = (s, (time.perf_counter() - t0) / 180)
    print(f"{label}: {1e3 * runs[label][1]:.2f} ms per step")
print(runs["gpu"][0].explain())
# [/backends]

# [graph]
g = IVPSolver(plan, "RK443", 2e-4, backend="cuda", f_backend="fused")
g.set_state({T: T_init})
g.run_graph(20)                                      # capture one step, replay it
graph_early = g.grid("T").copy()
t0 = time.perf_counter()
g.run_graph(180)
print(f"gpu + CUDA graph: {1e3 * (time.perf_counter() - t0) / 180:.2f} ms per step")
# [/graph]

# [check]
Tc, Tg = early["cpu"], early["gpu"]
diff_gpu = np.abs(Tc - Tg).max() / np.abs(Tc).max()
diff_graph = np.abs(graph_early - Tg).max() / np.abs(Tg).max()
late = np.abs(runs["cpu"][0].grid("T") - runs["gpu"][0].grid("T")).max()
print(f"after 20 steps: relative max |T_cpu - T_gpu| = {diff_gpu:.1e}; graph vs plain = {diff_graph:.1e}")
print(f"after 200 steps: max |T_cpu - T_gpu| = {late:.1e} (rounding amplified by the instability)")
assert diff_gpu <= 1e-11 and diff_graph <= 1e-13
print("OK: CPU and GPU agree to rounding; the graph replay is the same computation")
# [/check]
