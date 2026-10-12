"""Tutorial 2b -- the same Rayleigh--Benard IVP on a GPU (requires: gpu).

The model of tutorial 2a (native layer), at Ra = 1e5 and 128 x 64; only the backend changes.
The expert version is expert/t02b_rayleigh_benard_gpu.py.

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

import conatus as cn
# [/imports]


# [problem]
def rayleigh_benard(Ra):
    """The model of tutorial 2a (same equations, same conditions), at Rayleigh number Ra."""
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="Rayleigh-Benard 2D",
                 Ra=Ra, Pr=1)
    m.velocity("u")
    m.pressure("p")
    m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    return m


m = rayleigh_benard(Ra=1e5)
T_init = lambda x, z: -z + 1e-2 * np.cos(np.pi * x) * np.cos(np.pi * z)  # noqa: E731
# [/problem]

# [backends]
runs, early = {}, {}
for label, kw in [("cpu", dict(backend="cpu")),
                  ("gpu", dict(backend="cuda", f_backend="fused"))]:   # both choices explicit
    s = m.simulate(N=(128, 64), scheme="RK443", dt=2e-4, **kw)
    s.initial(T=T_init)
    s.run(steps=20)                                  # compare here: rounding not yet amplified
    early[label] = s.grid("T").copy()
    t0 = time.perf_counter()
    s.run(steps=180)                                 # timing (the flow becomes strongly nonlinear)
    runs[label] = (s, (time.perf_counter() - t0) / 180)
    print(f"{label}: {1e3 * runs[label][1]:.2f} ms per step")
print(runs["gpu"][0].explain())
# [/backends]

# [graph]
g = m.simulate(N=(128, 64), scheme="RK443", dt=2e-4, backend="cuda", f_backend="fused")
g.initial(T=T_init)
g.solver.run_graph(20)                               # expert: capture one step, replay it
graph_early = g.grid("T").copy()
t0 = time.perf_counter()
g.solver.run_graph(180)
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
