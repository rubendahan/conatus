"""Tutorial 15 -- covariant Lyapunov vectors of Kuramoto--Sivashinsky, L = 22: exponents, the
covariance check, tangencies and the eight physical modes.

u_t + u u_x + u_xx + u_xxxx = 0 on (0, 22), periodic, zero mean.
Expected values (checked at the end of the script):
  leading exponents (0.048, 0, 0, -0.003, -0.189, -0.256, -0.290, -0.310, -1.963, -1.967, ...)
      Cvitanovic, Davidchack & Siminos (2010), SIAM J. Appl. Dyn. Syst. 9, p. 12 (zero mean)
  eight "physical" modes: the first eight covariant vectors are entangled (frequent violations of
  the ordering of their local exponents, near-tangencies), the others are isolated (in pairs)
      Ding, Chate, Cvitanovic, Siminos & Takeuchi (2016), Phys. Rev. Lett. 117, 024101
A short run (4 trajectories, 200 time units each): the exponents carry their standard errors and
are compared within them; the long runs of the validation table go to 2500 time units.
Run: python t15_ks_covariant.py   (about two minutes of CPU)
"""
# [imports]
import numpy as np

import conatus as cn
# [/imports]

# [model]
L = 22.0
m = cn.Model(cn.Box(x=cn.periodic(L)), name="Kuramoto-Sivashinsky L=22")
m.scalar("u")
m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
sim = m.simulate(N=32, scheme="RK443", dt=0.05)   # 32 modes, as in the references


def start(c):
    """A smooth zero-mean initial state from four random modes."""
    return {"u": lambda x: 0.3 * sum(c[j, 0] * np.cos(2 * np.pi * (j + 1) * x / L)
                                     + c[j, 1] * np.sin(2 * np.pi * (j + 1) * x / L)
                                     for j in range(4))}


rng = np.random.default_rng(7)
starts = [start(rng.standard_normal((4, 2))) for _ in range(4)]
# [/model]

# [lyapunov]
res = sim.lyapunov(12, transient=100.0, align=50.0, T=200.0, reortho_every=10,
                   inner="euclidean", n_blocks=4, seed=1, starts=starts, zero_mean="u",
                   covariant=True, clv_window=40.0, clv_backward=40.0, clv_checkpoints=8)
print(res.explain())
# [/lyapunov]

# [domination]
dom = res.domination(tau=5.0)
print(dom.explain())
theta8 = res.angles(8)                     # span(gamma_1..gamma_8) vs the rest, every sample
theta4 = res.angles(4)
print(f"smallest angle at the split 8 | 9: {theta8.min():.3f} rad; at 4 | 5: "
      f"{theta4.min():.4f} rad ({np.mean(theta4 < 0.1):.0%} of the samples below 0.1 rad)")
# [/domination]

# [checks]
cds = np.array([0.048, 0.0, 0.0, -0.003, -0.189, -0.256, -0.290, -0.310, -1.963, -1.967])
dev = np.abs(res.exponents[:10] - cds)
assert np.all(dev < 4 * res.stderr[:10] + 0.01), (res.exponents[:10], res.stderr[:10])
# J gamma_i(t) is parallel to gamma_i(t + dt_interval), to the rounding floor of each vector
assert np.all(res.clv_covariance_residual < 100 * res.clv_covariance_floor)
assert dom.n_physical == 8 and dom.violation(8) == 0.0
assert res.clv_split_convergence[7] < 1e-6            # the split 8 | 9 itself has converged
assert theta8.min() > 10 * theta4.min()               # isolated split versus tangencies
print("checks passed: exponents within their errors of CDS 2010, eight physical modes")
# [/checks]
