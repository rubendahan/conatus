"""F405 (audit invariance): a trajectory's Lyapunov exponents do not depend on its place in the
batch of starts.

Kuramoto-Sivashinsky (L = 22, N = 24, RK443, dt = 0.05), two starts, 3 exponents over T = 4:
the starts in the order (a, b), (b, a), and b alone. The initial tangent vectors were one
stream of B*k draws of the seed, so b got other vectors (and other finite-time exponents, up to
0.2 apart) in each case.
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(405, "Lyapunov exponents independent of the batch order")
L = 22.0
m = cn.Model(cn.Box(x=cn.periodic(L)), name="KS")
m.scalar("u")
m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
sim = m.simulate(N=24, scheme="RK443", dt=0.05)


def seed(c):
    return {"u": lambda x: 0.3 * sum(c[j] * np.cos(2 * np.pi * (j + 1) * x / L + j) for j in range(3))}


a, b = seed([0.5, -1.0, 0.7]), seed([-0.2, 0.4, 1.1])
kw = dict(transient=2.0, align=2.0, T=4.0, reortho_every=4, inner="euclidean", n_blocks=2, seed=0,
          zero_mean="u")
ab = sim.lyapunov(3, starts=[a, b], **kw).per_trajectory
ba = sim.lyapunov(3, starts=[b, a], **kw).per_trajectory
alone = sim.lyapunov(3, starts=[b], **kw).per_trajectory
d1 = float(np.abs(ab[::-1] - ba).max())
d2 = float(np.abs(ab[1] - alone[0]).max())
check(d1 < 1e-11, f"reordered starts change the exponents by {d1:.2e}")
check(d2 < 1e-11, f"b alone vs b second in the batch: exponents differ by {d2:.2e}")
passed(f"b: {np.round(alone[0], 4)} in every place")
