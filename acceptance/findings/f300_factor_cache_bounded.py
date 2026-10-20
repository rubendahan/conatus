"""F300 (new, audit "state" 2026-10-20): the factorizations of a solver are a bounded cache.

Every (a, b) factorization of a M + b L was kept for the life of the solver. An adaptive step
(EmbeddedError(adapt=True): a new dt at every accepted or rejected step) adds new keys at every
step: 406 factor sets were held after 200 steps of a 4-mode problem, so a long adaptive run grows
until it runs out of memory. Expected: the number held stays bounded and the solution is right.
"""
from _acceptance import check, passed, start

import conatus as cn
from conatus.timestep import EmbeddedError

start(300, "an adaptive run keeps a bounded number of factorizations")
m = cn.Model(cn.Box(x=cn.walls(-1, 1)))
m.scalar("u")
m.equations("dt(u) = -u*u")
s = m.simulate(4, scheme="RK222", dt=.01).initial(u=1)
e = EmbeddedError(s.solver, adapt=True, tol=1e-8, safety=.9)
held = []
for _ in range(200):
    e.step()
    held.append(s.solver.system.n_factorizations)
err = abs(float(s.grid("u")[0]) - 1 / (1 + s.t))
check(err < 1e-6, f"u(t) error {err:.1e}")
check(held[-1] <= 32 and held[-1] <= held[49], f"factorizations held after 50 / 200 steps: "
      f"{held[49]} / {held[-1]} (grows with the number of steps)")
passed(f"{held[-1]} factorizations held after 200 adaptive steps")
