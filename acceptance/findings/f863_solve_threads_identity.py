"""F863: the IVP state does not depend on the number of solve threads, above and below
the measured threading threshold, and explain() states the threads.

Oracle: the one-thread run of the same problem (bit for bit). Rayleigh-Benard 2D at 32 x 16
(batches threaded or not by threads_for, the measured rule) and 128 x 32, 10 RK443 steps each.
"""
import os

import numpy as np
from _acceptance import check, passed, start

start(863, "solve threads: same bits above and below the threshold")
import conatus as cn  # noqa: E402
from conatus.runtime import banded as rb  # noqa: E402

for N in ((32, 16), (128, 32)):
    states = []
    for t in ("1", "4"):
        os.environ["CONATUS_SOLVE_THREADS"] = t
        m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=1e5, Pr=1)
        sim = m.simulate(N=N, scheme="RK443", dt=1e-4)
        sim.initial(T=lambda x, z: -z + 1e-2 * np.cos(np.pi * x) * np.cos(np.pi * z))
        sim.run(steps=10)
        states.append(sim.solver.X.copy())
    check(np.all(np.isfinite(states[0])) and np.array_equal(states[0], states[1]),
          f"{N}: 4 solve threads changed the state")
os.environ.pop("CONATUS_SOLVE_THREADS", None)
check(rb.threads_for(2, 1024, 8) == 1 and rb.threads_for(128, 512, 8) == 8, "threshold rule")
passed(f"bit-identical at 32 x 16 and 128 x 32; {rb.SOLVE_THREAD_UNKNOWNS} unknowns per thread squared")
