"""F705 (audit of 0.2.1, N16): the first adaptive step obeys dt_max and dt_min.

Oracle: u' = 1 from 0 gives u = t; the first step is the clamped one."""
from _acceptance import check, passed, start

start(705, "the first adaptive step obeys dt_max and dt_min")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402
from conatus.timestep import EmbeddedError  # noqa: E402

for dt0, want in ((1., .1), (.001, .01)):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations("dt(u) = 1")
    s = m.simulate(8, scheme="RK443", dt=dt0)
    s.initial(u=0.)
    EmbeddedError(s.solver, adapt=True, tol=1e-8, safety=.9, dt_min=.01, dt_max=.1).step()
    check(abs(s.t - want) < 1e-12, f"first step {s.t} from dt = {dt0}, bounds [.01, .1]")
passed("bounds applied to the first step")
