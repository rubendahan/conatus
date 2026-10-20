"""F360: run(until=nan/inf/True) and flowmap(T=nan) must be typed refusals (audit inputs).

until=nan and inf failed in int(round(...)) with a bare ValueError/OverflowError, until=True ran
to t = 1, flowmap(T=nan) was a bare ValueError.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(360, "non-finite or non-numeric horizons refused")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
s = m.simulate(16, scheme="RK222", dt=0.01).initial(u=lambda x: np.sin(x))
for bad in (float("nan"), float("inf"), True, "0.02"):
    r, _ = refused(s.run, until=bad)
    check(r, f"run(until={bad!r}) accepted")
for bad in (float("nan"), float("inf"), "0.1", -0.1):
    r, _ = refused(s.flowmap, bad)
    check(r, f"flowmap({bad!r}) accepted")
check(s.iteration == 0, "a refused call advanced the run")
passed("refused before any step")
