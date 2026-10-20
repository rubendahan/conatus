"""F353: time_parameter naming no parameter of the problem must be refused (audit inputs).

simulate(..., time_parameter="tt") (or IVPSolver(..., time_parameter="tt")) was kept and never
set: a forcing in the real time parameter stayed at its declared value for the whole run.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(353, "unknown time_parameter refused")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
r, exc = refused(m.simulate, 16, scheme="RK222", dt=0.01, time_parameter="tt")
check(r and "tt" in str(exc), "time_parameter='tt' accepted")
passed("refused with the name")
