"""F033: dealias=nan / inf must be refused (they passed the d < 1 check in 0.1.0)."""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(33, "non-finite dealias factors are refused")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.equations("dt(u) = -u*u")
for bad in (np.nan, np.inf):
    was_refused, _ = refused(m.simulate, 16, scheme="RK222", dt=0.01, dealias=bad)
    check(was_refused, f"dealias={bad} accepted")
passed("nan and inf refused")
