"""F030: initial(u=nan) must be refused for a scalar, as for an array.

0.1.0 accepted initial(u=np.nan) (the array form is checked, the scalar form was not).
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(30, "non-finite scalar initial data is refused")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.equations("dt(u) = -u")
s = m.simulate(8, scheme="RK222", dt=0.01)
for bad in (np.nan, np.inf):
    was_refused, _ = refused(s.initial, u=bad)
    check(was_refused, f"initial(u={bad}) accepted")
passed("nan and inf refused")
