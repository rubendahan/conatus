"""F015 (important): max_growth() finds a maximum at the edge of the interval.

A = -10000 (1x1): G(t) = exp(-20000 t) is maximal at t = 0 with gain 1. 0.1.0 returned
(0.0, 0.1587): the first scan point's neighbourhood was refined without comparing the ends.
(The review's script imported conatus.nonmodal.energy, which does not exist; max_growth and
StateSpace are public in conatus.nonmodal.)
"""
import numpy as np
from _acceptance import check, passed, start

start(15, "max_growth compares the interval ends")
from conatus.nonmodal import StateSpace, max_growth  # noqa: E402

ss = StateSpace(np.array([[-10000.0]]), np.eye(1), None, 0, np.inf, 1.0, 1e-12)
g, t = max_growth(ss, (0.0, 10.0), xtol=1e-6)
check(abs(g - 1.0) < 1e-6 and t < 1e-5, f"(G_max, t_max) = ({g}, {t}), expected (1, 0)")
passed(f"G_max = {g:.6g} at t = {t:.2g}")
