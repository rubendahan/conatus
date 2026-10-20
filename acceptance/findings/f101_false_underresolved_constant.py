"""F101 (new, critic 2026-10-16): a constant field is not "under-resolved".

du/dt + u = 0 from u = 1 on 4 Fourier modes: the state is the constant mode only (exactly
represented), yet 0.1.0 warns "under-resolved ... tail 1.0e+00 (N = 4)" and explain() records it.
Expected: no under-resolution warning (N = 8 gives none).
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(101, "no false under-resolution warning for an exactly resolved field")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.equations("dt(u) + u = 0")
s = m.simulate(4, scheme="RK222", dt=0.1).initial(u=1)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    s.run(steps=3)
bad = [str(w.message)[:160] for w in caught if "under-resolved" in str(w.message)]
check(not bad, f"warned: {bad[0] if bad else ''}")
passed("no warning")
