"""F703 (audit of 0.2.1, N29): conjugation of a native complex unknown.

Oracle: u' = conj(u) gives Re u = e^t, Im u = e^-t for u(0) = 1 + i."""
from _acceptance import check, passed, start

start(703, "conjugation of a native complex unknown")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402



def error(dt, steps):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi, complex=True)))
    m.scalar("u")
    m.equations("dt(u) = conj(u)")
    s = m.simulate(8, scheme="RK443", dt=dt)
    s.initial(u=1 + 1j)
    s.run(steps=steps)
    return np.max(np.abs(s.grid("u") - (np.exp(.2) + 1j * np.exp(-.2))))


coarse, fine = error(.01, 20), error(.005, 40)
check(coarse < 1e-6, f"conj(u) wrong by {coarse:.2e}")
check(coarse / fine > 6, f"error ratio {coarse / fine:.2f} under dt / 2: third order expected")
passed(f"errors {coarse:.1e}, {fine:.1e}")
