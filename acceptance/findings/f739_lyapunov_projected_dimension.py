"""F739 (audit of 0.2.1, N19): k vectors need a subspace of dimension k.

Oracle: u_t = -u, three real coefficients, zero_mean removes one.
"""
from _acceptance import check, passed, start

start(739, "Lyapunov vectors beyond the projected dimension")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

from conatus.errors import OptionError  # noqa: E402


def sim():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations("dt(u)+u=0")
    s = m.simulate(4, scheme="RK443", dt=.01)
    s.initial(u=0.)
    return s


kw = dict(transient=0., align=.1, T=.2, reortho_every=2, inner="euclidean", n_blocks=2, seed=41,
          zero_mean=("u",))
try:
    sim().lyapunov(3, **kw)
    check(False, "three vectors in a two-dimensional subspace")
except OptionError:
    pass
r = sim().lyapunov(2, **kw)
check(np.allclose(r.exponents, -1, atol=1e-5), f"{r.exponents}")
passed("refused above the rank, rates -1 below")
