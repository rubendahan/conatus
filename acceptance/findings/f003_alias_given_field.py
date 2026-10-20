"""F003 (blocking): the default dealiasing must be exact when explain() says "alias-free".

dt(u) = U*u*u with U = u = cos(7x) on 16 Fourier modes: the exact product cos^3(7x) =
3/4 cos 7x + 1/4 cos 21x, whose truncation has NO cos(3x) part. With 3/2 padding (24 points) the
cos 21x mode aliases onto cos 3x (0.25 in 0.1.0) because the degree count ignored the spatial
given field U. Expected: the cos(3x) coefficient of the right-hand side is 0 for the default
padding whenever explain() claims "exact (alias-free)", and for dealias=2 always.
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(3, "spatial given fields count in the dealiasing degree")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.given(U=lambda x: np.cos(7 * x))
m.equations("dt(u) = U*u*u")


def cos_coefficients(padding):
    s = m.simulate(16, scheme="RK222", dt=0.01, dealias=padding)
    s.initial(u=lambda x: np.cos(7 * x))
    rhs = s.solver.F(s.solver.X, 0.0)
    c = np.asarray(s.solver.system.scatter(rhs)[0])
    return c[6], c[14]            # RealFourier layout [cos 0, sin 0, cos 1, sin 1, ...]


claims = "alias-free" in m.compile(16).explain(full=True)
c3_default, c7_default = cos_coefficients(None)
c3_two, c7_two = cos_coefficients(2)
check(abs(c7_two - 0.75) < 1e-10, f"layout check: cos 7x coefficient {c7_two}, expected 0.75 "
      "(the script's coefficient layout is stale)")
check(abs(c3_two) < 1e-10, f"dealias=2: cos 3x coefficient {c3_two}, expected 0")
if claims:
    check(abs(c3_default) < 1e-10, f"default padding: cos 3x coefficient {c3_default:.3g} "
          "(aliased cos 21x) while explain() says alias-free")
passed(f"cos 3x: default {c3_default:.2g}, dealias=2 {c3_two:.2g}; alias-free claimed: {claims}")
