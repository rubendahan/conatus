"""F003 (blocking): the default dealiasing must be exact when explain() says "alias-free".

dt(u) = U*u*u with U = u = cos(7x) on 16 Fourier modes: the exact product cos^3(7x) =
3/4 cos 7x + 1/4 cos 21x, whose truncation has NO cos(3x) part. With 3/2 padding (24 points) the
cos 21x mode aliases onto cos 3x (0.25 in 0.1.0) because the degree count ignored the spatial
given field U. Expected: the cos(3x) coefficient of the right-hand side is 0 for the default
padding and for dealias=2, and the cos(7x) coefficient is the exact 3/4. The check does not
depend on the wording of explain() (review 2026-10-21, F14: it was gated on "alias-free").
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(3, "spatial given fields count in the dealiasing degree")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.given(U=lambda x: np.cos(7 * x))
m.equations("dt(u) = U*u*u")


def rhs_coefficients(padding):
    s = m.simulate(16, scheme="RK222", dt=0.01, dealias=padding)
    s.initial(u=lambda x: np.cos(7 * x))
    rhs = s.solver.F(s.solver.X, 0.0)
    return np.asarray(s.solver.system.scatter(rhs)[0])


# Independent oracle: cos^3(7x) = 3/4 cos 7x + 1/4 cos 21x; truncated to 16 modes (RealFourier
# layout [cos 0, sin 0, cos 1, sin 1, ...]) only cos 7x survives, with coefficient 3/4.
exact = np.zeros(16)
exact[14] = 0.75
for padding in (2, None):
    c = rhs_coefficients(padding)
    check(c.shape == exact.shape, f"dealias={padding}: {c.shape[0]} coefficients, expected 16 "
          "(the script's coefficient layout is stale)")
    err = np.abs(c - exact)
    check(err.max() < 1e-10, f"dealias={padding}: coefficient {int(err.argmax())} off by "
          f"{err.max():.3g} (cos 3x = {c[6]:.3g}: cos 21x aliased onto cos 3x)")
passed("default padding and dealias=2 both give the truncated cos^3(7x) = 3/4 cos 7x to 1e-10")
