"""F351: a switch given as text ("no") must be refused, never read as True (audit inputs).

periodic(L, complex="no") built a complex axis, simulate(..., reduced="no"/complexify="no")
switched the options on, series(f, logy="no") drew a log axis, eigenvalues(left="no") reached
LAPACK as a bare error.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(351, "switches are True or False, never the truth value of text")
r, _ = refused(cn.periodic, 2.0, complex="no")
check(r, "periodic(complex='no') accepted")
r, _ = refused(cn.series, lambda s: 0.0, logy="no")
check(r, "series(logy='no') accepted")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
for opt in ("complexify", "decouple", "reduced"):
    r, _ = refused(m.simulate, 16, scheme="RK222", dt=0.01, **{opt: "no"})
    check(r, f"simulate({opt}='no') accepted")
passed("text switches refused")
