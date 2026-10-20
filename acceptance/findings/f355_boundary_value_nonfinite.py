"""F355: m.at(boundary, u=nan / inf / True / None) must be refused (audit inputs).

The value reached the right-hand side of the boundary row as it was: a NaN boundary row
compiled and simulated without a word.
"""
from _acceptance import check, passed, refused, start

import conatus as cn

start(355, "non-finite or non-numeric boundary values refused")
for bad in (float("nan"), float("inf"), True, None):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)))
    m.scalar("u")
    m.equations("dt(u) - dz(dz(u)) = 0")
    r, _ = refused(m.at, "bottom", u=bad)
    check(r, f"at('bottom', u={bad!r}) accepted")
passed("nan, inf, True, None refused at the declaration")
