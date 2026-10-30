"""F736 (audit of 0.2.1, N11): a regular pencil is singular at its eigenvalues, so one probe proves nothing.

Oracle: the diagonal pencil with eigenvalues a and 1; a genuinely dependent pair of equations.
"""
from _acceptance import check, passed, start

start(736, "regularity probe colliding with an eigenvalue")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

from conatus.discretize.plan import RankError  # noqa: E402

a = .3719 + .6153j
for shift in (0.0, 1e-10):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi, complex=True)), a=a + shift)
    m.scalar("u")
    m.scalar("v")
    m.eigenvalue("s", growth="real")
    m.equations("s*u-a*u=0", "s*v-v=0")
    r = m.eigenvalues(N=8, k=0, spurious="none")
    v = np.asarray(r.values)
    v = v[np.argsort(v.real)]
    check(v.size == 2 and abs(v[0] - (a + shift)) < 1e-10 and abs(v[1] - 1) < 1e-10, f"values {v}")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi, complex=True)), a=a)
m.scalar("u")
m.scalar("v")
m.eigenvalue("s", growth="real")
m.equations("s*u-a*u+v=0", "2*s*u-2*a*u+2*v=0")
try:
    m.eigenvalues(N=8, k=0, spurious="none")
    check(False, "dependent equations accepted")
except RankError:
    pass
passed("collision solved, dependent equations refused")
