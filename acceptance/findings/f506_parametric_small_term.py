"""F506 (external review of 0.1.1, F2): parametric assembly deleted a real parameter term.

s u + u = 0, s v - 1e-14 r v = 0: eigenvalues -1 and 1e-14 r. 0.1.1 dropped the r coefficient
matrix (1e-14 of the pencil) as "identically zero" and returned [-1, 0] at r = 1e14. Correct
behaviour: [-1, 1] at r = 1e14, or a refusal.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(506, "a small parameter term of a parametric pencil is kept")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), r=1.0)
m.scalar("u")
m.scalar("v")
m.eigenvalue("s", growth="real")
m.equations("s*u + u = 0", "s*v - 1e-14*r*v = 0")
P = m.parametric(8, over=("r",))
check("identically zero" not in P.explain(), "explain() still calls a numerical cutoff an identity")
out = []
for r in (1.0, 1e14):
    was_refused, res = refused(P.solve, r=r)
    check(not (was_refused and r == 1.0), "r = 1 (the nominal value) refused")
    if was_refused:
        out.append(f"r={r:g}: refused")
        continue
    got = np.sort(res.values.real)
    exact = np.sort([-1.0, 1e-14 * r])
    check(np.allclose(got, exact, rtol=1e-10, atol=1e-30), f"r={r:g}: {got} (exact {exact})")
    out.append(f"r={r:g}: {got}")
passed("; ".join(out))
