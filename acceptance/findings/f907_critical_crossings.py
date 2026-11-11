"""F907 (pass 4 H7): critical() reports every sign change of the growth in the bracket and the
direction of the one it returns.

Oracle: s u = u_xx + ((r - 1)(r - 3)(r - 5) + 1) u on (0, pi), u = 0 at the walls, with a
decoupled stable v: the growth of sin x is (r - 1)(r - 3)(r - 5), zero at r = 1, 3 and 5. On
(0, 6) the three crossings are listed, the returned one is one of them with its direction
(growth increasing at 1 and 5, decreasing at 3) and a CorrectnessWarning is raised (0.2.4: 3.0,
no word). Control: bracket (0, 2), one crossing at 1, increasing, no warning.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(907, "critical() lists every sign change and the direction of the returned one")
import conatus as cn  # noqa: E402
import conatus.expert as X  # noqa: E402
from conatus.errors import CorrectnessWarning  # noqa: E402

m = cn.Model(cn.Box(x=cn.walls(0, np.pi)), r=1.0)
m.scalar("u")
m.scalar("v")
m.eigenvalue("s", growth="real")
m.equations("s*u - dx(dx(u)) - ((r-1)*(r-3)*(r-5)+1)*u = 0", "s*v - dx(dx(v)) + 5*v = 0")
m.at("walls", "u=0", "v=0")
P = m.parametric(16, over=("r",))
for bracket, n_cross in (((0.0, 6.0), 3), ((0.0, 2.0), 1)):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = X.critical(P, "r", bracket=bracket, xtol=1e-9)
    warned = any(issubclass(w.category, CorrectnessWarning) and "changes sign" in str(w.message)
                 for w in caught)
    where = f"bracket {bracket}: r_c = {c.value}, crossings {c.crossings}, {c.direction}"
    check(len(c.crossings) == n_cross and warned == (n_cross > 1), where)
    k = int(np.argmin([abs(c.value - r) for r in (1.0, 3.0, 5.0)]))
    check(abs(c.value - (1.0, 3.0, 5.0)[k]) <= 1e-7, where)
    check(("increases" if k != 1 else "decreases") in c.direction, where)
    check(sum(x["returned"] for x in c.crossings) == 1, where)
    check(any("sign change" in line for line in c.log), where)
passed("three crossings listed on (0, 6) with the returned one's direction and a warning; one on (0, 2)")
