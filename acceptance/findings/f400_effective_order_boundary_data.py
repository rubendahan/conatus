"""F400 (audit invariance, the class of finding 17): the stated global order includes the
boundary-data convention.

u_t = u_yy / 2 between walls moving as cos(t) (exact solution e^{-y} cos(t - y)), RK443 to t = 1
at dt = 1/80 and 1/160. With bc_time="step" (the Dedalus convention) the observed order is about
1 and with bc_time="stage" about 2, while solver.effective_order and explain()'s scheme line said
"global order 3". Expected: effective_order is the observed order (within 0.3) and explain()
gives it.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(400, "effective order with time-dependent boundary data")


def run(bc_time, n):
    m = cn.Model(cn.Box(y=cn.walls(0, 5)), name="moving walls")
    m.scalar("u")
    m.equations("dt(u) - dy(dy(u))/2 = 0")
    m.at("bottom", u="cos(t)")
    m.at("top", u=lambda t: np.exp(-5) * np.cos(t - 5))
    s = m.simulate(N=40, scheme="RK443", dt=1.0 / n, dealias=1, bc_time=bc_time)
    s.initial(u=lambda y: np.exp(-y) * np.cos(-y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.run(steps=n)
    y = s.coordinate("y")
    return float(np.max(np.abs(s.grid("u") - np.exp(-y) * np.cos(s.t - y)))), s.solver


rows = []
for bc in ("step", "stage"):
    (e1, _), (e2, solver) = run(bc, 80), run(bc, 160)
    p = float(np.log2(e1 / e2))
    stated = getattr(solver, "effective_order", None)
    rows.append(f"{bc}: observed {p:.2f}, stated {stated}")
    check(stated is not None and abs(p - stated) < 0.3, rows[-1])
    check(f"global order {stated}" in solver.explain(), f"{bc}: explain() does not give order {stated}")
passed("; ".join(rows))
