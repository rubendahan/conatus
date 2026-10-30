"""F747 (audit of 0.2.1, N31): the refined time lies in the segment, and its state and defect are of that time.

Oracle: the RK222 stability polynomial R(h) = (1 + (2g-1) h) / (1 + g h)^2, g = 1 - 1/sqrt(2), with brentq.
"""
from _acceptance import check, passed, start

start(747, "Poincare refinement stays in its segment")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

from scipy.optimize import brentq  # noqa: E402

from conatus.manifolds import PoincareSection, poincare_crossings  # noqa: E402
from conatus.manifolds.trajectory import SolverIntegrator  # noqa: E402

g = 1 - 1 / np.sqrt(2)
R = lambda h: (1 + (2 * g - 1) * h) / (1 + g * h) ** 2  # noqa: E731
for step in (.5, 1., 2.):
    m = cn.Model(cn.Box(x=cn.periodic(1.)))
    m.scalar("y")
    m.equations("dt(y) = -y")
    sim = m.simulate(N=4, scheme="RK222", dt=step)
    sim.initial(y=1.)
    tr = SolverIntegrator(sim.solver, dt=step).integrate(sim.solver.X, step)
    i = int(np.argmax(np.abs(sim.solver.X)))
    level = 1 + (R(step) - 1) * .999999
    sec = PoincareSection(lambda x, level=level, i=i: float(x[i] - level), direction=-1, name="decay")
    out = poincare_crossings(tr, sec, xtol=1e-13, refine=True)
    check(out.t.size == 1 and 0 <= out.t[0] <= step, f"dt {step}: {out.t}")
    root = brentq(lambda h, level=level: R(h) - level, 1e-9, step, xtol=1e-14)
    check(abs(out.t[0] - root) < 1e-5 * step, f"dt {step}: {out.t[0]} against {root}")
    check(abs(out.X[0, i] - R(out.t[0])) < 1e-11, "state is not of the returned time")
    d = np.linalg.norm(tr.restep(out.t[0]) - tr(out.t[0]))
    check(abs(out.interp_defect[0] - d) < 1e-12, f"defect {out.interp_defect[0]} against {d}")
passed("time, state and defect agree")
