"""F602 (audit of 0.1.2, F02): a FlowMap keeps the step it was built with.

Oracle: implicit Euler on u_t = -u, X_N = (1 + dt)^-N X_0; 10 steps at dt = 0.1: 1.1^-10."""
from _acceptance import check, passed, start

start(602, "FlowMap call/trajectory/jvp unchanged by solver.set_dt")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.equations("dt(u)+u=0")
s = m.simulate(N=8, scheme="RK111", dt=.1)
s.initial(u=1)
x = s.solver.X.copy()
i = int(np.argmax(np.abs(x)))
fm = s.flowmap(1.0)
s.solver.set_dt(.2)
exact = 1.1 ** -10
vals = {"call": fm(x)[i] / x[i], "trajectory": fm.trajectory(x)[-1][i] / x[i],
        "jvp": fm.jvp(x, x)[i] / x[i]}
for k, v in vals.items():
    check(abs(v - exact) < 1e-14, f"{k} gives {v!r}, expected 1.1^-10 = {exact!r}")
check(s.solver.dt == .2, f"solver dt left at {s.solver.dt}")
passed(f"all three = {exact:.16g} after set_dt(0.2)")
