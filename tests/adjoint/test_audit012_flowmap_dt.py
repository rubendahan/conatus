"""Audit of 0.1.2, F02: a FlowMap keeps the step it was built with.

Oracle: implicit Euler (RK111) on u_t = -u gives X_N = (1 + dt)^-N X_0 (closed form); a map of
10 steps built at dt = 0.1 is (1.1)^-10 whatever the solver's step is afterwards.
"""
import numpy as np

import conatus as cn


def test_flowmap_call_trajectory_jvp_keep_the_construction_step():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations("dt(u)+u=0")
    s = m.simulate(N=8, scheme="RK111", dt=.1)
    s.initial(u=1)
    x = s.solver.X.copy()
    fm = s.flowmap(1.0)
    exact = 1.1 ** -10
    s.solver.set_dt(.2)
    i = int(np.argmax(np.abs(x)))
    assert abs(fm(x)[i] / x[i] - exact) < 1e-14
    assert abs(fm.trajectory(x)[-1][i] / x[i] - exact) < 1e-14
    assert abs(fm.jvp(x, x)[i] / x[i] - exact) < 1e-14
    assert s.solver.dt == 0.2              # the solver's step is its own again
    assert fm.dt == 0.1
