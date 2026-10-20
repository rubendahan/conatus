"""F367: a Revolve budget of 0 checkpoints must be refused (review of the inputs audit).

revolve_actions(n, 0) was the schedule of checkpoints=1 (one stored state at a time):
revolve_cost(n, 0)["peak_stored"] == 1, so the budget was not a bound. flowmap(T,
checkpoints=0) built that schedule. Every accepted budget bounds the peak.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn
from conatus.expert import revolve_actions, revolve_cost

start(367, "Revolve budget 0 refused; budgets bound the peak")
for f in (revolve_actions, revolve_cost):
    r, _ = refused(f, 10, 0)
    check(r, f"{f.__name__}(10, 0) accepted")
for n in (2, 9, 40):
    for c in (1, 2, 4):
        check(revolve_cost(n, c)["peak_stored"] <= c, f"peak above the budget at n={n}, c={c}")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
s = m.simulate(16, scheme="RK222", dt=0.01).initial(u=lambda x: np.sin(x))
r, _ = refused(s.flowmap, 0.1, checkpoints=0)
check(r, "flowmap(checkpoints=0) accepted")
s.flowmap(0.1, checkpoints=1)
passed("0 refused, 1 accepted, peak <= budget")
