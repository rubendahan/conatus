"""F373: the refusals of the inputs audit do not catch valid uses (review of the audit).

NumPy scalars, 0-d arrays, Fraction and Decimal values, ints for floats, NumPy bools and 0 / 1
for switches, pathlib.Path for paths: accepted as the numbers and switches they are (several
were refused by the first version of the audit's checks: run(until=np.array(...)),
cn.Disk(np.array(1.0)), CFL(safety=np.array(0.5)), ncc_cutoff=np.array(...), complex=0).
"""
import decimal
import fractions
import pathlib

import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(373, "valid uses accepted")
cn.Disk(np.array(1.0))
cn.Disk(1)
cn.walls(np.float64(0.0), decimal.Decimal("1.5"))
cn.periodic(2.0, complex=np.bool_(True))
cn.periodic(2.0, complex=0)
cn.Model(cn.Box(x=cn.periodic(1.0)), Ra=fractions.Fraction(1, 2))
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
s = m.simulate(16, scheme="RK222", dt=np.float64(0.01)).initial(u=lambda x: np.sin(x))
s.run(until=np.array(0.02))
s.run(until=decimal.Decimal("0.04"))
check(s.iteration == 4, f"iteration {s.iteration} != 4")
s.flowmap(fractions.Fraction(1, 50))
p = s.save(pathlib.Path("state.npz"))
check(p.exists(), "Path checkpoint not written")
v = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="adv")
v.velocity("u")
v.equations("dt(u) = 0")
sv = v.simulate(16, scheme="RK222", dt=0.01).initial(u=lambda x: (2.0 + 0 * x,))
check(abs(sv.cfl("u", safety=np.array(0.5)) - 0.5 * (2 * np.pi / 16) / 2.0) < 1e-12, "CFL step")
passed("accepted")
