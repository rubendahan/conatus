"""F302 (new, audit "state" 2026-10-20): sim.coordinate("theta") on the sphere is the plan's
cached grid.

th = sim.coordinate("theta"); th -= pi/2 (colatitude to latitude, in place) changed every later
sim.coordinate("theta") of that simulation (and the coordinates its output files describe), with
no error. Expected: coordinate() returns the caller's own array.
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(302, "sim.coordinate() returns the caller's own array on the sphere")
m = cn.Model(cn.Sphere(1.0))
m.scalar("T")
m.equations("dt(T) - lap(T) = 0")
s = m.simulate((16, 8), scheme="RK222", dt=1e-3)
th = s.coordinate("theta")
ref = th.copy()
try:
    th -= np.pi / 2
except ValueError:
    pass                    # a read-only array refusing the edit is acceptable too
check(np.array_equal(s.coordinate("theta"), ref), "coordinate('theta') changed after editing "
      "the array an earlier call returned")
passed("theta unchanged")
