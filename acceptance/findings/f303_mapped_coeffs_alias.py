"""F303 (new, audit "state" 2026-10-20): the coefficients of a simulation on a mapped domain
(cn.MappedBox) are views of its state.

for v in sim.coeffs("T").values(): v *= 0 zeroed the simulation itself (the flat-domain
simulations return copies). Expected: the state is unchanged by any edit of a returned array.
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(303, "sim.coeffs() of a mapped-domain simulation does not alias its state")
box = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=lambda x: 0.05 * np.cos(x), top=1.0))
m = cn.Model(box, kappa=1.0)
m.scalar("T")
m.equations("dt(T) - kappa*lap(T) = 0")
m.at("walls", T=0)
sim = m.simulate((16, 8), scheme="RK222", dt=1e-3)
sim.initial(T=lambda x, z: np.sin(np.pi * z))
g0 = sim.grid("T").copy()
for v in sim.coeffs("T").values():
    v *= 0.0
check(np.array_equal(sim.grid("T"), g0), "the simulation's state changed when the arrays "
      "returned by coeffs() were edited")
passed("state unchanged")
