"""F037: the mapped (terrain) solver must refuse dt = inf up front (accepted in 0.1.0).

0.1.0 accepted dt=inf and failed later with a singular-matrix error after NaN warnings.
Expected: a refusal that names dt, for inf, nan, 0 and a negative step.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(37, "non-finite time step refused on the mapped path")
geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0.0, top=1.0))
m = cn.Model(geo)
m.scalar("u")
m.equations("dt(u) - dz(dz(u)) = 0")
m.at("bottom", "u=0")
m.at("top", "u=0")
results = []
for bad in (np.inf, np.nan, 0.0, -0.1):
    def go(dt=bad):
        s = m.simulate((8, 8), scheme="RK222", dt=dt).initial(u=lambda x, z: np.sin(np.pi * z) + 0 * x)
        s.run(steps=1)
        return s
    was_refused, exc = refused(go)
    # the refusal must be about dt, not a later failure of the solve (inf made the system singular)
    results.append((bad, was_refused and "dt" in str(exc)))
check(all(r for _, r in results), "not refused as a bad dt: " + ", ".join(f"dt={b}" for b, r in results if not r))
passed("inf, nan, 0, negative refused")
