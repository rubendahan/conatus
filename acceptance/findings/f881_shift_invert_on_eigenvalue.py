"""F881 (hunt of 0.3.0, 2026-11-10, L1): shift-invert with the target on an eigenvalue.
Neumann s u = u'' on [0, pi] (s = -n^2, n >= 0): target=0 raised a bare RuntimeError ("Factor is
exactly singular"), target=-1 with n_modes=3 returned one mode, the others dropped as infinite.
Oracle: the three values of {0, -1, -4, -9, ...} nearest the target.
"""
from _acceptance import check, passed, start
start(881, "shift-invert with the target on an eigenvalue")
import numpy as np  # noqa: E402
import conatus as cn  # noqa: E402

m = cn.Model(cn.Box(z=cn.walls(0, np.pi)), name="neu")
m.eigenvalue("s", growth="real")
m.scalar("u")
m.equations("s*u - dz(dz(u)) = 0")
m.at("walls", "dz(u) = 0")
exact = -np.arange(10.0) ** 2
for t in (0.0, -1.0, -4.0, 0.1):
    r = m.eigenvalues(N=24, method="shift-invert", target=t, n_modes=3)
    near = np.sort(exact[np.argsort(np.abs(exact - t), kind="stable")][:3])
    check(len(r) == 3 and np.allclose(np.sort(r.values.real), near, atol=1e-10),
          f"target {t}: {np.sort(r.values.real)} vs {near}")
passed("targets on 0, -1 and -4 return the three nearest eigenvalues")
