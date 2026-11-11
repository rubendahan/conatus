"""F880 (hunt of 0.3.0, 2026-11-10, L2): dense EVP eigenvalues do not depend on the units.
s T = kappa T'' on [0, H], T = 0 at the walls, in SI units (kappa = 1.4e-7, H = 4e3 lost 3 digits,
kappa = 1e-6, H = 2.9e6 was off by a factor 6.6) and s u = c u'' with c from 1e-16 to 1e16 (c = 1e16
returned no finite mode). Oracle: s_n = -kappa (n pi / H)^2.
"""
from _acceptance import check, passed, start
start(880, "dense EVP invariant under a change of units")
import numpy as np  # noqa: E402
import conatus as cn  # noqa: E402


def solve(kappa, H):
    m = cn.Model(cn.Box(z=cn.walls(0, H)), name="si", kappa=kappa)
    m.eigenvalue("s", growth="real")
    m.scalar("T")
    m.equations("s*T - kappa*dz(dz(T)) = 0")
    m.at("walls", T=0)
    return m.eigenvalues(N=32)


ref = solve(1.0, 1.0)
for kappa, H in ((1.4e-7, 4e3), (1e3, 2e8), (1e-6, 2.9e6), (1e-16, np.pi), (1e16, np.pi)):
    r = solve(kappa, H)
    top = np.sort(r.values.real)[::-1][:4]
    oracle = -kappa * (np.arange(1, 5) * np.pi / H) ** 2
    check(np.allclose(top, oracle, rtol=1e-11, atol=0), f"kappa={kappa:g} H={H:g}: {top / oracle}")
    check(len(r) == len(ref) and r.n_infinite == ref.n_infinite,
          f"kappa={kappa:g} H={H:g}: {len(r)} finite modes, {len(ref)} nondimensional")
passed("five SI and extreme scalings match the closed form to 1e-11 with the full mode count")
