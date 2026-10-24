"""F603 (audit of 0.1.2, F03): the finite/infinite split of dense QZ does not depend on the scale
of each equation; a removed pair that could be finite is reported with its modulus.

Oracle: det(L + s M) = (s - 1)(s - 1/eps) for L = -I, M = diag(1, eps) (closed form) and
scipy.linalg.eigvals."""
from _acceptance import check, passed, start

start(603, "dense QZ: same spectrum in equivalent equation units")
import numpy as np  # noqa: E402
from scipy.linalg import eigvals  # noqa: E402

from conatus.expert import solve_dense  # noqa: E402

out = []
for eps in (1e-13, 1e-17):
    L, M = -np.eye(2), np.diag([1.0, eps])
    D = np.diag([1.0, 1 / eps])
    rs = [solve_dense(L, M), solve_dense(D @ L, D @ M)]
    check(rs[0].values.size == rs[1].values.size,
          f"eps {eps:g}: {rs[0].values.size} vs {rs[1].values.size} finite roots after scaling")
    if eps == 1e-13:                 # a genuine large root, well above n eps: kept
        exact = np.sort(eigvals(-L, M).real)
        for r in rs:
            check(np.allclose(np.sort(r.values.real), exact, rtol=1e-10),
                  f"eps 1e-13: {r.values} vs {exact}")
    else:                            # below n eps: removed, but reported with its modulus
        for r in rs:
            amb = getattr(r, "infinite_ambiguous", np.zeros(0))
            check(amb.size == 1 and abs(amb[0] * eps - 1) < 1e-6,
                  f"eps 1e-17: removed root not reported (infinite_ambiguous {amb}, log {r.log})")
    out.append(rs[0].values.size)
passed(f"finite roots {out} in both units")
