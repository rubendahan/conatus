"""F521 (external review of 0.1.1, F4): verify() measures the residual of the current vector.

verify() printed the residual stored by track(): after the mode vector was replaced by a
coordinate vector, the right residual was still 3e-19 (PASS) while the backward error of the
new vector is 1e-3. Expected: the residual verify() reports equals the one computed here with
NumPy, and the check fails.
"""
import numpy as np
from _acceptance import check, passed, start

start(521, "verify() measures the residual of the current eigenvector")
import conatus as cn  # noqa: E402
import conatus.expert as X  # noqa: E402

m = cn.problems.native.orr_sommerfeld(Re=1e4, alpha=1.0, flow="poiseuille")
P = m.parametric(32, over=("Re", "alpha"))
r = P.solve(Re=1e4, alpha=1.0)
i = int(np.argmin(abs(r.values - (0.2375 + 0.0037j))))
mode = P.track(r.values[i], r.vectors[:, i], left=False, Re=1e4, alpha=1.0)
mode.x[:] = 0.0
mode.x[0] = 1.0
L, M = P.matrices(Re=1e4, alpha=1.0)
oracle = np.linalg.norm((L + mode.sigma * M) @ mode.x) / (
    (np.linalg.norm(L, 1) + abs(mode.sigma) * np.linalg.norm(M, 1)) * np.linalg.norm(mode.x))
right = next(c for c in X.verify(P, mode).checks if c.name == "right residual")
print(f"  oracle {oracle:.3e}, verify {right.value:.3e} {right.status}")
check(abs(right.value - oracle) <= 1e-8 * oracle, "verify() did not measure the current vector")
check(right.status == "FAIL", "a vector with backward error 1e-3 passed")
passed(f"right residual {right.value:.2e} = NumPy oracle, FAIL")
