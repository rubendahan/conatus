"""F733 (audit of 0.2.1, F03): QZ split independent of the equation scale, and the choices kept.

The pencil L = -I, M = diag(1, 1e-17) and a zero row of M with rounding noise are the same matrices in double precision, so one rule covers both: removed as infinite and reported with its modulus.
Oracle: exact eigenvalues of diagonal pencils.
"""
from _acceptance import check, passed, start

start(733, "QZ split: scale invariance and the rank choices")
import numpy as np  # noqa: E402


from conatus.expert import solve_dense  # noqa: E402

r1 = solve_dense(-np.eye(2), np.diag([1.0, 1e-17]))
r2 = solve_dense(np.array([[-1.0, 0], [0, -1e17]]), np.diag([1.0, 1.0]))
for r in (r1, r2):
    check(r.values.size == 1 and r.infinite_ambiguous.size == 1
          and abs(r.infinite_ambiguous[0] / 1e17 - 1) < 1e-8, "1e17 is removed and reported")
r = solve_dense(-np.diag([1.0, 1e13]), np.eye(2))
check(r.n_infinite == 0 and np.allclose(np.sort(r.values.real), [1, 1e13], rtol=1e-10, atol=0), "1e13 finite")
r = solve_dense(-np.eye(2), np.array([[1.0, 0], [1e-17, 1e-17]]))
check(r.values.size == 1 and abs(r.values[0] - 1) < 1e-10, "noise row: its pair is not returned finite")
r = solve_dense(-np.eye(2), np.diag([1.0, 0.0]))
check(r.values.size == 1 and r.n_infinite == 1, "L = -I, M = diag(1, 0): one finite, one infinite")
passed("the rank choices hold")
