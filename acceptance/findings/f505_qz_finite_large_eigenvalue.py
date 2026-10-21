"""F505 (external review of 0.1.1, F7): dense QZ removed an exactly finite eigenvalue as infinite.

L = -diag(1, 1e13), M = I: both eigenvalues are finite (M is nonsingular). 0.1.1 returned [1] and
one infinite eigenvalue at 1e13 and 1e14. Correct behaviour: [1, big] for every big.
"""
import numpy as np
from _acceptance import check, passed, start

from conatus.expert import solve_dense

start(505, "a nonsingular mass matrix has no infinite eigenvalue")
for big in (1e10, 1e12, 1e13, 1e14, 1e16):
    L, M = -np.diag([1.0, big]), np.eye(2)
    exact = np.sort(np.linalg.eigvals(-L).real)          # M = I: the eigenvalues of -L
    r = solve_dense(L, M)
    check(r.n_infinite == 0 and r.values.size == 2
          and np.allclose(np.sort(r.values.real), exact, rtol=1e-12, atol=0),
          f"big={big:g}: values {r.values}, n_infinite {r.n_infinite} (exact {exact})")
passed("[1, big] for big = 1e10 .. 1e16")
