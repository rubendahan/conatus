"""F505 (external review of 0.1.1, F7): dense QZ removed an exactly finite eigenvalue as infinite.

L = -diag(1, 1e13), M = I: both eigenvalues are finite (M is nonsingular). 0.1.1 returned [1] and
one infinite eigenvalue at 1e13 and 1e14. Correct behaviour: [1, big] for big up to 1e14. At
big = 1e16 the second row of the equilibrated pencil is (-1 | 1e-16), below n eps: the same pencil
as L = -I, M = diag(1, 1e-16), whose mass row cannot be told from zero. The split must not depend
on the equation scale (audit of 0.1.2, F03), so 1e16 is removed in both forms but reported with its
modulus in ``infinite_ambiguous``, never dropped silently.
"""
import numpy as np
from _acceptance import check, passed, start

from conatus.expert import solve_dense

start(505, "a nonsingular mass matrix has no infinite eigenvalue")
for big in (1e10, 1e12, 1e13, 1e14):
    L, M = -np.diag([1.0, big]), np.eye(2)
    exact = np.sort(np.linalg.eigvals(-L).real)          # M = I: the eigenvalues of -L
    r = solve_dense(L, M)
    check(r.n_infinite == 0 and r.values.size == 2
          and np.allclose(np.sort(r.values.real), exact, rtol=1e-12, atol=0),
          f"big={big:g}: values {r.values}, n_infinite {r.n_infinite} (exact {exact})")
r = solve_dense(-np.diag([1.0, 1e16]), np.eye(2))
amb = getattr(r, "infinite_ambiguous", np.zeros(0))
kept = r.values.size == 2 and np.allclose(np.sort(r.values.real), [1.0, 1e16], rtol=1e-12)
flagged = r.values.size == 1 and amb.size == 1 and abs(amb[0] / 1e16 - 1) < 1e-10
check(kept or flagged, f"big=1e16: neither kept nor reported (values {r.values}, "
                       f"infinite_ambiguous {amb})")
passed("[1, big] for big = 1e10 .. 1e14; 1e16 " + ("kept" if kept else "reported as ambiguous"))
