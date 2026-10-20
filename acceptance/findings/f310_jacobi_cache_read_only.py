"""F310 (new, review of the audit "state" 2026-10-20, F1): the cached Jacobi operators are
shared mutable matrices.

discretize.jacobi_family._J/_S/_D are lru_cached and returned the cached scipy.sparse matrix:
one in-place scaling (M *= c, M.data[...] = v) by any caller changed every operator built
afterwards with the same (N, a, b). Expected: the cached matrices are read-only.
"""
import numpy as np
from _acceptance import check, passed, start

from conatus.discretize import jacobi_family as jf

start(310, "cached Jacobi operators cannot be changed in place")
for fn in (jf._J, jf._S, jf._D):
    M = fn(10, 1.5, 0.5)
    before = M.toarray()
    try:
        M.data[0] = 99.0
        changed = True
    except ValueError:
        changed = False
    check(not changed, f"{fn.__name__}: the cached matrix accepted an in-place write")
    check(np.array_equal(fn(10, 1.5, 0.5).toarray(), before), f"{fn.__name__}: the cache changed")
passed("_J, _S, _D read-only")
