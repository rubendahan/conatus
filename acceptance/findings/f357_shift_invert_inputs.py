"""F357: shift-invert must refuse an n_modes it cannot return and a non-finite target.

n_modes larger than size - 2 was clipped to size - 2 while the log said the requested count;
target=nan factorised a NaN matrix ("Factor is exactly singular"), target="0.2" was a bare
UFuncTypeError (audit inputs, 2026-10-20).
"""
import numpy as np
from _acceptance import check, passed, refused, start

from conatus.expert import solve_sparse

start(357, "shift-invert count and target checked")
L, M = np.diag(np.arange(1.0, 7.0)), np.eye(6)
for kw in (dict(target=0.5, n_modes=5), dict(target=float("nan"), n_modes=2),
           dict(target="0.2", n_modes=2), dict(target=0.5, n_modes=2.5)):
    r, _ = refused(solve_sparse, L, M, **kw)
    check(r, f"solve_sparse({kw}) accepted")
res = solve_sparse(L, M, -1.1, n_modes=2)
check(np.allclose(sorted(res.values.real), [-2.0, -1.0]), f"values {res.values}")
passed("refused; a valid call returns the two nearest eigenvalues")
