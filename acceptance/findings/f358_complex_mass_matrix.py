"""F358: generalized_eigenvalues must keep a complex M (audit inputs, 2026-10-20).

M was cast to float: its imaginary part dropped with a ComplexWarning, and the eigenvalues of
J v = lam M v returned for another matrix. Shapes that differ were a bare SciPy ValueError.
"""
import warnings

import numpy as np
from _acceptance import check, passed, refused, start

from conatus.expert import generalized_eigenvalues

start(358, "complex M kept, shapes checked")
with warnings.catch_warnings():
    warnings.simplefilter("error")
    w = generalized_eigenvalues(np.eye(2), np.diag([1.0, 1j]))
check(np.allclose(sorted(w, key=lambda z: z.imag), [-1j, 1.0]), f"eigenvalues {w}")
r, _ = refused(generalized_eigenvalues, np.eye(3), np.eye(2))
check(r, "shape mismatch accepted")
passed("eigenvalues of the complex pencil; mismatch refused")
