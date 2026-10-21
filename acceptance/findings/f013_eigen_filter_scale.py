"""F013 (important): finite/infinite eigenvalue filtering is invariant under common scaling.

(A, B) = (I, 1e-12 I) and (1e12 I, I) have the same eigenvalues 1e12 (twice). 0.1.0 returned []
for the first and [1e12, 1e12] for the second. Both are compared with the known spectrum, not
with each other (review 2026-10-21, F14: an implementation that always returns [] passed the
pairwise comparison). The pencil (diag(1, 2), diag(1, 0)) has one finite eigenvalue 1 and one
infinite eigenvalue, which must be removed at every scale.
"""
import numpy as np
from _acceptance import check, passed, start

start(13, "generalized eigenvalues do not depend on the equation's scale")
from conatus.bifurcation.equilibria import generalized_eigenvalues  # noqa: E402

I2 = np.eye(2)
EXACT = np.array([1e12, 1e12])                     # analytic spectrum of both pairs
for scale in (1.0, 1e-6, 1e6):
    small = np.sort_complex(np.asarray(generalized_eigenvalues(scale * I2, scale * 1e-12 * I2)))
    big = np.sort_complex(np.asarray(generalized_eigenvalues(scale * 1e12 * I2, scale * I2)))
    for name, got in (("(I, 1e-12 I)", small), ("(1e12 I, I)", big)):
        check(got.shape == EXACT.shape and np.allclose(got, EXACT, rtol=1e-10, atol=0),
              f"scale {scale}: {name} gives {got}, expected [1e12, 1e12]")
    one = np.asarray(generalized_eigenvalues(scale * np.diag([1.0, 2.0]), scale * np.diag([1.0, 0.0])))
    check(one.shape == (1,) and abs(one[0] - 1.0) < 1e-12,
          f"scale {scale}: (diag(1, 2), diag(1, 0)) gives {one}, expected [1] (one infinite removed)")
passed("both pairs give [1e12, 1e12] and the singular pencil [1] at scales 1, 1e-6, 1e6")
