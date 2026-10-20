"""F013 (important): finite/infinite eigenvalue filtering is invariant under common scaling.

(A, B) = (I, 1e-12 I) and (1e12 I, I) have the same eigenvalues 1e12 (twice). 0.1.0 returned []
for the first and [1e12, 1e12] for the second.
"""
import numpy as np
from _acceptance import check, passed, start

start(13, "generalized eigenvalues do not depend on the equation's scale")
from conatus.bifurcation.equilibria import generalized_eigenvalues  # noqa: E402

I2 = np.eye(2)
results = []
for scale in (1.0, 1e-6, 1e6):
    small = np.sort_complex(np.asarray(generalized_eigenvalues(scale * I2, scale * 1e-12 * I2)))
    big = np.sort_complex(np.asarray(generalized_eigenvalues(scale * 1e12 * I2, scale * I2)))
    results.append((scale, small, big))
    check(small.shape == big.shape, f"scale {scale}: {small} vs {big}")
    check(np.allclose(small, big, rtol=1e-10), f"scale {scale}: {small} vs {big}")
passed(f"both pairs give {results[0][2]}")
