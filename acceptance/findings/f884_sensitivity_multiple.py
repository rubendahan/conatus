"""F884 (hunt 5D of 0.3.0, 2026-11-10, Q3): sensitivity at a double eigenvalue.
L(p) = -(I + p B), M = I, B = [[1, 2], [2, -1]]: the branches through 1 have the slopes +-sqrt(5)
(Rellich); the first-order formula returned 1 and -1 with no warning. Oracle: eig(B).
"""
from _acceptance import check, passed, start
start(884, "eigenvalue sensitivity at a multiple eigenvalue")
import numpy as np  # noqa: E402
from conatus.errors import SpectralError  # noqa: E402
from conatus.solvers.evp import cluster_sensitivity, sensitivity, solve_dense  # noqa: E402

B = np.array([[1.0, 2.0], [2.0, -1.0]])
res = solve_dense(-np.eye(2), np.eye(2), left=True)
refused = 0
for i in range(2):
    try:
        sensitivity(res, i, np.eye(2), -B)
    except SpectralError:
        refused += 1
check(refused == 2, f"{2 - refused} slope(s) returned for a double eigenvalue")
slopes = np.sort(cluster_sensitivity(res, [0, 1], np.eye(2), -B).real)
check(np.allclose(slopes, [-np.sqrt(5), np.sqrt(5)], rtol=1e-12), f"slopes {slopes}")
passed("refused per mode; the branch slopes are +-sqrt(5)")
