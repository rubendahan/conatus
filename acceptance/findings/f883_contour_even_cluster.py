"""F883 (hunt 5C of 0.3.0, 2026-11-10): contour count with a double eigenvalue near an edge.
A double eigenvalue at 0.123 - 0.999j in the box [-1, 1]^2 was counted 2 for 3 (the 2 pi turn of the
phase within one step aliased to 0), at 48 of 197 positions along the edge, with no warning.
Oracle: the eigenvalues of a diagonal pencil are given.
"""
from _acceptance import check, passed, start
start(883, "contour count: an even cluster near an edge")
import numpy as np  # noqa: E402
from conatus.contour import box, count  # noqa: E402
from conatus.errors import ConvergenceError  # noqa: E402

wrong, refused = 0, 0
for d in (1e-3, -1e-3):
    for x0 in np.linspace(-0.98, 0.98, 197):
        ev = np.array([x0 - 1e-7 + 1j * (-1 + d), x0 + 1e-7 + 1j * (-1 + d), 0.1 + 0.2j])
        truth = 1 + 2 * (d > 0)
        try:
            r = count((np.diag(-ev), np.eye(3)), box(re=(-1, 1), im=(-1, 1)))
        except ConvergenceError:
            refused += 1
            continue
        wrong += r.count != truth
check(wrong == 0, f"{wrong} wrong counts of 394")
passed(f"394 positions: none wrong, {refused} refused")
