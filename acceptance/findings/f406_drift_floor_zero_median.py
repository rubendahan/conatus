"""F406 (invariance review 2026-10-20): the drift filter keeps a neutral mode of a spectrum whose
eigenvalues are mostly zero.

The drift floor was DRIFT_FLOOR x median |sigma| of the finer spectrum: 0 when most eigenvalues
are 0, so a neutral mode computed as rounding noise at both resolutions (1e-17 then 2e-17) got a
drift ratio of 1 and was filtered out. Expected: ratio at rounding level, and the same ratios
when every eigenvalue is written in another unit (x 1e4).
"""
import numpy as np
from _acceptance import check, passed, start

start(406, "drift floor robust to a zero median")
from conatus.solvers.evp import drift_ratios  # noqa: E402

lo = np.array([1e-17, 0.0, 1.0, 2.0])
hi = np.array([2e-17, 0, 0, 0, 0, 0, 1.0, 2.0])
d = drift_ratios(lo, hi)
check(d[0] < 1e-6, f"neutral rounding-noise mode has drift ratio {d[0]:.3g} (dropped at drift_tol 1e-6)")
check(np.allclose(drift_ratios(1e4 * lo, 1e4 * hi), d, rtol=1e-12, atol=0), "ratios depend on the unit")
passed(f"ratio {d[0]:.1e}, unit-free")
