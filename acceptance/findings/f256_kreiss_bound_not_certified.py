"""F256 (audit "assurances", 2026-10-20): kreiss_constant does not certify a local maximum.

Two non-normal blocks; the scan grid sees the first (Re z |R| = 250 at z = 1) and misses the
second (2500 at Im z = 10). 0.1.0 printed "K = sup_(Re z > 0) ... = 250; hence 250 <=
sup_t ||e^(tA)|| <= 2718.29 (e k K)" while sup_t ||e^(tA)|| = 3679.
"""
import numpy as np
from _acceptance import check, passed, skipped, start

start(256, "kreiss_constant upper growth bound from a local maximum")
import conatus.nonmodal as nonmodal  # noqa: E402

if not hasattr(nonmodal, "kreiss_constant"):      # pseudospectra are cut from the public build
    skipped("kreiss_constant is not shipped in this build")
from conatus.nonmodal import StateSpace, kreiss_constant  # noqa: E402

lam = -0.01 + 10j
A = np.zeros((4, 4), complex)
A[:2, :2] = [[-1, 1000], [0, -1]]
A[2:, 2:] = [[lam, 100], [0, lam]]
ss = StateSpace(A, np.eye(4), None, 0, np.inf, 1.0, 1e-8)
kr = kreiss_constant(ss, np.logspace(-1, 1, 9), np.linspace(-3, 3, 13), xtol=1e-8)
txt = kr.explain()
# independent value (review 2026-10-21, F14): sup_t ||e^(tA)||_2 from scipy expm on a t grid;
# the lower bound K <= sup_t ||e^(tA)|| (Kreiss) must hold
from scipy.linalg import expm  # noqa: E402
# the second block peaks at t = 1 / 0.01 = 100: 100 t e^(-0.01 t) = 1e4 / e
growth = max(np.linalg.norm(expm(t * A), 2) for t in np.linspace(0, 300, 3001))
check(3600 < growth < 3700, f"oracle: sup_t ||e^(tA)|| = {growth}, expected about 3679")
check(kr.lower_bound_growth <= growth * (1 + 1e-6), f"lower bound {kr.lower_bound_growth} above the "
      f"true growth {growth}")
claims = "K = sup" in txt or (kr.upper_bound_growth < 3678 and "only if" not in txt)
check(not claims, f"certified bound below the true growth 3679: {txt}")
passed(f"K >= {kr.K:.4g}, upper bound stated as conditional")
