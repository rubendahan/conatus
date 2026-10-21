"""F254 (audit "assurances", 2026-10-20): hinf_norm never calls a scan-end maximum the sup.

lambda = -0.01 - i: ||H||inf = 100 at omega = 1. With omegas in [-0.5, 0.5] 0.1.0 printed
"||H||inf = sup_omega sigma_1(R) = 1.9996 at omega = 0.5".
"""
import warnings

import numpy as np
from _acceptance import check, need, passed, start

start(254, "hinf_norm maximum at an end of the scan")
need("conatus.nonmodal.resolvent")       # SKIP where the resolvent family is cut (public build)
from conatus.nonmodal import StateSpace  # noqa: E402
from conatus.expert import hinf_norm  # noqa: E402

ss = StateSpace(np.array([[-0.01 - 1.0j]]), np.eye(1), None, 0, np.inf, 1.0, 1e-8)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    h = hinf_norm(ss, np.linspace(-0.5, 0.5, 11), xtol=1e-8)
txt = h.explain()
# value and flag (review 2026-10-21, F14): the scan maximum is at omega = 0.5, where
# |(-i omega - A)^-1| = 1 / |0.01 + 0.5i| = 1.99960; it is flagged as a scan end
exact = 1 / abs(0.01 + 0.5j)
check(abs(h.Hinf - exact) < 1e-8 * exact and abs(h.omega - 0.5) < 1e-8,
      f"scan maximum {h.Hinf} at omega {h.omega}, expected {exact} at 0.5")
check(bool(h.at_scan_end), "at_scan_end is False for a maximum at the end of the scan")
check("sup_" not in txt, f"explain() claims the supremum: {txt}")
passed(txt[:100])
