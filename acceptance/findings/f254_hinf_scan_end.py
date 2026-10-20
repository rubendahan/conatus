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
check("sup_" not in txt, f"explain() claims the supremum: {txt}")
passed(txt[:100])
