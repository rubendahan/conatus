"""F092: the POD retained-energy fraction is normalised by the TOTAL energy.

Snapshots with energies 1, 1/4, 1/16, ... (8 modes), rank 2 kept: the retained fraction is
(1 + 1/4) / sum. 0.1.0 normalised on the kept modes only and reported 1.
"""
import re

import numpy as np
from _acceptance import check, passed, start

start(92, "POD energy fraction over all modes")
from conatus.datadriven import DataInnerProduct, pod  # noqa: E402

rng = np.random.default_rng(1)
n, m, r = 40, 8, 2
U, _ = np.linalg.qr(rng.standard_normal((n, m)))
V, _ = np.linalg.qr(rng.standard_normal((m, m)))
sigma = 2.0 ** -np.arange(m)
X = U @ np.diag(sigma) @ V.T
p = pod(X, inner_product=DataInnerProduct.quadrature(np.ones(n)), rank=r, method="svd", mean="keep")
true = float(np.sum(sigma[:r] ** 2) / np.sum(sigma ** 2))
text = p.explain()
found = [float(x) for x in re.findall(r"fraction[^0-9]*([0-9.]+(?:e-?\d+)?)", text)]
values = found + [float(getattr(p, a)) for a in ("energy_fraction", "retained_fraction", "kept_fraction")
                  if hasattr(p, a)]
check(values, f"no retained-energy fraction reported (expected {true:.6g}); explain: {text[:200]}")
check(all(abs(v - true) < 1e-6 for v in values), f"reported fraction(s) {values}, expected {true:.6g}")
passed(f"fraction {values[0]:.6g}")
