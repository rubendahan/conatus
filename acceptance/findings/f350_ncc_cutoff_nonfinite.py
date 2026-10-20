"""F350: ncc_cutoff=nan or inf must be refused (audit inputs, 2026-10-20).

0.1.x kept only the first Chebyshev coefficient of every non-constant coefficient with
ncc_cutoff=nan (U = 1 - y^2 became its mean, the Orr-Sommerfeld spectrum changed without a word),
and ncc_cutoff=inf also switched off the refusal of an under-resolved coefficient.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(350, "non-finite or out-of-range ncc_cutoff refused")
m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
m.scalar("v")
m.given(U=lambda y: 1 - y**2)
m.eigenvalue("c", growth="imag")
m.define("L2(f) = dy(dy(f)) - alpha**2*f")
m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")
for bad in (np.nan, np.inf, 1.0, -1e-8):
    was_refused, _ = refused(m.eigenvalues, 32, ncc_cutoff=bad)
    check(was_refused, f"ncc_cutoff={bad} accepted")
passed("nan, inf, 1 and a negative cutoff refused")
