"""F039: the low-level IMEX integrator must not drop the complex part of F.

M = I, L = 0, F(x, t) = 1j: one RK222 step of 0.1 from x = 0 gives x = 0.1j. 0.1.0 built real
factors from the real M and L and silently cast F to real (x = 0). Expected: 0.1j or a refusal.
"""
import numpy as np
import scipy.sparse as sp
from _acceptance import check, need, passed, refused, start

start(39, "low-level IMEX keeps a complex right-hand side")
need("conatus.runtime")
from conatus.runtime.imex import ImexIntegrator  # noqa: E402

n = 3


def go():
    integ = ImexIntegrator(sp.identity(n, format="csr"), sp.csr_matrix((n, n)), [], "RK222", 0.1,
                           F=lambda x, t: np.full(n, 1j))
    return integ.run(np.zeros(n, dtype=complex), 0.0, 1)[0]


was_refused, x = refused(go)
if was_refused:
    passed("complex F with real operators refused")
err = float(np.max(np.abs(np.asarray(x) - 0.1j)))
check(err < 1e-14, f"x = {x}, expected 0.1j")
passed(f"x = 0.1j (error {err:.2g})")
