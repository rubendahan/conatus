"""F042: reusing an SBDF ImexIntegrator must not carry a hidden history into a new run.

run(x0, 0, 5) twice from the same x0 must give the same answer (the second call restarts from
x0), or the integrator must refuse / offer an explicit continuation. In 0.1.0 the second call
reused the multistep history of the first run.
"""
import numpy as np
import scipy.sparse as sp
from _acceptance import check, need, passed, refused, start

start(42, "IMEX runs from a new start state do not reuse an old history")
need("conatus.runtime")
from conatus.runtime.imex import ImexIntegrator  # noqa: E402

n = 3
integ = ImexIntegrator(sp.identity(n, format="csr"), sp.identity(n, format="csr"), [], "SBDF3", 0.1)
x0 = np.ones(n)
a = np.asarray(integ.run(x0, 0.0, 5)[0])
was_refused, r = refused(integ.run, x0, 0.0, 5)
if was_refused:
    passed("second run from a new state refused")
b = np.asarray(r[0])
fresh = np.asarray(ImexIntegrator(sp.identity(n, format="csr"), sp.identity(n, format="csr"), [],
                                  "SBDF3", 0.1).run(x0, 0.0, 5)[0])
check(np.allclose(a, fresh, rtol=0, atol=1e-15), f"first run {a[0]} vs fresh integrator {fresh[0]}")
check(np.allclose(b, a, rtol=0, atol=1e-15), f"second run from the same x0: {b[0]}, first {a[0]}")
passed(f"x(0.5) = {a[0]:.12g} both times")
