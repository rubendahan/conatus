"""F734 (audit of 0.2.1, F05): a certification is bound to its root and evidence.

Oracle: construction, the evidence is changed by hand.
"""
from _acceptance import check, passed, start

start(734, "root certification follows its evidence")
import numpy as np  # noqa: E402


from conatus.floquet.roots import CertifiedRoot  # noqa: E402


def root():
    r = CertifiedRoot(mu=0.5 + 0j, residual=1e-12, iterations=3, params={"a": 1.0},
                      convergence={"N=8": 0.5 + 1e-11, "N=16": 0.5 + 0j})
    return r.certify(residual_tol=1e-8, convergence_tol=1e-6)


check(root().status == "certified", "control")
r = root()
r.params["a"] = 2.0
check(r.status == "provisional", "parameters changed in place")
r = root()
r.convergence["N=16"] = 7.0
check(r.status == "provisional", "study changed in place")
for bad in (-np.inf, np.nan, np.inf, -1.0):
    r = CertifiedRoot(mu=0.5 + 0j, residual=bad, iterations=3, params={},
                      convergence={"N=8": 0.5 + 0j, "N=16": 0.5 + 0j})
    check(r.certify(residual_tol=1e-8, convergence_tol=1e-6).status == "provisional", f"residual {bad}")
r = root()
r.mu = complex("nan")
check(r.status == "provisional", "mu replaced")
passed("status withdrawn on every change")
