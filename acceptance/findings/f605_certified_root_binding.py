"""F605 (audit of 0.1.2, F05): CertifiedRoot certifies only a finite root its evidence is about,
and a change of the root after certify() withdraws the certification."""
from _acceptance import check, passed, start

start(605, "CertifiedRoot evidence bound to the root")
from conatus.floquet import CertifiedRoot  # noqa: E402

r = CertifiedRoot(mu=100 + 0j, residual=0.0, iterations=1, params={},
                  convergence={"N=16": 0j, "N=32": 0j})
r.certify(residual_tol=1e-12, convergence_tol=1e-10)
check(r.status != "certified", "mu = 100 certified by a study converged at 0")
r.mu = complex(float("nan"), 0.0)
r.certify(residual_tol=1e-12, convergence_tol=1e-10)
check(r.status != "certified", "mu = NaN certified")
mu = 0.25 - 0.1j
g = CertifiedRoot(mu=mu, residual=0.0, iterations=1, params={},
                  convergence={"N=16": mu, "N=32": mu})
g.certify(residual_tol=1e-12, convergence_tol=1e-10)
check(g.status == "certified", f"consistent root not certified: {g.reasons}")
g.mu = 3 + 0j
check(g.status != "certified", "certification kept after mu changed")
passed("unrelated, NaN and changed roots are not certified")
