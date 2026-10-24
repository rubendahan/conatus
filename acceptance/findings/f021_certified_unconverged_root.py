"""F021: a root whose secant did not converge cannot be "certified".

CertifiedRoot(converged=False) with a small residual and a tight resolution study: 0.1.0's
certify() returned status "certified".
"""
from _acceptance import check, passed, refused, start

start(21, "certify() requires a converged root")
from conatus.floquet import CertifiedRoot  # noqa: E402

root = CertifiedRoot(mu=0.1 + 0.2j, residual=1e-14, iterations=30, params={},
                     convergence={"N=32": 0.1 + 0.2j, "N=48": 0.1 + 0.2j}, converged=False)
was_refused, r = refused(root.certify, residual_tol=1e-10, convergence_tol=1e-8)
if was_refused:
    passed("certify() refused an unconverged root")
check(root.status != "certified", f"status {root.status!r} with converged=False")
# audit of 0.1.2, F19 / F05: the evidence must be about the current root, which must be finite,
# and a change after certify() withdraws the certification
other = CertifiedRoot(mu=100 + 0j, residual=0.0, iterations=1, params={},
                      convergence={"N=16": 0j, "N=32": 0j})
other.certify(residual_tol=1e-12, convergence_tol=1e-10)
check(other.status != "certified", "a root of 100 certified by a study converged at 0")
good = CertifiedRoot(mu=0.1 + 0.2j, residual=1e-14, iterations=3, params={},
                     convergence={"N=32": 0.1 + 0.2j, "N=48": 0.1 + 0.2j})
good.certify(residual_tol=1e-10, convergence_tol=1e-8)
check(good.status == "certified", f"a consistent root not certified: {good.reasons}")
good.mu = complex(float("nan"), 0.0)
check(good.status != "certified", "status still certified after mu was replaced by NaN")
passed(f"status {root.status!r}, reasons {root.reasons}; unrelated and stale evidence refused")
