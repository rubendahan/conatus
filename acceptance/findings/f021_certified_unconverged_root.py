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
passed(f"status {root.status!r}, reasons {root.reasons}")
