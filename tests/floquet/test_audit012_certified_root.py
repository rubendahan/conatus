"""Audit of 0.1.2, F05: a CertifiedRoot certifies only a finite root that its own evidence is
about, and loses the certification when the root or the evidence changes afterwards."""
import math

from conatus.floquet import CertifiedRoot


def _root(mu, study):
    return CertifiedRoot(mu=mu, residual=0.0, iterations=1, params={}, convergence=study)


def test_unrelated_root_is_not_certified():
    r = _root(100 + 0j, {"N=16": 0j, "N=32": 0j}).certify(residual_tol=1e-12,
                                                          convergence_tol=1e-10)
    assert r.status == "provisional"
    assert any("not evidence for this root" in s for s in r.reasons)


def test_nan_root_is_not_certified():
    r = _root(complex(math.nan, 0), {"N=16": 0j, "N=32": 0j}).certify(residual_tol=1e-12,
                                                                      convergence_tol=1e-10)
    assert r.status == "provisional" and any("not finite" in s for s in r.reasons)


def test_matching_root_is_certified_and_mutation_withdraws_it():
    mu = 0.1 + 0.2j
    r = _root(mu, {"N=16": mu + 1e-13, "N=32": mu}).certify(residual_tol=1e-12,
                                                             convergence_tol=1e-10)
    assert r.status == "certified"
    r.mu = complex(math.nan, 0)
    assert r.status == "provisional"
    r.mu = mu
    r.certify(residual_tol=1e-12, convergence_tol=1e-10)
    assert r.status == "certified"
    r.convergence["N=48"] = 5 + 0j              # in place: seen by explain()
    assert "status provisional" in r.explain()
