"""Findings 20 and 21: CertifiedRoot could be converged=False with status
"certified", and the documented contrast / winding checks were not part of certify()."""
import pytest

from conatus.errors import OptionError
from conatus.floquet.roots import CertifiedRoot


def _root(converged=True):
    r = CertifiedRoot(mu=0.1 + 1j, residual=1e-14, iterations=5, params={},
                      convergence={"N=32": 0.1 + 1j, "N=48": 0.1 + 1j})
    r.converged = converged
    return r


def test_unconverged_root_is_never_certified():
    r = _root(converged=False).certify(residual_tol=1e-10, convergence_tol=1e-8)
    assert r.status == "provisional" and any("not converged" in x for x in r.reasons)


def test_certified_says_what_was_checked():
    r = _root().certify(residual_tol=1e-10, convergence_tol=1e-8)
    assert r.status == "certified" and "residual and resolution study only" in r.explain()
    r = _root().certify(residual_tol=1e-10, convergence_tol=1e-8, contrast=1e-9,
                        contrast_max=1e-6, winding=1.0001)
    assert r.status == "certified" and "contrast" in r.explain() and "winding" in r.explain()


def test_failed_evidence_blocks_certification():
    assert _root().certify(residual_tol=1e-10, convergence_tol=1e-8, contrast=0.5,
                           contrast_max=1e-6).status == "provisional"
    assert _root().certify(residual_tol=1e-10, convergence_tol=1e-8,
                           winding=0.4).status == "provisional"
    with pytest.raises(OptionError):
        _root().certify(residual_tol=1e-10, convergence_tol=1e-8, contrast=1e-9)
