"""Sparse Olver–Townsend operators vs dense oracles (numpy.polynomial, scipy Gegenbauer)."""
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.polynomial import chebyshev as C
from scipy.special import eval_gegenbauer

from conatus.discretize import ultraspherical as us
from conatus.errors import DiscretizationError

N = 24
t = np.linspace(-1, 1, 41)
c = np.random.default_rng(0).standard_normal(N) * np.exp(-0.3 * np.arange(N))


def val(lam, coef):
    if lam == 0:
        return C.chebval(t, coef)
    return eval_gegenbauer(np.arange(len(coef))[:, None], lam, t[None, :]).T @ coef


@pytest.mark.parametrize("order", [1, 2, 3])
def test_derivative(order):
    D = us.differentiation(N, 0, order)
    assert sp.issparse(D) and D.nnz == N - order
    ref = C.chebval(t, C.chebder(c, order))
    assert np.max(abs(val(order, D @ c) - ref)) < 1e-12 * np.max(abs(ref)) * 10 ** order


@pytest.mark.parametrize("lam", [0, 1, 2, 3])
def test_conversion(lam):
    cl = us.conversion_chain(N, 0, lam) @ c
    S = us.conversion(N, lam)
    assert S.nnz <= 2 * N
    assert np.max(abs(val(lam + 1, S @ cl) - val(lam, cl))) < 1e-14


@pytest.mark.parametrize("lam", [0, 1, 2])
def test_multiplication_exact_for_polynomial(lam):
    a = C.chebinterpolate(lambda s: np.cos(2 * s) + s ** 3, 12)
    cz = c.copy()
    cz[-14:] = 0                                   # keep the product inside the space
    M, rep = us.multiplication(a, N, lam, exact=True)    # a is the polynomial itself (no sample)
    cl = us.conversion_chain(N, 0, lam) @ cz
    assert M.nnz <= N * (2 * len(a) + 1)           # banded
    assert np.max(abs(val(lam, M @ cl) - C.chebval(t, a) * val(lam, cl))) < 1e-13
    assert rep.cutoff is None and rep.n_coeffs == 13


def test_ncc_tail_reported_never_silent():
    a = C.chebinterpolate(np.exp, 6)               # under-resolved exp: refused
    with pytest.raises(DiscretizationError, match="not resolved"):
        us.multiplication(a, N, 0)
    _, rep = us.multiplication(a, N, 0, tail_tol=1e-2)   # accepted as declared, and reported
    assert rep.resolved and "sampled tail accepted up to 0.01" in rep.explain()
    a_good = C.chebinterpolate(np.exp, 20)
    _, rep2 = us.multiplication(a_good, N, 2)
    assert rep2.resolved and "no truncation" in rep2.explain()
    _, rep3 = us.multiplication(a_good, N, 0, cutoff=1e-8)
    assert rep3.n_coeffs < 21 and "dropped" in rep3.explain() and rep3.dropped_max > 0


def test_boundary_rows_and_integral():
    for lam in (0, 2):
        cl = us.conversion_chain(N, 0, lam) @ c
        for d in (0, 1, 2):
            for s in (-1.0, 1.0):
                ref = C.chebval(s, C.chebder(c, d)) if d else C.chebval(s, c)
                assert abs(us.evaluation_row(N, lam, s, d) @ cl - ref) < 1e-11
    ci = C.chebint(c)
    assert abs(us.integration_row(N, 0) @ c - (C.chebval(1, ci) - C.chebval(-1, ci))) < 1e-14
    assert abs(us.integration_row(N, 2) @ (us.conversion_chain(N, 0, 2) @ c)
               - (C.chebval(1, ci) - C.chebval(-1, ci))) < 1e-13


def test_non_integer_lambda_refused():
    with pytest.raises(NotImplementedError, match="Legendre"):
        us.conversion(N, 0.5)
