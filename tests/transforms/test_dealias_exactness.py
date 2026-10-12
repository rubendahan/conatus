"""Dealiasing padding rule against the exact projection of a product.

For a polynomial basis (Chebyshev T, Legendre, Gegenbauer, Jacobi, compound Chebyshev) and for
Fourier, the product of two fields of N modes evaluated on the grid of floor(N * dealias) points
and truncated back to N modes is the exact L2 projection of the product as soon as dealias >= 3/2
(Gauss-type quadrature of M points is exact to degree 2M - 1 >= 3N - 3; Fourier: no wave number
of the product folds onto |k| < N/2 when M >= 3N/2). With dealias = 1 it is not. The reference is
the same product on a grid 8 times finer, which is exact for these bases.

Hermite and Laguerre functions carry an exp(-y^2/2) / exp(-y/2) envelope: a product is not a finite
series in the same functions and the Gauss quadrature projection is *not* exact at 3/2 or 2. The
measured errors (random coefficients of unit size, N = 16, seed 2 of the audit script: 1.6e-1 / 3.0e-2 at 3/2 / 2 on Hermite,
3.7e-1 / 1.2e-1 on Laguerre; with coefficients decaying like e^{-0.3 n}: 3.7e-3 / 2.2e-5 and
7.3e-4 / 8.2e-5) are declared by the family's explain(); the test pins that the error falls with
the padding and that the declaration says so.
"""
import numpy as np
import pytest

from conatus.discretize.families import family_of
from conatus.ir import CartesianCoordinates, Domain, Field
from conatus.ir.bases import (Chebyshev, ComplexFourier, CompoundChebyshev, Hermite, Jacobi, Laguerre,
                              Legendre, RealFourier, Ultraspherical)
from conatus.reference import transforms as T
from conatus.timestep.spectral import SpectralEvaluator

cs = CartesianCoordinates("x")
x = cs["x"]
N = 16


def _basis(name, d):
    return {"chebyshev": lambda: Chebyshev(x, N, (0.3, 2.1), dealias=d),
            "legendre": lambda: Legendre(x, N, (0.3, 2.1), dealias=d),
            "gegenbauer": lambda: Ultraspherical(x, N, (0.3, 2.1), lam=1, dealias=d),
            "jacobi": lambda: Jacobi(x, N, (0.3, 2.1), a=0.5, b=-0.25, dealias=d),
            "compound": lambda: CompoundChebyshev(x, (8, 8), (0.3, 1.0, 2.1), dealias=d),
            "real_fourier": lambda: RealFourier(x, N, (0.5, 3.5), dealias=d),
            "complex_fourier": lambda: ComplexFourier(x, N, (0.5, 3.5), dealias=d),
            "hermite": lambda: Hermite(x, N, center=0.2, scale=1.1, dealias=d),
            "laguerre": lambda: Laguerre(x, N, start=0.1, scale=0.8, dealias=d)}[name]()


def _coeffs(name, rng, decay=0.0):
    c = [rng.standard_normal(N) * np.exp(-decay * np.arange(N)) for _ in range(2)]
    if name == "complex_fourier":
        c = [v + 1j * rng.standard_normal(N) * np.exp(-decay * np.arange(N)) for v in c]
        for v in c:
            v[N // 2] = 0                                # Nyquist entry: always 0 (convention)
    if name == "real_fourier":
        for v in c:
            v[1] = 0                                     # msin_0: always 0 (convention)
    return c


def _product_error(name, d, decay=0.0, seed=1):
    rng = np.random.default_rng(seed)
    cu, cv = _coeffs(name, rng, decay)
    b = _basis(name, d)
    D = Domain(cs, [b])
    u, v = Field("u", D), Field("v", D)
    got = SpectralEvaluator(D).evaluate(u * v, {u: cu, v: cv})
    fine = _basis(name, 8.0)
    ref = T.forward(T.backward(cu, fine) * T.backward(cv, fine), fine)
    return np.max(np.abs(got - ref)) / np.max(np.abs(ref))


EXACT = ["chebyshev", "legendre", "gegenbauer", "jacobi", "compound", "real_fourier",
         "complex_fourier"]


@pytest.mark.parametrize("name", EXACT)
def test_three_halves_rule_is_exact(name):
    assert _product_error(name, 1.0) > 1e-2                     # aliased without padding
    for d in (1.5, 2.0):
        assert _product_error(name, d) < 2e-15, (name, d)


@pytest.mark.parametrize("name", ["hermite", "laguerre"])
def test_unbounded_products_are_not_exact_and_say_so(name):
    e = [_product_error(name, d) for d in (1.5, 2.0, 3.0)]
    assert e[0] > 1e-2 and e[0] > e[1] > e[2], e                 # not exact; falls with padding
    ed = [_product_error(name, d, decay=0.3) for d in (1.5, 2.0)]
    assert 1e-6 < ed[0] < 5e-2 and ed[1] < ed[0], ed             # resolved fields: smaller
    text = family_of(_basis(name, 1.5)).explain()
    assert "not exact" in text and "3/2" in text, text


def test_integration_rows_and_grams_use_refined_weights():
    """Quality pass 9 (Q8.8): exact-degree integration rows and Gram matrices on the refined
    Gauss-Legendre weights. With scipy's roots_legendre the Legendre Gram diagonal was off by
    3.1e-13 (101 points) and 1.9e-12 (201 points) relative; refined: 1.0e-14 / 3.2e-14 (the
    evaluation of P_n itself)."""
    from conatus.discretize.ultraspherical import integration_row
    from conatus.linearize.inner_product import basis_gram
    N = 200
    n = np.arange(N)
    # int_{-1}^{1} C_n^(1)(t) dt = 2/(n+1) for even n, 0 for odd n
    exact = np.where(n % 2 == 0, 2.0 / (n + 1), 0.0)
    # literal bounds kept (Q10.15, margins 2.7-3.0): dominated by the recurrence
    # evaluation of C_n^(1) and P_n (eval_gegenbauer, eval_jacobi), which no product rounding model
    # describes; the pre-pass-9 weights (3.1e-13 / 1.9e-12) fail them 3-19x
    assert np.max(np.abs(integration_row(N, 1) - exact)) < 1e-13   # |C_n^(1)| <= n + 1: eps n floor
    G = basis_gram(Legendre(x, N, (-1.0, 1.0), dealias=1), N)
    assert np.max(np.abs(np.diag(G) * (2 * n + 1) / 2 - 1)) < 1e-13
    assert np.max(np.abs(G - np.diag(np.diag(G)))) < 1e-13
