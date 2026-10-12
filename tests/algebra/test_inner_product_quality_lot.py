"""Quality pass 4: the Gram matrix of ``InnerProduct`` is the one of the declared basis.

``InnerProduct.gram`` applied the closed-form Chebyshev T Gram to every non-periodic basis: the
energy norm (and the adjoint W⁻¹AᴴW, the optimal gain of ``Linearization.periodic_operator``) was
wrong for Legendre, Jacobi, Gegenbauer and compound intervals, and infinite on Hermite/Laguerre
axes. Each Gram is checked here against ∫ f² of a known function, from the coefficients the
basis's own transform gives."""
import numpy as np
import pytest

from conatus.discretize.families import family_of
from conatus.ir import CartesianCoordinates
from conatus.ir.bases import (Chebyshev, CompoundChebyshev, Hermite, Jacobi, Laguerre, Legendre,
                              Ultraspherical)
from conatus.linearize.inner_product import basis_gram

x = CartesianCoordinates("x")["x"]


@pytest.mark.parametrize("make", [
    lambda: Legendre(x, 12, (0.0, 2.0), dealias=1),
    lambda: Jacobi(x, 12, (0.0, 2.0), a=0.5, b=-0.3, dealias=1),
    lambda: Ultraspherical(x, 12, (0.0, 2.0), lam=1.0, dealias=1),
    lambda: Ultraspherical(x, 12, (0.0, 2.0), lam=2.0, dealias=1),
    lambda: CompoundChebyshev(x, (10, 12), (0.0, 0.7, 2.0), dealias=1),
])
def test_gram_of_bounded_bases_integrates_a_polynomial_exactly(make):
    b = make()
    fam = family_of(b)
    g = fam.grid(dealias=False)
    c = fam.forward(1 + g + 0.3 * g ** 3, dealias=False)
    exact = 2 + 2 * 2 ** 2 / 2 + 2 ** 3 / 3 + 2 * 0.3 * 2 ** 4 / 4 + 2 * 0.3 * 2 ** 5 / 5 \
        + 0.09 * 2 ** 7 / 7                         # ∫_0^2 (1 + x + 0.3 x³)² dx
    assert abs(c @ basis_gram(b, b.size) @ c - exact) < 1e-11 * exact


def test_gram_of_unbounded_bases_is_finite_and_exact():
    b = Hermite(x, 16, center=0.5, scale=2.0, dealias=1)
    fam = family_of(b)
    g = fam.grid(dealias=False)
    y = (g - 0.5) / 2.0
    c = fam.forward((1 + y) * np.exp(-y ** 2 / 2), dealias=False)     # psi_0 and psi_1 only
    assert abs(c @ basis_gram(b, b.size) @ c - 2.0 * 1.5 * np.sqrt(np.pi)) < 1e-12
    b = Laguerre(x, 16, start=0.0, scale=1.5, dealias=1)
    fam = family_of(b)
    g = fam.grid(dealias=False)
    y = g / 1.5
    c = fam.forward((1 + y) * np.exp(-y / 2), dealias=False)          # L_0 and L_1 only
    assert abs(c @ basis_gram(b, b.size) @ c - 1.5 * 5.0) < 1e-11


def test_chebyshev_gram_is_unchanged():
    from conatus.linearize.inner_product import chebyshev_gram
    b = Chebyshev(x, 9, (-1.0, 3.0), dealias=1)
    assert np.array_equal(basis_gram(b, 9), chebyshev_gram(9, -1.0, 3.0))
