"""Level B (§70): reference transforms, round trips with and without dealiasing."""
import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

from conatus.ir import CartesianCoordinates, Chebyshev, ComplexFourier, RealFourier
from conatus.reference import transforms as T

cs = CartesianCoordinates("x")
x = cs["x"]
BASES = [lambda d: Chebyshev(x, 24, (-1, 3), dealias=d),
         lambda d: RealFourier(x, 24, (0, 5), dealias=d),
         lambda d: ComplexFourier(x, 24, (0, 5), dealias=d)]


@pytest.mark.parametrize("make", BASES)
@pytest.mark.parametrize("dealias", [1, 3 / 2, 2])
def test_coeff_grid_coeff_roundtrip(make, dealias):
    b = make(dealias)
    rng = np.random.default_rng(1)
    c = rng.standard_normal((b.size, 3))
    if isinstance(b, ComplexFourier):
        c = c + 1j * rng.standard_normal(c.shape)
        c[b.size // 2] = 0
    if isinstance(b, RealFourier):
        c[1] = 0
    g = T.backward(c, b, axis=0)
    assert g.shape[0] == b.grid_size
    assert np.max(abs(T.forward(g, b, axis=0) - c)) < 1e-14 * np.max(abs(c)) * b.size


def test_chebyshev_matches_numpy_values():
    b = Chebyshev(x, 16, (-1, 3), dealias=3 / 2)
    c = np.random.default_rng(2).standard_normal(16)
    t = 2 * (T.grid(b) + 1) / 4 - 1
    assert np.max(abs(T.backward(c, b) - C.chebval(t, c))) < 1e-13


def test_real_fourier_convention():
    b = RealFourier(x, 8, (0, 2 * np.pi), dealias=1)
    X = T.grid(b)
    c = T.forward(3 + 2 * np.cos(X) - 5 * np.sin(3 * X), b)
    expected = np.zeros(8)
    expected[0], expected[2], expected[7] = 3, 2, 5      # (-sin) coefficient
    assert np.max(abs(c - expected)) < 1e-14


def test_forward_truncates_aliased_grid_modes():
    b = RealFourier(x, 8, (0, 2 * np.pi), dealias=3 / 2)
    X = T.grid(b)
    c = T.forward(np.cos(X) + np.cos(5 * X), b)          # k = 5 is beyond N/2 = 4
    assert abs(c[2] - 1) < 1e-14 and np.max(abs(np.delete(c, 2))) < 1e-14


def test_wrong_sizes_refused():
    b = Chebyshev(x, 16, (-1, 1), dealias=3 / 2)
    with pytest.raises(ValueError, match="grid points"):
        T.forward(np.zeros(16), b)
    with pytest.raises(ValueError, match="even"):
        T.forward(np.zeros(9), RealFourier(x, 9, (0, 1), dealias=1), dealias=False)
