"""Gauss quadrature nodes and weights of the curvilinear transforms.

numpy's ``leggauss`` weights carry a relative error growing with M (3.9e-12 at M = 97 near the
ends) and scipy's ``roots_jacobi(M, 0, 1/2)`` weights 1e-14 absolute: the spherical harmonic
transform had |F B - I| = 1.6e-13 at L = 64, the disk transform 1.1e-13 at N = 40 and the ball
6e-14 at N = 32 (a cubic's round trip 60x Dedalus'). All three now use the Newton-refined
long-double nodes and Christoffel weights of ``discretize.jacobi_family``. The second source was
the normalisation s_n of the Jacobi polynomials, a difference of four float64 log-gammas
(|lgamma| eps: 1e-13 at n = 60); it is now a long-double product of rational ratios
(``polar_radial.log_jacobi_norms``). Checked against 40-digit closed forms and through F B = I:
sphere L = 64 1.6e-13 -> 5e-15, disk N = 40 1.1e-13 -> 6e-15, ball N = 32 6e-14 -> 3e-15.
"""
import numpy as np
import pytest
from rounding_bounds import EPS, product_identity_bound

from conatus.discretize import polar_radial as pr
from conatus.discretize.jacobi_family import _nodes, gauss_legendre
from conatus.discretize.sphere_ops import BallOps, SphereHarmonics

mp = pytest.importorskip("mpmath")
DPS = 40


@pytest.fixture(autouse=True)
def _digits():
    """Working precision for this module only (never the process-wide mpmath setting)."""
    with mp.workdps(DPS):
        yield


@pytest.mark.parametrize("M", [25, 97])
def test_gauss_legendre_weights_to_rounding(M):
    """Exact node x*_j (Newton in 40 digits from the returned one), exact weight
    2 / ((1 - x*^2) P_M'(x*)^2): returned node within 1 ulp, weight within 4e-15 relative
    (numpy's leggauss: 1e-13 at M = 97 near the ends)."""
    x, w = gauss_legendre(M)
    assert np.all(np.diff(x) > 0)

    def P(t):
        return mp.legendre(M, t)

    for j in (0, 1, M // 3, M // 2, M - 1):
        xs = mp.mpf(float(x[j]))
        for _ in range(4):
            xs -= P(xs) / mp.diff(P, xs)
        dP = mp.diff(P, xs)
        ws = 2 / ((1 - xs * xs) * dP * dP)
        assert abs(float(xs) - x[j]) <= np.spacing(abs(x[j])) + 1e-300, (M, j)
        assert abs(float((mp.mpf(float(w[j])) - ws) / ws)) < 4e-15, (M, j)


def test_gauss_jacobi_ball_weights_integrate_exactly():
    """Gauss-Jacobi(0, 1/2), M = 48: sum_j w_j z_j^q = int (1+z)^(1/2) z^q dz for q < 2M (40 digits)."""
    M = 48
    z, w = _nodes(M, 0.0, 0.5)
    for q in (0, 1, 7, 40, 95):
        ref = mp.quad(lambda t: mp.sqrt(1 + t) * t ** q, [-1, 0, 1])
        got = mp.fsum(mp.mpf(float(wi)) * mp.mpf(float(zi)) ** q for wi, zi in zip(w, z))
        assert abs(float((got - ref) / max(abs(ref), 1))) < 2e-15, q


def _fb(F, B, ok=None):
    """max |F B - I| / bound over the entries (ok: the valid rows/columns), with the worst-case
    rounding bound 2 M eps |F| |B| of tests/rounding_bounds.py ( rule: the former literals
    1e-14 / 5e-15 were met at 6.0e-15 / 2.8e-15, margins 1.7-2.1, run
   )."""
    D = np.abs(F @ B - np.eye(B.shape[1])) / product_identity_bound(F, B)
    return (D if ok is None else D[np.ix_(ok, ok)]).max()


def test_transforms_are_orthogonal_to_rounding():
    """F B = I to the rounding bound of the product; the scipy-weight transforms of before quality
    pass 8 (1.6e-13 sphere, 1.1e-13 disk, 6e-14 ball) exceed it 3 to 8 times."""
    H = SphereHarmonics(64)
    M = 97
    for m, s in ((0, 0), (3, 0), (1, 1), (5, -1)):
        ok = H.valid(m, s)
        assert _fb(H.forward(m, s, M), H.backward(m, s, M), ok) <= 1, (m, s)
    d = pr.DiskOps(40, 1.0)
    for k in (0, 1, 4, 10):
        assert _fb(d.forward(k, 60, 40), d.backward(k, 60, 0, 0, 40)) <= 1, k
    b = BallOps(32, 1.0)
    for k in (0, 3):
        n = b.size(k)
        assert _fb(b.forward(k, 48, n), b.backward(k, 48, n)) <= 1, k
    # sensitivity: the former errors are above the bound (2 M eps <= 2 x 97 x 2.2e-16 = 4.3e-14)
    assert 2 * 97 * EPS < 1.6e-13 / 3 and 2 * 60 * EPS < 1.1e-13 / 4 and 2 * 48 * EPS < 6e-14 / 2.5


@pytest.mark.parametrize("a,b", [(0, 0), (0, 0.5), (2, 3.5), (3, 3), (50, 47), (0, 10.5)])
def test_jacobi_norms_to_rounding(a, b):
    from conatus.discretize.polar_radial import log_jacobi_norms
    got = log_jacobi_norms(a, b, 70)
    for n in (0, 1, 7, 69):
        ref = mp.log(2 / mp.mpf(2 * n + a + b + 1)) + mp.loggamma(n + a + 1) + mp.loggamma(n + b + 1) \
            - mp.loggamma(n + a + b + 1) - mp.loggamma(n + 1)
        assert abs(float(mp.mpf(float(got[n])) - ref / 2)) < 2e-16 * max(1, abs(float(ref))), (a, b, n)
