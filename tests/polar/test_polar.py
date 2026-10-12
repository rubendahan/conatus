"""Polar geometry: regular disk and annulus operators.

References: Vasil et al., J. Comput. Phys. 325 (2016) 53-73 (regular polar bases); zeros of J_m from
scipy.special.jn_zeros (tabulated in Abramowitz & Stegun, Table 9.5); Taylor-Couette critical point for
eta = 0.5, outer cylinder at rest: Re_c = 68.19, k_c = 3.16 (Recktenwald, Lucke & Muller, Phys. Rev. E 48,
4444 (1993), Table I; values to be re-checked against the source, see polar.md).
"""
import numpy as np
import pytest
import sympy as sp_
from scipy.special import jn_zeros

from conatus.discretize.polar import Annulus, Disk, cheb_coeffs, evaluate, taylor_couette_eigs, to_T
from conatus.ir import PolarCoordinates


def test_polar_coordinates_declare_metric_and_regularity():
    pc = PolarCoordinates("phi", "r")
    assert pc.scale_factors() == ("r", 1)
    assert "r^|m| w(r^2)" in pc.regularity()
    assert "DiskBasis" in pc.explain() and "eps" in pc.regularity()


@pytest.mark.parametrize("m", [0, 1, 3])
def test_disk_poisson_exact_polynomial(m):
    # u = rho^m (1 - rho^2): w(z) = (1 - z)/2 and lap u = -4(m+1) rho^m
    d = Disk(16)
    g = np.zeros(16); g[0] = -4 * (m + 1)
    w, res = d.helmholtz(m).solve(d.rhs(g), [0.0])
    exact = np.zeros(16); exact[:2] = [0.5, -0.5]
    assert np.abs(w - exact).max() < 1e-13 and res < 1e-13


def _disk_mms(m, k2, N):
    rho = sp_.symbols("rho", positive=True)
    f = rho ** m * (1 - rho ** 2) * sp_.exp(rho ** 2)            # regular, f(1) = 0
    g = sp_.diff(f, rho, 2) + sp_.diff(f, rho) / rho - m * m * f / rho ** 2 - k2 * f
    q = sp_.lambdify(rho, sp_.simplify(g / rho ** m), "numpy")   # g = rho^m q(rho^2)
    fw = sp_.lambdify(rho, sp_.simplify(f / rho ** m), "numpy")
    d = Disk(N)
    z = np.cos(np.pi * (np.arange(N) + 0.5) / N)
    rr = np.sqrt((1 + z) / 2)
    w, res = d.helmholtz(m, k2).solve(d.rhs(cheb_coeffs(q(rr))), [0.0])
    zt = np.linspace(-1, 1, 41)
    return np.abs(evaluate(w, 0, zt) - fw(np.sqrt((1 + zt) / 2))).max(), res


@pytest.mark.parametrize("m,k2", [(0, 0.0), (2, 0.0), (1, 3.0)])
def test_disk_helmholtz_mms_converges_spectrally(m, k2):
    e8, _ = _disk_mms(m, k2, 8)                                    # under-resolved: ~1e-6
    e20, res = _disk_mms(m, k2, 20)                                # resolved: roundoff
    assert e20 < 1e-12 and e20 < 1e-5 * e8 and res < 1e-12


@pytest.mark.parametrize("m", [0, 1, 2, 3])
def test_disk_bessel_eigenvalues(m):
    lam = Disk(32).bessel_modes(m, 4)
    assert np.abs(lam / jn_zeros(m, 4) ** 2 - 1).max() < 1e-12


@pytest.mark.parametrize("m", [0, 1, 2, 4])
def test_disk_div_grad_is_laplacian_and_curl_grad_vanishes(m):
    d = Disk(20)
    rng = np.random.default_rng(m)
    w = np.zeros(20); w[:12] = rng.standard_normal(12)
    A, B = d.gradient(m)
    a, b = to_T(A @ w, 1), to_T(B @ w, 1)
    Da, Db = d.divergence(m)
    Ca, Cb = d.curl(m)
    lap = to_T(d.laplacian(m) @ w, 2)
    assert np.abs(to_T(Da @ a + Db @ b, 1) - lap).max() < 1e-9 * np.abs(lap).max()
    assert np.abs(to_T(Ca @ a + Cb @ b, 1)).max() < 1e-9 * np.abs(lap).max()


@pytest.mark.parametrize("m", [0, 1, 3])
def test_disk_gradient_is_regular_and_exact_near_the_axis(m):
    # f = rho^m w(2 rho^2 - 1) with a random w; compare the spin representation with f_r, (i m / r) f
    d = Disk(16)
    w = np.zeros(16); w[:8] = np.random.default_rng(10 + m).standard_normal(8)
    a, b = (to_T(M @ w, 1) for M in d.gradient(m))
    for rho in (0.0, 1e-6, 1e-3, 0.3, 0.9):
        z = 2 * rho * rho - 1
        up = rho ** (m + 1) * evaluate(a, 0, z)[0]
        um = rho ** abs(m - 1) * evaluate(b, 0, z)[0]
        assert np.isfinite(up) and np.isfinite(um)                 # finite at the axis, no eps
        if rho > 0:
            h = 1e-7
            f = lambda r: r ** m * evaluate(w, 0, 2 * r * r - 1)[0]
            fr = (f(rho + h) - f(rho - h)) / (2 * h)                  # rho >= 1e-6 > h
            ur, uphi = (up + um) / 2, (up - um) / 2j
            assert abs(ur - fr) < 1e-5 * max(1, abs(fr))
            assert abs(uphi - 1j * m * f(rho) / rho) < 1e-7 * max(1, abs(f(rho) / rho))


def test_disk_modes_have_parity_minus_one_to_the_m():
    # rho^m w(rho^2) is odd/even in rho with m: f(-rho) = (-1)^m f(rho) by construction
    w = np.random.default_rng(0).standard_normal(8)
    for m in range(5):
        f = lambda r: r ** m * evaluate(w, 0, 2 * r * r - 1)[0]
        assert abs(f(-0.37) - (-1) ** m * f(0.37)) < 1e-14


def test_negative_m_is_refused_explicitly():
    with pytest.raises(NotImplementedError, match="conj"):
        Disk(8).laplacian(-1)


@pytest.mark.parametrize("m,k2", [(0, 0.0), (2, 0.0), (3, 2.0)])
def test_annulus_helmholtz_mms(m, k2):
    r = sp_.symbols("r", positive=True)
    f = sp_.sin(sp_.pi * (r - sp_.Rational(1, 2))) * sp_.exp(r)     # zero at r = 0.5 and r = 1.5
    g = sp_.diff(f, r, 2) + sp_.diff(f, r) / r - m * m * f / r ** 2 - k2 * f
    gf, ff = sp_.lambdify(r, g, "numpy"), sp_.lambdify(r, f, "numpy")
    an = Annulus(28, 0.5, 1.5)
    t = np.cos(np.pi * (np.arange(28) + 0.5) / 28)
    sys_ = an.helmholtz(m, k2)
    x, res = sys_.solve(an.rhs(cheb_coeffs(gf(an.c + an.h * t))), [0.0, 0.0])
    rr = np.linspace(0.5, 1.5, 41)
    assert np.abs(evaluate(x, 0, (rr - an.c) / an.h) - ff(rr)).max() < 1e-11 and res < 1e-12
    assert "PASS" in sys_.explain() and "r-multiplied" in sys_.explain()


def test_annulus_r_div_and_r_curl_of_a_gradient():
    an = Annulus(24, 1.0, 2.0)
    m = 2
    r = sp_.symbols("r", positive=True)
    f = sp_.cos(r) * r ** 2
    fr, fl = sp_.lambdify(r, sp_.diff(f, r), "numpy"), sp_.lambdify(r, f, "numpy")
    lapf = sp_.lambdify(r, sp_.diff(f, r, 2) + sp_.diff(f, r) / r - m * m * f / r ** 2, "numpy")
    t = np.cos(np.pi * (np.arange(24) + 0.5) / 24)
    rg = an.c + an.h * t
    ur, uphi = cheb_coeffs(fr(rg)), cheb_coeffs(1j * m * fl(rg) / rg)      # u = grad f
    Dr, Dp = an.r_divergence(m)
    Cr, Cp = an.r_curl(m)
    rt = np.linspace(-1, 1, 21)
    rv = an.c + an.h * rt
    div = evaluate(to_T(Dr @ ur + Dp @ uphi, 1), 0, rt)
    curl = evaluate(to_T(Cr @ ur + Cp @ uphi, 1), 0, rt)
    assert np.abs(div - rv * lapf(rv)).max() < 1e-10
    assert np.abs(curl).max() < 1e-10


def test_taylor_couette_growth_changes_sign_near_published_Re_c():
    for N in (24, 32):
        grow = lambda Re: max(w.real for w in taylor_couette_eigs(N, 0.5, Re, 3.16) if abs(w) < 1e3)
        assert grow(60.0) < -0.05 and grow(75.0) > 0.03 and abs(grow(68.19)) < 1e-4


@pytest.mark.slow
def test_taylor_couette_critical_point():
    from scipy.optimize import brentq, minimize_scalar
    g = lambda Re, k: max(w.real for w in taylor_couette_eigs(28, 0.5, Re, k) if abs(w) < 1e3)
    res = minimize_scalar(lambda k: brentq(lambda R: g(R, k), 60, 80, xtol=1e-10),
                          bracket=(2.9, 3.16, 3.4), tol=1e-8)
    assert abs(res.fun - 68.19) < 0.01 and abs(res.x - 3.16) < 0.01
