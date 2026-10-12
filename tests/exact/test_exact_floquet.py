"""Floquet–Hill exponents against closed forms (audit coverage hole 3).

The Floquet tests so far compared Hill roots with a numerical monodromy or with published tables;
none met an exponent known in closed form, and none went through ``hill_from_plan`` with a mass
matrix M != I. Here:

(a) Mathieu x'' + (a - 2q cos 2t) x = 0 at small q: characteristic exponent nu from the
    perturbation series of the characteristic value (DLMF 28.15.1, A&S 20.3.15)
        a = nu^2 + q^2 / (2 (nu^2 - 1)) + (5 nu^2 + 7) q^4 / (32 (nu^2 - 1)^3 (nu^2 - 4))
            + (9 nu^4 + 58 nu^2 + 29) q^6 / (64 (nu^2 - 1)^5 (nu^2 - 4) (nu^2 - 9)) + O(q^8);
    the discrepancy of the series truncated at q^4 must *be* the q^6 term (checked to a factor),
    which pins the series and the Hill root together.
(b) Meissner x'' + (a + b sgn cos t) x = 0: piecewise-constant coefficient, exponent from the trace
    of the product of the two exact transfer matrices (cos / sin of sqrt(a +- b) pi). The square
    wave has harmonics ~ 1/m, so the Hill truncation converges algebraically: the observed order
    (measured 3.0 for these cases, Nh = 8..128) is part of the test.
(c) M dx/dt + L(t) x = 0 with M != I, built by ``hill_from_plan`` from an IR EVP (the path where the
    compiler assembles M): Mathieu as a 2-unknown system with masses 2 and 3, and a Dirichlet
    diffusion problem m (s u + a u) + u - D u'' + b cos(wt) u = 0, whose exponents are exactly
    -(m a + 1 + D (n pi/2)^2) / m (u = e^{-((m a + 1 + D k^2) t + b sin(wt)/w)/m} sin k(z+1)).
    Both carry a *bare* unknown (coefficient 1, cancelled by a ``Parameter`` so that the IR does not
    fold it) and a nested eigenvalue ``c * (s u + ...)`` ( family).

Mutation check ( method, plugin outside the repository): (c) fails with reintroduced (bare
unknown in M) and with reintroduced (sigma part of a product dropped when the eigenvalue is
not a direct factor). (a) and (b) build their Hill blocks by hand and never call the compiler: they
test the Hill engine and the root finder, not M, and are unaffected by these mutations by design.
"""
import numpy as np
import pytest

import conatus.expert as C
from conatus.discretize import discretize
from conatus.floquet.hill_general import HillBlocks, exponents_dense, hill_from_plan, secant_root


def _dist_mod(mu, ref, omega, kmax=40):
    """Distance between two exponents modulo i omega (harmonic relabelling of Floquet exponents)."""
    return min(abs(mu - ref - 1j * omega * k) for k in range(-kmax, kmax + 1))


# ---------------------------------------------------------------------------------------------
# (a) Mathieu, small q: perturbation series of the characteristic value
# ---------------------------------------------------------------------------------------------

def _mathieu_series_terms(nu, q):
    """Terms q^0 + q^2, q^4, q^6 of a(nu, q) (DLMF 28.15.1; valid away from integer nu)."""
    n2 = nu * nu
    t2 = n2 + q ** 2 / (2 * (n2 - 1))
    t4 = (5 * n2 + 7) * q ** 4 / (32 * (n2 - 1) ** 3 * (n2 - 4))
    t6 = (9 * n2 ** 2 + 58 * n2 + 29) * q ** 6 / (64 * (n2 - 1) ** 5 * (n2 - 4) * (n2 - 9))
    return t2, t4, t6


def _mathieu_hill(a, q, Nh):
    """z = (x, x'), dz/dt + L(t) z = 0, L(t) = [[0, -1], [a - 2q cos 2t, 0]], period pi (omega = 2)."""
    L0 = np.array([[0, -1], [a, 0]], complex)
    Cq = np.array([[0, 0], [-q, 0]], complex)        # -2q cos 2t = -q (e^{2it} + e^{-2it})
    return HillBlocks(diag=lambda mu, n: L0 + (mu + 2j * n) * np.eye(2), couplings={1: Cq, -1: Cq},
                      Nh=Nh, size=2, label="Mathieu", omega=2.0, mass=np.eye(2))


@pytest.mark.parametrize("nu,q", [(0.4, 0.05), (1.5, 0.1)])
def test_mathieu_small_q_matches_perturbation_series(nu, q):
    """Given nu, a(nu, q) from the series; the Hill root must be i nu (mod 2i).

    Series to q^4: the error must be the omitted q^6 term, delta nu = t6 / (2 nu) (d a / d nu = 2 nu
    at leading order), to within a factor 1.5 (the remaining O(q^8) and the second-order change of
    nu). Series to q^6: error below 2e-11 at q = 0.05 (O(q^8) ~ 1e-12) and 1e-9 at q = 0.1.
    """
    t2, t4, t6 = _mathieu_series_terms(nu, q)
    predicted = abs(t6) / (2 * nu)
    r4 = secant_root(_mathieu_hill(t2 + t4, q, 16), 1j * nu + 1e-3, tol=1e-14, max_iterations=80)
    r6 = secant_root(_mathieu_hill(t2 + t4 + t6, q, 16), 1j * nu + 1e-3, tol=1e-14, max_iterations=80)
    assert r4.converged and r6.converged          # an unconverged search raises (Y8)
    mu4, mu6 = r4.mu, r6.mu
    err4, err6 = _dist_mod(mu4, 1j * nu, 2.0), _dist_mod(mu6, 1j * nu, 2.0)
    assert predicted / 1.5 < err4 < 1.5 * predicted          # the discrepancy is the q^6 term
    assert err6 < (2e-11 if q <= 0.05 else 1e-9)


# ---------------------------------------------------------------------------------------------
# (b) Meissner: piecewise-constant Hill equation, closed-form exponent
# ---------------------------------------------------------------------------------------------

def _transfer(k, length):
    """Exact transfer matrix of x'' + k x = 0 over ``length`` (k of any sign, complex sqrt)."""
    w = np.sqrt(complex(k))
    return np.array([[np.cos(w * length), np.sin(w * length) / w],
                     [-w * np.sin(w * length), np.cos(w * length)]])


def _meissner_exact(a, b):
    """Exponents of x'' + (a + b sgn cos t) x = 0 (period 2 pi): the monodromy over
    [-pi/2, 3pi/2] is P = T(a - b, pi) T(a + b, pi), det P = 1, so rho^2 - tr(P) rho + 1 = 0."""
    P = _transfer(a - b, np.pi) @ _transfer(a + b, np.pi)
    rho = np.roots([1.0, -np.trace(P), 1.0]).astype(complex)
    return np.log(rho) / (2 * np.pi)


def _meissner_hill(a, b, Nh):
    """Hill blocks with the square wave truncated at |m| <= Nh:
    sgn cos t = (4/pi) sum_k (-1)^k cos((2k+1) t) / (2k+1), so c_{+-m} = (2/pi) (-1)^k / m, m = 2k+1."""
    L0 = np.array([[0, -1], [a, 0]], complex)
    C = {}
    for m in range(-Nh, Nh + 1):
        if m % 2:
            k = (abs(m) - 1) // 2
            C[m] = np.array([[0, 0], [b * (2 / np.pi) * (-1) ** k / abs(m), 0]], complex)
    return HillBlocks(diag=lambda mu, n: L0 + (mu + 1j * n) * np.eye(2), couplings=C, Nh=Nh, size=2,
                      label="Meissner", omega=1.0, mass=np.eye(2))


@pytest.mark.parametrize("a,b", [(1.5, 0.5),      # stable: exponent +- i (sqrt 2 - 1)/2 exactly
                                 (0.3, 0.2),      # first tongue: real part != 0, Im = 1/2
                                 (1.0, 0.6)])     # unstable, real exponent
def test_meissner_exponent_closed_form(a, b):
    """Algebraic convergence to the closed form, measured on this login (Nh = 8..128): error
    ~ 2.3e-3 Nh^-3 (1.5, 0.5), observed order 3.00 +- 0.01 in all three cases (the square-wave
    harmonics decay as 1/m). Test: error(64) < 2e-7 and observed order in [2.7, 3.3] from Nh 32->64."""
    exact = _meissner_exact(a, b)
    errs = []
    for Nh in (31, 63):                         # odd: the last retained harmonic is a nonzero one
        r = secant_root(_meissner_hill(a, b, Nh), exact[0] + 1e-3, tol=1e-13, max_iterations=80)
        assert r.converged and not r.reasons
        mu = r.mu
        errs.append(min(_dist_mod(mu, e, 1.0) for e in exact))
    order = np.log(errs[0] / errs[1]) / np.log(63 / 31)
    assert errs[1] < 2e-7 and 2.7 < order < 3.3
    if (a, b) == (1.5, 0.5):                    # tr P = 2 cos(2 pi nu) with nu = (sqrt 2 - 1)/2
        assert abs(abs(exact[0].imag) - (np.sqrt(2) - 1) / 2) < 1e-14


# ---------------------------------------------------------------------------------------------
# (c) hill_from_plan with M != I: bare unknown + nested eigenvalue ( family)
# ---------------------------------------------------------------------------------------------

def _cheb(N):
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    b = C.Chebyshev(z, N, (-1, 1), dealias=1)
    return z, b, C.Domain(cs, [b])


def test_plan_mathieu_system_with_mass_diag_2_3():
    """2 (s x - v) + x - p x = 0 and 3 (s v + a x) + v - p v + U(t) x = 0, p = 1, U = -6q cos 2t:
    M = diag(2, 3), x'' + (a - 2q cos 2t) x = 0 again. Pointwise in z (4 Chebyshev coefficients, no
    derivative), so each exponent appears 4 times: all 2 x 4 exponents of the principal strip
    |Im mu| <= 1 must be +- i nu, with a(nu, q) from the series to q^6 (error ~ 1e-12, test (a))."""
    nu, q = 0.4, 0.05
    a = sum(_mathieu_series_terms(nu, q))
    z, b, D = _cheb(4)
    X, V, U = (C.Field(n, D, dtype=complex) for n in ("x", "v", "U"))
    s = C.Parameter("s", dtype=complex)
    c2, c3, p, A = C.Parameter("c2", 2.0), C.Parameter("c3", 3.0), C.Parameter("p", 1.0), C.Parameter("A", a)
    pb = C.EVP([X, V], eigenvalue=s)
    pb.add_equation((c2 * (s * X - V) + X - p * X, 0))            # nested s, bare x
    pb.add_equation((c3 * (s * V + A * X) + V - p * V + U * X, 0))  # nested s, bare v, base flow
    harm = {1: lambda Z: -3 * q + 0 * Z + 0j, -1: lambda Z: -3 * q + 0 * Z + 0j}   # -6q cos 2t
    H = hill_from_plan(lambda ncc: discretize(pb, ncc={U: ncc[U]}), {U: harm}, Nh=12, omega=2.0,
                       label="Mathieu, M = diag(2, 3)")
    w = exponents_dense(H, finite_cap=1e6)
    strip = w[np.abs(w.imag) <= 1.0]
    assert w.n_dropped == 0 and len(strip) == 8
    assert max(min(abs(m - 1j * nu), abs(m + 1j * nu)) for m in strip) < 1e-11


def test_plan_diffusion_with_periodic_coefficient_and_mass():
    """m (s u + a u) + u - D u'' + U(t) u + taus = 0, u(+-1) = 0, U = b cos(w t) uniform in z:
    u = exp(-((m a + 1 + D k^2) t + b sin(wt) / w) / m) sin(k (z + 1)), k = n pi / 2, hence the
    exponents -(m a + 1 + D k^2) / m exactly (the periodic factor carries no growth)."""
    m_, a_, d_, b_, om = 2.0, 0.4, 0.3, 1.5, 3.0
    z, b, D = _cheb(20)                         # N = 16 leaves 6e-9 on n = 3 (spatial, not Floquet)
    u, U = C.Field("u", D, dtype=complex), C.Field("U", D, dtype=complex)
    t1, t2 = C.Field("t1", D.without(z), dtype=complex), C.Field("t2", D.without(z), dtype=complex)
    s = C.Parameter("s", dtype=complex)
    m, a, d = C.Parameter("m", m_), C.Parameter("a", a_), C.Parameter("D", d_)
    pb = C.EVP([u, t1, t2], eigenvalue=s)
    pb.add_equation((m * (s * u + a * u) + u - d * C.lap(u) + U * u + C.lift(t1, b, -1) + C.lift(t2, b, -2), 0))
    pb.add_equation((C.interp(u, z, "left"), 0))
    pb.add_equation((C.interp(u, z, "right"), 0))
    harm = {1: lambda Z: b_ / 2 + 0 * Z + 0j, -1: lambda Z: b_ / 2 + 0 * Z + 0j}
    H = hill_from_plan(lambda ncc: discretize(pb, ncc={U: ncc[U]}), {U: harm}, Nh=6, omega=om,
                       label="diffusion, periodic coefficient")
    w = exponents_dense(H, finite_cap=1e4)
    strip = w[np.abs(w.imag) <= om / 2]
    lead = strip[np.argsort(-strip.real)][:3]
    exact = -(m_ * a_ + 1 + d_ * (np.arange(1, 4) * np.pi / 2) ** 2) / m_
    assert np.abs(lead.real - exact).max() < 1e-10
    assert np.abs(lead.imag).max() < 1e-10                     # real exponents (mod i w)
